"""Teacher LLM backends: llama-server, with Ollama as fallback."""
from __future__ import annotations
import json
import math
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

from ..core import gpu, interrupt
from . import ollama
from ..core.paths import P
from ..core.util import LOG, read_json, write_json

LETTERS = "ABCDEFGHIJ"
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]


def chat_prompt(system: str, user: str) -> str:
    """Chat prompt with thinking disabled."""
    return ("<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" + user +
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def _http(url: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class LlamaServer:
    name = "llama-server"

    def __init__(self, cfg, gguf: str, params: dict, ollama_exe: str):
        self.cfg = cfg
        self.gguf, self.params = gguf, params or {}
        self.exe = ollama.llama_server_exe(ollama_exe) if ollama_exe else None
        self.libdir = ollama.lib_dir(ollama_exe) if ollama_exe else None
        self.url = "http://127.0.0.1:%d" % cfg.llm_port
        self.proc = None
        self.slots = max(1, cfg.llm_slots or 4)
        self.backend_lib: Optional[str] = None
        self.lock = threading.Lock()
        self.crashes: List[float] = []
        self.logf = P.logs / "llama_server.log"

    def available(self) -> bool:
        return bool(self.exe and os.path.exists(self.exe) and os.path.exists(self.gguf))

    def alive(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def _backend_libs(self) -> List[Optional[str]]:
        """GPU backend libraries, best first."""
        if not self.libdir or not self.libdir.is_dir():
            return [None]

        def rank(d: Path):
            m = re.search(r"(\d+)", d.name)
            r = 0 if "cuda" in d.name else 1 if ("rocm" in d.name or "hip" in d.name) else 2
            return (r, -int(m.group(1)) if m else 0)

        libs = []
        for d in sorted((x for x in self.libdir.iterdir() if x.is_dir()), key=rank):
            for f in d.iterdir():
                if re.match(r"(lib)?ggml-(cuda|hip|rocm|vulkan|metal)\.(dll|so|dylib)$", f.name):
                    libs.append(str(f))
        return libs or [None]

    def _launch(self, lib: Optional[str], slots: int, ngl: int = 999) -> bool:
        self.stop()
        env = dict(os.environ)
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        extra = [str(self.libdir)] if self.libdir else []
        if lib:
            extra.append(os.path.dirname(lib))
            env["GGML_BACKEND_PATH"] = lib
        env[var] = os.pathsep.join(extra + [env.get(var, "")])
        ctx = self.cfg.llm_ctx_per_slot
        args = [self.exe, "--model", self.gguf, "--host", "127.0.0.1",
                "--port", str(self.cfg.llm_port), "--no-webui",
                "-c", str(slots * ctx), "-np", str(slots), "-ngl", str(ngl),
                "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                "-b", "1024", "-ub", "512"]
        lf = open(self.logf, "ab")
        lf.write(("\n==== %s launch slots=%d lib=%s\n" % (time.ctime(), slots, lib)).encode())
        lf.flush()
        self.proc = subprocess.Popen(args, env=env, stdout=lf, stderr=lf,
                                     **ollama.detached_kwargs())
        interrupt.on_exit(self.stop)
        for _ in range(240):
            if self.alive():
                self.slots = slots
                return True
            if self.proc.poll() is not None or interrupt.stopping():
                break
            time.sleep(0.5)
        self.stop()
        return False

    def stop(self) -> None:
        if self.proc is not None:
            ollama.kill_tree(self.proc)
            interrupt.forget(self.stop)
            self.proc = None
            time.sleep(0.5)

    def setup(self) -> bool:
        """Start on a GPU backend and fit the slot count to free memory."""
        if not self.available():
            return False
        weights_mb = os.path.getsize(self.gguf) // 2 ** 20
        ctx = self.cfg.llm_ctx_per_slot
        cache_key = "%s|%d|%s" % (os.path.basename(self.gguf), ctx,
                                  (gpu.status(0) or {}).get("total"))
        cache = read_json(P.state / "llm_slots.json", {}) or {}
        g0 = gpu.status(max_age=0)
        for lib in self._backend_libs():
            if interrupt.stopping():
                return False
            if not self._launch(lib, 2):
                continue
            g1 = gpu.status(max_age=0)
            on_gpu = not (g0 and g1) or (g1["used"] - g0["used"]) >= weights_mb // 3
            if not on_gpu:
                LOG.warning("llama-server: %s did not put the weights on the GPU - trying "
                            "the next backend", os.path.basename(os.path.dirname(lib or "")) or "built-in")
                continue
            self.backend_lib = lib
            if self.cfg.llm_slots:
                ok = self._launch(lib, self.cfg.llm_slots)
            else:
                ok = self._fit(lib, g0, g1, cache.get(cache_key))
            if not ok:
                ok = self._launch(lib, 2)
            if ok:
                cache[cache_key] = self.slots
                write_json(P.state / "llm_slots.json", cache)
                g = gpu.status(max_age=0)
                LOG.info("teacher: llama-server on %s with %d parallel slots x %d ctx "
                         "(GPU %s/%s MB)", os.path.basename(os.path.dirname(lib or "")) or "cpu",
                         self.slots, ctx, g and g["used"], g and g["total"])
                return True
        return False

    def _fit(self, lib, g0, g2, cached: Optional[int]) -> bool:
        """Largest slot count that fits in free memory."""
        cap = max(1, self.cfg.llm_max_slots)
        if cached:
            if self._launch(lib, min(cap, int(cached))):
                return True
        if not (g0 and g2):
            return self._launch(lib, min(cap, 8))
        probe = min(cap, 10)
        if not self._launch(lib, probe):
            return self._launch(lib, 2)
        g10 = gpu.status(max_age=0)
        per_slot = max(8.0, (g10["used"] - g2["used"]) / max(1, probe - 2))
        free = g10["total"] - g10["used"] - self.cfg.llm_vram_margin_mb
        target = int(min(cap, probe + max(0, free) // per_slot))
        LOG.info("teacher: ~%.0f MB per slot, %d MB free -> %d slots", per_slot,
                 max(0, g10["total"] - g10["used"]), target)
        while target > probe:
            if self._launch(lib, target):
                return True
            target = max(probe, int(target * 0.8))
        return self._launch(lib, probe)

    def _ensure(self) -> None:
        with self.lock:
            if self.alive() or interrupt.stopping():
                return
            self.crashes = [t for t in self.crashes if time.time() - t < 600] + [time.time()]
            slots = self.slots
            if len(self.crashes) >= 3 and slots > 1:
                slots = max(1, slots // 2)
                self.crashes = []
                LOG.warning("llama-server died 3 times in 10 min - halving to %d slots", slots)
            LOG.warning("llama-server is down - restarting")
            self._launch(self.backend_lib, slots)

    def complete(self, body: dict) -> dict:
        body = dict(body)
        body.setdefault("stop", STOP_TOKENS)
        body.setdefault("cache_prompt", True)
        last = None
        for attempt in range(8):
            interrupt.check()
            try:
                return _http(self.url + "/completion", body, self.cfg.llm_timeout_s)
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (400, 422):
                    raise
            except Exception as e:
                last = e
            if interrupt.sleep(min(20, 2 * (attempt + 1))):
                raise interrupt.Interrupted()
            self._ensure()
        raise RuntimeError("llama-server failed repeatedly: %r" % (last,))


class OllamaBackend:
    """Fallback backend, one request at a time."""
    name = "ollama"

    def __init__(self, cfg, model: str):
        self.cfg, self.model = cfg, model
        self.url = ollama.SYSTEM_URL
        self.slots = 1
        self._srv = None

    def setup(self) -> bool:
        if not ollama.is_up(self.url):
            self._srv = ollama.Server(self.cfg.auto_install).__enter__()
            self.url = self._srv.url
        return True

    def stop(self) -> None:
        if self._srv:
            self._srv.__exit__()
            self._srv = None

    def complete(self, body: dict) -> dict:
        opts = {"num_predict": body.get("n_predict", 256),
                "temperature": body.get("temperature", 0.7),
                "num_ctx": self.cfg.llm_ctx_per_slot}
        for k in ("top_p", "top_k", "seed", "presence_penalty"):
            if k in body:
                opts[k] = body[k]
        req = {"model": self.model, "prompt": body["prompt"], "raw": True, "stream": False,
               "options": dict(opts, stop=STOP_TOKENS), "keep_alive": "30m"}
        if body.get("json_schema"):
            req["format"] = body["json_schema"]
        if body.get("n_probs"):
            req["logprobs"] = True
            req["top_logprobs"] = int(body["n_probs"])
        for attempt in range(8):
            interrupt.check()
            try:
                r = _http(self.url + "/api/generate", req, self.cfg.llm_timeout_s)
                out = {"content": r.get("response", ""), "tokens_predicted": r.get("eval_count", 0)}
                lp = r.get("logprobs")
                if isinstance(lp, list) and lp:
                    out["completion_probabilities"] = lp
                return out
            except urllib.error.HTTPError as e:
                if e.code == 400 and ("logprobs" in req or "format" in req):
                    req.pop("logprobs", None)
                    req.pop("top_logprobs", None)
                    continue
                if interrupt.sleep(min(20, 3 * (attempt + 1))):
                    raise interrupt.Interrupted()
            except Exception:
                if interrupt.sleep(min(20, 3 * (attempt + 1))):
                    raise interrupt.Interrupted()
        raise RuntimeError("ollama generate failed repeatedly")


class Teacher:
    """JSON generation and option probabilities."""

    def __init__(self, backend):
        self.b = backend
        self.tokens = 0
        self.calls = 0
        self._lock = threading.Lock()

    @property
    def workers(self) -> int:
        return max(2, self.b.slots + 2)

    def _count(self, r: dict) -> None:
        with self._lock:
            self.tokens += int(r.get("tokens_predicted") or 0)
            self.calls += 1

    def generate_json(self, system: str, user: str, schema: dict, *, max_tokens: int,
                      temperature: float, top_p: float, seed: int):
        body = {"prompt": chat_prompt(system, user), "n_predict": max_tokens,
                "temperature": temperature, "top_p": top_p, "top_k": 40, "seed": seed,
                "json_schema": schema}
        r = self.b.complete(body)
        self._count(r)
        text = (r.get("content") or "").strip()
        return loose_json(text), text

    def option_probs(self, system: str, user: str, labels: List[str]) -> Optional[List[float]]:
        """Probability of each single-token label as the reply."""
        body = {"prompt": chat_prompt(system, user), "n_predict": 1, "temperature": 0.0,
                "n_probs": 20, "seed": 0}
        r = self.b.complete(body)
        self._count(r)
        return label_probs(r, labels)


def label_probs(r: dict, labels: List[str]) -> Optional[List[float]]:
    mass = {lab: 0.0 for lab in labels}
    cps = r.get("completion_probabilities") or []
    if cps:
        first = cps[0]
        tops = first.get("top_logprobs") or first.get("probs") or []
        for t in tops:
            tok = str(t.get("token", t.get("tok_str", ""))).strip()
            if "logprob" in t:
                p = math.exp(float(t["logprob"]))
            else:
                p = float(t.get("prob", 0.0))
            tok = tok.rstrip(").:").lstrip("(")
            if tok in mass:
                mass[tok] += p
    total = sum(mass.values())
    if total <= 1e-6:
        c = (r.get("content") or "").strip().lstrip("(")[:1]
        if c in mass:
            return [1.0 if lab == c else 0.0 for lab in labels]
        return None
    return [mass[lab] / total for lab in labels]


def loose_json(text: str):
    """Parse JSON from model output."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    start = t.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = re.sub(r",\s*([}\]])", r"\1", t[start:i + 1])
                try:
                    return json.loads(chunk)
                except Exception:
                    return None
    return None


class _NullBackend:
    name, slots = "fake", 4

    def stop(self) -> None:
        pass


class FakeTeacher(Teacher):
    """Deterministic teacher for the offline self-test."""

    def __init__(self):
        super().__init__(_NullBackend())
        self.answers: dict = {}
        self.lock = threading.Lock()

    def generate_json(self, system, user, schema, *, max_tokens, temperature, top_p, seed):
        import random as _r
        rng = _r.Random(seed)
        props = schema["properties"]
        n = rng.randint(100, 999)
        q = "Case %d-%d: which outcome applies here?" % (seed % 100000, n)
        state = ("Ticket %d from customer %d reports issue code %d in region %d, "
                 "logged at %02d:%02d with priority marker P%d."
                 % (seed % 99991, n, rng.randint(1, 999), rng.randint(1, 50),
                    rng.randint(0, 23), rng.randint(0, 59), rng.randint(1, 4)))
        obj = {"state": state, "question": q, "why": "because the context says so"}
        if "state" not in props:
            obj = _fake_fill(schema, rng)
            self._count({"tokens_predicted": 40})
            return obj, json.dumps(obj)
        if "options" in props and props["answer"].get("type") == "array":
            k = props["options"]["minItems"]
            opts = ["tag %s %d" % (w, n) for w in ("alpha", "bravo", "charlie", "delta",
                                                     "echo", "foxtrot", "golf", "hotel")[:k]]
            on = sorted(rng.sample(range(k), props["answer"]["minItems"]))
            obj.update(options=opts, answer=on)
            right = {opts[i] for i in on}
        elif "options" in props:
            k = props["options"]["minItems"]
            opts = ["outcome %s %d" % (w, n) for w in ("alpha", "bravo", "charlie", "delta",
                                                         "echo", "foxtrot", "golf", "hotel")[:k]]
            a = rng.randrange(k)
            obj.update(options=opts, answer=a)
            right = opts[a]
        elif props["answer"].get("type") == "integer":
            m = re.search(r"MUST be (\d+)", user)
            obj["answer"] = int(m.group(1))
            right = str(obj["answer"])
        else:
            m = re.search(r'MUST be "(yes|no)"', user)
            obj["answer"] = m.group(1)
            right = obj["answer"]
        with self.lock:
            self.answers[q] = right
        self._count({"tokens_predicted": 120})
        return obj, json.dumps(obj)

    def option_probs(self, system, user, labels):
        self._count({"tokens_predicted": 1})
        if system.startswith("You audit"):
            return [0.0, 0.0, 0.05, 0.15, 0.8]
        m = re.search(r"^QUESTION: (.*)$", user, re.M)
        right = self.answers.get(m.group(1).strip()) if m else None
        if right is None:
            return [1.0 / len(labels)] * len(labels)
        if isinstance(right, set):
            asked = re.search(r'Does option [A-J]\) "(.*)" apply\?', user)
            yes = bool(asked and asked.group(1) in right)
            return [0.9, 0.1] if yes else [0.1, 0.9]
        shown = dict(re.findall(r"^([A-J])\) (.*)$", user, re.M))
        hit = next((lab for lab in labels if shown.get(lab, lab) == right), None)
        if hit is None:
            return [1.0 / len(labels)] * len(labels)
        rest = 0.1 / max(1, len(labels) - 1)
        return [0.9 if lab == hit else rest for lab in labels]


def _fake_fill(schema: dict, rng):
    t = schema.get("type")
    if t == "object":
        return {k: _fake_fill(v, rng) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        n = schema.get("minItems", 2)
        return [_fake_fill(schema.get("items", {"type": "string"}), rng) + " %d" % i for i in range(n)]
    return "short description %d" % rng.randint(1, 999)


_TEACHER: Optional[Teacher] = None


def get_teacher(cfg) -> Teacher:
    """Start the teacher backend once and return it."""
    global _TEACHER
    if _TEACHER is not None:
        return _TEACHER
    if cfg.llm_backend == "fake":
        _TEACHER = FakeTeacher()
        return _TEACHER
    files = ollama.ensure_model(cfg.llm_model, cfg.auto_install)
    exe = ollama.find_binary()
    if cfg.free_ollama_vram:
        freed = ollama.free_gpu()
        if freed:
            LOG.info("unloaded %s from the system Ollama to free the GPU", ", ".join(freed))
    backend = None
    if cfg.llm_backend in ("auto", "llama") and exe:
        srv = LlamaServer(cfg, files["gguf"], files["params"], exe)
        if srv.setup():
            backend = srv
        elif cfg.llm_backend == "llama":
            raise RuntimeError("llama-server could not start - see logs/llama_server.log")
        else:
            LOG.warning("llama-server unavailable - falling back to Ollama (much slower: "
                        "one request at a time)")
    if backend is None:
        backend = OllamaBackend(cfg, cfg.llm_model)
        backend.setup()
    _TEACHER = Teacher(backend)
    return _TEACHER


def shutdown() -> None:
    """Stop the teacher and free the GPU."""
    global _TEACHER
    if _TEACHER is not None:
        try:
            _TEACHER.b.stop()
        finally:
            _TEACHER = None
