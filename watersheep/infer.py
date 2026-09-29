"""Model export and inference.

    ws = WaterSheep.load()                 # newest export
    ws = WaterSheep.load("owner/name")     # or an export folder, an export name, a Hugging Face repo
    ws.ask(request)                        # {state, questions} -> {model, answers, usage}
    ws.decide(state, question, options)
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from .metrics import concentration, sigmoid, softmax
from .model import CharTok, Tok, WaterSheepNet, encode_records, encoder_from_dir
from .paths import P
from .schema import YESNO, is_digit_scale, noul_question
from .util import LOG, now_ts, write_json

TYPES_IN = {"noul": "binary", "binary": "binary", "yes/no": "binary", "boolean": "binary",
            "choice": "choice", "score": "score", "multi": "multi", "multi_choice": "multi",
            "multilabel": "multi", "multi-label": "multi"}
TYPES_OUT = {"binary": "noul", "choice": "choice", "score": "score", "multi": "multi"}
MAX_CHOICES, MAX_LEVELS = 255, 10
HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+(@[\w./-]+)?$")   # owner/name[@revision]
HUB_FILES = ["watersheep.json", "model.safetensors", "encoder/*", "tokenizer/*"]


def export(cfg, run_dir: Path, tok, encoder_spec, temps: dict, metrics: dict,
           run_id: str, multi_threshold: float = 0.5) -> Path:
    from safetensors.torch import save_file
    from .trainer import load_ckpt
    blob = load_ckpt(run_dir / "best.pt")
    d = P.export / ("watersheep_%s" % now_ts())
    (d / "encoder").mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().contiguous().cpu() for k, v in blob["model"].items()}
    save_file(sd, str(d / "model.safetensors"))
    if isinstance(encoder_spec, dict):
        from transformers import ModernBertConfig
        ModernBertConfig(**encoder_spec).save_pretrained(str(d / "encoder"))
    else:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(encoder_spec).save_pretrained(str(d / "encoder"))
    is_char = isinstance(tok, CharTok)
    if not is_char:
        tok.save(d / "tokenizer")
    write_json(d / "watersheep.json", {
        "format": 2, "name": d.name.replace("watersheep_", "watersheep-"),
        "run_id": run_id, "encoder": str(encoder_spec),
        "tokenizer": "char" if is_char else "hf",
        "max_len": cfg.max_len, "max_question_tokens": cfg.max_question_tokens,
        "max_option_tokens": cfg.max_option_tokens, "max_options": cfg.max_options,
        "head_layers": cfg.head_layers, "temperatures": temps, "metrics": metrics,
        "multi_threshold": multi_threshold, "created": now_ts()})
    (P.export / "LATEST").write_text(d.name, encoding="utf-8")
    return d


def latest_export() -> Optional[Path]:
    f = P.export / "LATEST"
    if f.exists():
        d = P.export / f.read_text(encoding="utf-8").strip()
        if d.exists():
            return d
    dirs = sorted(P.export.glob("watersheep_*"))
    return dirs[-1] if dirs else None


def find_export(path=None) -> Optional[Path]:
    """A local export: a folder, a folder under the project root, or an export name."""
    if not path:
        return latest_export()
    p = Path(path).expanduser()
    for d in (p, P.root / p, P.export / p):
        if (d / "watersheep.json").exists():
            return d
    return None


def hub_download(ref: str) -> Path:
    """An export on the Hugging Face Hub ("owner/name" or "owner/name@revision"), cached locally."""
    repo, _, rev = ref.partition("@")
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo, revision=rev or None, allow_patterns=HUB_FILES))
    except ImportError as e:
        raise FileNotFoundError("loading %s from Hugging Face needs huggingface_hub "
                                "(pip install huggingface_hub)" % ref) from e
    except Exception as e:
        why = {"RepositoryNotFoundError": "no such repository (private ones need a login or HF_TOKEN)",
               "GatedRepoError": "the repository is gated - accept its terms and log in",
               "RevisionNotFoundError": "no revision %r" % rev,
               "LocalEntryNotFoundError": "offline and not in the local cache"}.get(type(e).__name__)
        raise FileNotFoundError("could not get %s from Hugging Face: %s" % (ref, why or e)) from e


def infer_type(options: Optional[Sequence[str]]) -> str:
    if not options:
        return "binary"
    low = [str(o).strip().lower() for o in options]
    if low == YESNO:
        return "binary"
    if is_digit_scale(low):
        v = [int(o) for o in low]
        if v == list(range(v[0], v[0] + len(v))):
            return "score"
    return "choice"


def render_state(state) -> str:
    """State (text, object or list) as text."""
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, (list, tuple)):
        return "\n".join(render_state(x) for x in state)
    if isinstance(state, dict):
        return "\n".join("%s: %s" % (k, v if isinstance(v, (str, int, float, bool)) or v is None
                                     else json.dumps(v, ensure_ascii=False)) for k, v in state.items())
    return str(state)


def _labels(criteria):
    """criteria -> [(key, option text)]"""
    if isinstance(criteria, dict):
        return [(str(k), "%s: %s" % (k, v) if str(v or "").strip() else str(k))
                for k, v in criteria.items()]
    return [(str(k), str(k)) for k in (criteria or [])]


class WaterSheep:
    def __init__(self, net, tok, meta: dict, device, name: str = "watersheep"):
        self.net, self.tok, self.meta, self.device, self.name = net, tok, meta, device, name
        self.temps = meta.get("temperatures") or {}
        self.threshold = float(meta.get("multi_threshold", 0.5))
        self.cap = max(2, min(10, int(meta.get("max_options") or 10)))

    @classmethod
    def load(cls, path=None, device: Optional[str] = None) -> "WaterSheep":
        """None = the newest local export; otherwise an export folder, an export name, or a
        Hugging Face repo id ("owner/name", optionally "owner/name@revision")."""
        from safetensors.torch import load_file
        d = find_export(path)
        hub = d is None and bool(path) and HUB_ID.match(str(path)) is not None
        if hub:
            d = hub_download(str(path))
        if not d or not (d / "watersheep.json").exists():
            if path:
                raise FileNotFoundError("no WaterSheep model at %r - give an export folder, an export "
                                        "name or a Hugging Face repo id (owner/name)" % str(path))
            raise FileNotFoundError("no exported WaterSheep model found - run `python run.py`, "
                                    "or load a Hugging Face repo id (owner/name)")
        meta = json.loads((d / "watersheep.json").read_text(encoding="utf-8"))
        tok = CharTok() if meta.get("tokenizer") == "char" else Tok.load(str(d / "tokenizer"))
        net = WaterSheepNet(encoder_from_dir(d / "encoder"), meta.get("head_layers", 1), 0.0)
        net.load_state_dict(load_file(str(d / "model.safetensors")))
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        net.to(dev).eval()
        if meta.get("name"):
            name = str(meta["name"])
        elif hub:   # the snapshot folder is a commit hash, so name it after the export
            name = "watersheep-%s" % meta["created"] if meta.get("created") else str(path).replace("/", "--")
        else:
            name = d.name.replace("watersheep_", "watersheep-")
        return cls(net, tok, meta, dev, name)

    @torch.no_grad()
    def _logits(self, recs: List[dict]):
        """Logits per record and the input token count."""
        m = self.meta
        enc = list(encode_records(self.tok, recs, m["max_len"], m["max_question_tokens"],
                                  m["max_option_tokens"]))
        out, tokens = [], 0
        for i in range(0, len(enc), 64):
            part = enc[i:i + 64]
            T, K, B = max(len(a) for a, _ in part), max(len(p) for _, p in part), len(part)
            pad = getattr(self.tok, "pad", 0)
            x = torch.full((B, T), pad, dtype=torch.long)
            am = torch.zeros((B, T), dtype=torch.long)
            op = torch.zeros((B, K), dtype=torch.long)
            om = torch.zeros((B, K), dtype=torch.bool)
            for b, (ids, pos) in enumerate(part):
                x[b, :len(ids)] = torch.tensor(ids)
                am[b, :len(ids)] = 1
                op[b, :len(pos)] = torch.tensor(pos)
                om[b, :len(pos)] = True
                tokens += len(ids)
            dt = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() \
                else torch.float32
            with torch.autocast(self.device.type, dtype=dt, enabled=self.device.type == "cuda"):
                z = self.net(x.to(self.device), am.to(self.device), op.to(self.device),
                             om.to(self.device)).float().cpu().numpy()
            for (ids, pos), r, zz in zip(part, recs[i:i + 64], z):
                out.append(zz[:len(pos)] if len(pos) == len(r["options"]) else None)
        return out, tokens

    def _probs(self, t: str, state: str, question: str, opts: List[str], size: int = 0):
        """Probabilities over any number of options."""
        T = self.temps.get(t, 1.0)
        n = len(opts)
        size = min(size or self.cap, n)
        tok = 0
        while True:
            if n <= size:
                (z,), k = self._logits([{"type": t, "state": state, "question": question, "options": opts}])
                tok += k
                if z is not None:
                    return (sigmoid(z, T) if t == "multi" else softmax(z, T)), tok
            else:
                groups = [list(range(k, min(n, k + size))) for k in range(0, n, size)]
                zs, k = self._logits([{"type": t, "state": state, "question": question,
                                       "options": [opts[i] for i in g]} for g in groups])
                tok += k
                if all(z is not None for z in zs):
                    break
            if size <= 2:
                raise ValueError("the options do not fit in the model's input - shorten them")
            size = max(2, min(size - 1, 8) if size > 8 else size // 2)
        if t == "multi":
            p = np.zeros(n)
            for g, z in zip(groups, zs):
                p[g] = sigmoid(z, T)
            return p, tok
        keep = max(1, size // len(groups))
        local, finalists = {}, []
        for g, z in zip(groups, zs):
            pg = softmax(z, T)
            local.update({i: pg[k] for k, i in enumerate(g)})
            finalists += [g[k] for k in np.argsort(-pg)[:keep]]
        pf, tok2 = self._probs(t, state, question, [opts[i] for i in finalists], size)
        p = np.zeros(n)
        for g in groups:
            top = max((i for i in g if i in finalists), key=lambda i: local[i])
            scale = pf[finalists.index(top)] / max(1e-12, local[top])
            for i in g:
                p[i] = local[i] * scale
        for k, i in enumerate(finalists):
            p[i] = pf[k]
        return p / p.sum(), tok + tok2

    def ask(self, request: dict) -> dict:
        """Answer {state, questions} with {model, answers, usage}."""
        state = render_state(request.get("state"))
        answers, tokens = {}, 0
        for name, q in (request.get("questions") or {}).items():
            try:
                a, n = self._answer(state, q)
                tokens += n
            except (ValueError, KeyError, TypeError) as e:
                a = {"type": str(q.get("type", "?")) if isinstance(q, dict) else "?", "error": str(e)}
            answers[name] = a
        return {"model": self.name, "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0}}

    def _answer(self, state: str, q: dict):
        t = TYPES_IN.get(str(q.get("type", "")).lower())
        if t is None:
            raise ValueError("unknown question type %r (noul, choice, score or multi)" % q.get("type"))
        instr = str(q.get("instructions") or q.get("question") or "").strip()
        if not instr:
            raise ValueError("instructions are required")
        crit = q.get("criteria")
        r4 = lambda x: round(float(x), 4)
        if t == "binary":
            c = crit if isinstance(crit, dict) else {}
            question = noul_question(instr, str(c.get("true", c.get("yes", "")) or ""),
                                     str(c.get("false", c.get("no", "")) or ""))
            p, n = self._probs("binary", state, question, list(YESNO))
            return {"type": "noul", "noul": r4(p[0])}, n
        if t == "score":
            levels = list(crit.values()) if isinstance(crit, dict) else list(crit or [])
            levels = [str(x) for x in levels]
            if not 2 <= len(levels) <= MAX_LEVELS:
                raise ValueError("score needs 2-%d levels in criteria" % MAX_LEVELS)
            p, n = self._probs("score", state, instr, levels)
            return {"type": "score", "score": r4(sum(i * v for i, v in enumerate(p))),
                    "confidence": r4(concentration(p)),
                    "legend": {str(i): l for i, l in enumerate(levels)},
                    "probabilities": {str(i): r4(v) for i, v in enumerate(p)}}, n
        labels = _labels(crit)
        if not 2 <= len(labels) <= MAX_CHOICES:
            raise ValueError("%s needs 2-%d labels in criteria" % (t, MAX_CHOICES))
        if len({k for k, _ in labels}) != len(labels):
            raise ValueError("labels must be unique")
        p, n = self._probs(t, state, instr, [text for _, text in labels])
        probs = {k: r4(v) for (k, _), v in zip(labels, p)}
        if t == "multi":
            thr = float(q.get("threshold", self.threshold))
            return {"type": "multi", "choices": [k for k, v in probs.items() if v >= thr],
                    "confidence": r4(np.mean([max(v, 1 - v) for v in p])),
                    "probabilities": probs}, n
        best = max(probs, key=probs.get)
        return {"type": "choice", "choice": best, "confidence": r4(concentration(p)),
                "probabilities": probs}, n

    def decide_many(self, items: List[dict]) -> List[dict]:
        """Answer items {state, question, options, type}."""
        out = []
        for it in items:
            opts = [str(o) for o in (it.get("options") or YESNO)]
            t = TYPES_IN.get(str(it.get("type") or "").lower()) or infer_type(opts)
            if t == "binary":
                opts = list(YESNO)
            try:
                p, _ = self._probs(t, it.get("state") or "", it["question"], opts)
            except ValueError as e:
                out.append({"error": str(e)})
                continue
            res = {"type": t, "probs": {o: float(v) for o, v in zip(opts, p)}}
            if t == "multi":
                res.update(answer=[o for o, v in zip(opts, p) if v >= self.threshold],
                           confidence=float(np.mean([max(v, 1 - v) for v in p])))
            else:
                i = int(np.argmax(p))
                res.update(answer=opts[i], index=i, confidence=float(p[i]))
            if t == "binary":
                res["p_yes"] = float(p[0])
            elif t == "score":
                res["expected"] = float(sum(k * v for k, v in enumerate(p)))
                if is_digit_scale(opts):
                    res["expected"] += int(opts[0])
            out.append(res)
        return out

    def decide(self, state: str, question: str, options: Optional[Sequence[str]] = None,
               type: Optional[str] = None) -> dict:
        return self.decide_many([{"state": state, "question": question, "options": options,
                                  "type": type}])[0]
