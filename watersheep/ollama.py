"""Ollama: install, pull the teacher model, locate its weights."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from . import interrupt
from .download import download
from .paths import P
from .util import LOG, human_bytes

SYSTEM_URL = "http://127.0.0.1:11434"
PRIVATE_PORT = 11534
WIN = os.name == "nt"


class OllamaError(RuntimeError):
    pass


def find_binary() -> Optional[str]:
    exe = shutil.which("ollama")
    if exe:
        return exe
    cands = [P.ollama_bin / ("ollama.exe" if WIN else "bin/ollama"), P.ollama_bin / "ollama"]
    if WIN:
        cands += [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
                  Path(os.environ.get("ProgramFiles", "")) / "Ollama" / "ollama.exe"]
    for c in cands:
        try:
            if c.exists():
                return str(c)
        except OSError:
            continue
    return None


def install() -> Optional[str]:
    """Portable Ollama in ./ollama_bin when none is installed."""
    exe = find_binary()
    if exe:
        return exe
    rel = "https://github.com/ollama/ollama/releases/latest/download/"
    P.ollama_bin.mkdir(parents=True, exist_ok=True)
    if WIN:
        name = "ollama-windows-amd64.zip"
    elif sys.platform == "darwin":
        name = "ollama-darwin.tgz"
    else:
        name = "ollama-linux-amd64.tgz"
    LOG.info("Ollama not found - downloading the portable build (%s, resumable)", name)
    arc = P.downloads / name
    if not download(rel + name, arc, name):
        return None
    LOG.info("unpacking %s into %s", name, P.ollama_bin)
    if name.endswith(".zip"):
        zipfile.ZipFile(arc).extractall(P.ollama_bin)
    else:
        import tarfile
        tarfile.open(arc).extractall(P.ollama_bin)
    exe = find_binary()
    if exe and not WIN:
        os.chmod(exe, 0o755)
    if exe:
        LOG.info("Ollama ready: %s", exe)
    return exe


def lib_dir(exe: str) -> Path:
    return Path(os.path.realpath(exe)).parent / "lib" / "ollama"


def llama_server_exe(exe: str) -> Optional[str]:
    name = "llama-server.exe" if WIN else "llama-server"
    p = lib_dir(exe) / name
    if p.exists():
        return str(p)
    return shutil.which(name)


def _get(url: str, timeout: int = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url: str, payload: dict, timeout: int = 600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def is_up(url: str) -> bool:
    try:
        _get(url + "/api/version", timeout=3)
        return True
    except Exception:
        return False


def kill_tree(proc) -> None:
    """Kill a process and its children."""
    if not proc or proc.poll() is not None:
        return
    try:
        if WIN:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
            for _ in range(30):
                if proc.poll() is not None:
                    return
                time.sleep(0.1)
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def detached_kwargs() -> dict:
    """Start children in their own process group."""
    if WIN:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {"start_new_session": True}


class Server:
    """The system Ollama, or a private one while pulling weights."""

    def __init__(self, auto_install: bool = True):
        self.proc = None
        self.url = SYSTEM_URL
        self.auto_install = auto_install

    def __enter__(self):
        if is_up(SYSTEM_URL):
            return self
        exe = find_binary() or (install() if self.auto_install else None)
        if not exe:
            raise OllamaError("Ollama is not installed and could not be installed "
                              "automatically - get it from https://ollama.com/download")
        self.url = "http://127.0.0.1:%d" % PRIVATE_PORT
        env = dict(os.environ, OLLAMA_HOST="127.0.0.1:%d" % PRIVATE_PORT)
        lf = open(P.logs / "ollama_server.log", "ab")
        self.proc = subprocess.Popen([exe, "serve"], env=env, stdout=lf, stderr=lf,
                                     **detached_kwargs())
        interrupt.on_exit(self._stop)
        for _ in range(90):
            if is_up(self.url):
                return self
            if interrupt.sleep(1):
                self._stop()
                raise interrupt.Interrupted("stopped while starting ollama")
        self._stop()
        raise OllamaError("ollama server did not start - see logs/ollama_server.log")

    def _stop(self):
        kill_tree(self.proc)
        self.proc = None

    def __exit__(self, *a):
        interrupt.forget(self._stop)
        self._stop()

    def tags(self) -> list:
        try:
            return [m.get("name", "") for m in _get(self.url + "/api/tags").get("models", [])]
        except Exception:
            return []

    def pull(self, model: str) -> None:
        names = self.tags()
        if model in names or model + ":latest" in names:
            return
        LOG.info("pulling %s through Ollama (resumable; first run only)", model)
        req = urllib.request.Request(self.url + "/api/pull",
                                     data=json.dumps({"model": model, "stream": True}).encode(),
                                     headers={"Content-Type": "application/json"})
        last = 0.0
        with urllib.request.urlopen(req, timeout=7200) as r:
            for raw in r:
                interrupt.check("pull stopped - it resumes on the next run")
                try:
                    ev = json.loads(raw)
                except Exception:
                    continue
                if "error" in ev:
                    raise OllamaError("pull failed: %s" % ev["error"])
                if ev.get("total") and time.time() - last > 5:
                    last = time.time()
                    LOG.info("  %s %s / %s", ev.get("status", ""),
                             human_bytes(ev.get("completed", 0)), human_bytes(ev["total"]))
        LOG.info("pulled %s", model)

    def unload_all(self) -> list:
        """Unload resident models."""
        out = []
        try:
            for m in _get(self.url + "/api/ps").get("models", []):
                name = m.get("name") or m.get("model")
                if name:
                    _post(self.url + "/api/generate", {"model": name, "keep_alive": 0}, 60)
                    out.append(name)
        except Exception:
            pass
        return out


def model_files(model: str) -> Optional[dict]:
    """GGUF path and default parameters of a pulled model."""
    root = Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")
    name, _, tag = model.partition(":")
    parts = name.split("/")
    if len(parts) == 1:
        path = ["registry.ollama.ai", "library", parts[0]]
    elif len(parts) == 2:
        path = ["registry.ollama.ai"] + parts
    else:
        path = parts
    mf = root.joinpath("manifests", *path, tag or "latest")
    if not mf.exists():
        return None
    out = {"gguf": None, "params": {}}
    try:
        layers = json.loads(mf.read_text(encoding="utf-8"))["layers"]
    except Exception:
        return None
    for layer in layers:
        blob = root / "blobs" / layer["digest"].replace(":", "-")
        if layer["mediaType"].endswith("image.model"):
            out["gguf"] = str(blob)
        elif layer["mediaType"].endswith("image.params"):
            try:
                out["params"] = json.loads(blob.read_text(encoding="utf-8"))
            except Exception:
                pass
    return out if out["gguf"] and os.path.exists(out["gguf"]) else None


def ensure_model(model: str, auto_install: bool = True) -> dict:
    files = model_files(model)
    if files:
        return files
    with Server(auto_install) as srv:
        srv.pull(model)
    files = model_files(model)
    if not files:
        raise OllamaError("model %s still missing after pull" % model)
    return files


def free_gpu(auto_install: bool = False) -> list:
    """Unload models an idle system Ollama keeps on the GPU."""
    if not is_up(SYSTEM_URL):
        return []
    srv = Server(auto_install)
    srv.url = SYSTEM_URL
    return srv.unload_all()
