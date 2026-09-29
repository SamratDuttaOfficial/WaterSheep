"""Small helpers: IO, hashing, formatting, logging."""
from __future__ import annotations
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

LOG = logging.getLogger("watersheep")
_IO_LOCK = threading.Lock()


def setup_logging(logfile: Path, verbose: bool = False) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:  # Windows consoles default to cp1252
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logfile.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname).1s %(message)s", "%H:%M:%S"))
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname).1s %(name)s %(message)s"))
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(sh)
    root.addHandler(fh)
    for noisy in ("urllib3", "filelock", "datasets", "huggingface_hub", "matplotlib",
                  "PIL", "fsspec", "httpx", "httpcore", "kagglehub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def atomic_write(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp%d" % os.getpid())
    f = open(tmp, "wb") if isinstance(data, bytes) else \
        open(tmp, "w", encoding="utf-8", newline="\n")
    try:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    finally:
        f.close()
    os.replace(tmp, path)


def write_json(path, obj) -> None:
    atomic_write(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        LOG.warning("corrupt json %s - ignoring", p)
        return default


def append_jsonl(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
    with _IO_LOCK, open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line)
        f.flush()


def write_jsonl(path, rows) -> int:
    """Atomically write rows; returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp%d" % os.getpid())
    n = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            n += 1
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return n


def iter_jsonl(path):
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def read_jsonl(path) -> list:
    return list(iter_jsonl(path))


def sha1(obj) -> str:
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def stable_int(text: str) -> int:
    """Deterministic 64-bit hash."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^\w\s]")


def norm_text(s: str) -> str:
    return _WS.sub(" ", _NONWORD.sub(" ", (s or "").lower())).strip()


def clean_text(s, max_chars: int = 0) -> str:
    """Normalise whitespace, drop control characters, optionally truncate."""
    if s is None:
        return ""
    s = str(s).replace("\r\n", "\n").replace("\r", "\n")
    s = "".join(ch for ch in s if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    if max_chars and len(s) > max_chars:
        cut = s[:max_chars]
        sp = cut.rfind(" ")
        s = (cut[:sp] if sp > max_chars * 0.8 else cut).rstrip() + " ..."
    return s


def human_time(s: float) -> str:
    s = int(max(0, s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600:02d}h"


def human_bytes(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def human_num(n: float) -> str:
    for u in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{u}".replace(".0", "")
        n /= 1000
    return f"{n:.1f}T"


def rmtree(p) -> None:
    shutil.rmtree(Path(p), ignore_errors=True)


def dir_size(p) -> int:
    p = Path(p)
    if not p.exists():
        return 0
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


class RateMeter:
    """Moving throughput for log lines."""

    def __init__(self, window: float = 120.0):
        self.window = window
        self.events: list = []

    def add(self, n: float = 1.0) -> None:
        t = time.time()
        self.events.append((t, n))
        while self.events and t - self.events[0][0] > self.window:
            self.events.pop(0)

    def rate(self) -> float:
        if len(self.events) < 2:
            return 0.0
        dt = self.events[-1][0] - self.events[0][0]
        return sum(n for _, n in self.events[1:]) / dt if dt > 0 else 0.0


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
