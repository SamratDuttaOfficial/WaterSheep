"""Resumable downloads."""
from __future__ import annotations
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from ..core import interrupt
from ..core.util import LOG, human_bytes, human_time

UA = {"User-Agent": "watersheep"}


def remote_size(url: str) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def _watch(path: Path, total: int, label: str, stop: threading.Event) -> None:
    t0 = time.time()
    start = path.stat().st_size if path.exists() else 0
    while not stop.wait(20):
        try:
            got = path.stat().st_size
        except OSError:
            continue
        rate = (got - start) / max(1e-6, time.time() - t0)
        eta = (total - got) / rate if (total and rate > 0) else 0
        LOG.info("  %s %s%s  %s/s%s", label, human_bytes(got),
                 (" / " + human_bytes(total)) if total else "", human_bytes(rate),
                 ("  eta " + human_time(eta)) if eta else "")


def download(url: str, dest: Path, label: str = "download", attempts: int = 20) -> bool:
    """Download `url` to `dest`, resuming partial files."""
    dest = Path(dest)
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    total = remote_size(url)
    if total:
        LOG.info("  %s is %s", label, human_bytes(total))
    curl = shutil.which("curl")
    for attempt in range(1, attempts + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        if total and have >= total:
            break
        if interrupt.stopping():
            LOG.info("  %s paused at %s - resumes on the next run", label, human_bytes(have))
            return False
        finished = False
        try:
            if curl:
                stop = threading.Event()
                threading.Thread(target=_watch, args=(tmp, total, label, stop),
                                 daemon=True).start()
                try:
                    rc = subprocess.run([curl, "-fL", "-C", "-", "--retry", "3",
                                         "--connect-timeout", "60", "--no-progress-meter",
                                         "-o", str(tmp), url], check=False, timeout=7200).returncode
                finally:
                    stop.set()
                finished = rc == 0
            else:
                hdr = dict(UA)
                if have:
                    hdr["Range"] = "bytes=%d-" % have
                with urllib.request.urlopen(urllib.request.Request(url, headers=hdr),
                                            timeout=120) as r:
                    mode = "ab" if have and r.status == 206 else "wb"
                    with open(tmp, mode) as f:
                        while not interrupt.stopping():
                            chunk = r.read(1 << 20)
                            if not chunk:
                                finished = True
                                break
                            f.write(chunk)
        except Exception as e:
            LOG.warning("  %s attempt %d failed: %s", label, attempt, e)
        now = tmp.stat().st_size if tmp.exists() else 0
        if (total and now >= total) or (not total and finished and now > 0):
            break
        if now <= have and interrupt.sleep(5):
            return False
    final = tmp.stat().st_size if tmp.exists() else 0
    if not final or (total and final < total):
        LOG.error("  %s incomplete (%s of %s) - re-run to continue", label,
                  human_bytes(final), human_bytes(total) if total else "?")
        return False
    tmp.replace(dest)
    LOG.info("  %s complete: %s", label, human_bytes(dest.stat().st_size))
    return True
