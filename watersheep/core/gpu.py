"""GPU status via nvidia-smi."""
from __future__ import annotations
import shutil
import subprocess
import time
from typing import Optional

_cache: dict = {"t": 0.0, "v": None}


def status(max_age: float = 1.0) -> Optional[dict]:
    """Name, memory (MB) and utilisation, or None."""
    if time.time() - _cache["t"] < max_age:
        return _cache["v"]
    exe = shutil.which("nvidia-smi")
    v = None
    if exe:
        try:
            r = subprocess.run([exe, "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20)
            line = (r.stdout or "").strip().splitlines()[0]
            name, used, total, util = [x.strip() for x in line.split(",")]
            v = {"name": name, "used": int(float(used)), "total": int(float(total)),
                 "util": int(float(util)) if util.replace(".", "").isdigit() else 0}
        except Exception:
            v = None
    _cache.update(t=time.time(), v=v)
    return v


def free_mb() -> int:
    s = status(max_age=0)
    return max(0, s["total"] - s["used"]) if s else 0


def has_nvidia() -> bool:
    return status(max_age=0) is not None
