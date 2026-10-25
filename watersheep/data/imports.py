"""Strict local WaterSheep-record imports.

These files are read from the local machine at training time. Their contents are not
copied into the repository, and malformed lines fail closed instead of disappearing.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from typing import List

from .schema import validate

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def batch_name(path: str, index: int) -> str:
    p = Path(path).expanduser()
    stem = _SAFE.sub("-", p.stem).strip("-") or "records"
    return "import:%02d:%s" % (index, stem)


def fingerprint(path: str) -> str:
    p = Path(path).expanduser()
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(path: str, cfg) -> List[dict]:
    """Read and validate one local NDJSON file, failing on the first bad line."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError("record import does not exist: %s" % p)
    out: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                raise ValueError("%s:%d: invalid JSON: %s" % (p, lineno, e)) from e
            if not isinstance(row, dict):
                raise ValueError("%s:%d: record is not an object" % (p, lineno))
            row = dict(row)
            row.setdefault("source", "import:" + p.stem)
            row.setdefault("kind", "real")
            ok, why = validate(row, cfg.max_options)
            if not ok:
                raise ValueError("%s:%d: invalid WaterSheep record (%s)" % (p, lineno, why))
            out.append(row)
    if not out:
        raise ValueError("record import is empty: %s" % p)
    return out
