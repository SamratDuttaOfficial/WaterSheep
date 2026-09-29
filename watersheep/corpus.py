"""Records and sources excluded in the web UI."""
from __future__ import annotations
import threading
from typing import Dict, Iterable

from .paths import P
from .util import read_json, sha1, write_json

_LOCK = threading.Lock()


def _file():
    return P.state / "corpus_edits.json"


def load() -> dict:
    d = read_json(_file(), {}) or {}
    return {"deleted": dict(d.get("deleted") or {}), "disabled": sorted(set(d.get("disabled") or []))}


def _save(d: dict) -> None:
    write_json(_file(), d)


def delete(items: Dict[str, str]) -> int:
    """Exclude records; `items` maps record id to source."""
    with _LOCK:
        d = load()
        new = {k: v for k, v in items.items() if k not in d["deleted"]}
        d["deleted"].update(new)
        _save(d)
        return len(new)


def restore(ids: Iterable[str] = (), source: str = "") -> int:
    with _LOCK:
        d = load()
        drop = set(ids) | {k for k, v in d["deleted"].items() if source and v == source}
        n = sum(1 for k in drop if d["deleted"].pop(k, None) is not None)
        _save(d)
        return n


def set_disabled(source: str, off: bool) -> None:
    with _LOCK:
        d = load()
        s = set(d["disabled"])
        (s.add if off else s.discard)(source)
        d["disabled"] = sorted(s)
        _save(d)


def digest() -> str:
    d = load()
    if not d["deleted"] and not d["disabled"]:
        return ""
    return sha1({"del": sorted(d["deleted"]), "off": d["disabled"]})


def excluder():
    """Predicate for records excluded from training."""
    d = load()
    deleted, off = set(d["deleted"]), set(d["disabled"])

    def skip(r: dict) -> bool:
        return (r.get("id") in deleted or r.get("source") in off or
                (r.get("kind") == "synth" and "synth" in off))
    return skip
