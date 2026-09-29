"""Stage manifest and artifact registry."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import P
from .util import LOG, now_iso, read_json, write_json


class Manifest:
    def __init__(self) -> None:
        self.d: Dict[str, Any] = read_json(P.manifest, {}) or {}
        self.d.setdefault("stages", {})
        self.d.setdefault("created", now_iso())

    def save(self) -> None:
        self.d["updated"] = now_iso()
        write_json(P.manifest, self.d)

    def entry(self, stage: str) -> dict:
        return self.d["stages"].get(stage, {})

    def stale_reason(self, stage: str, hash_: str) -> Optional[str]:
        e = self.entry(stage)
        if not e:
            return "never run"
        if e.get("status") != "done":
            return "status=" + str(e.get("status"))
        if e.get("hash") != hash_:
            return "inputs changed"
        return None

    def mark_running(self, stage: str, hash_: str) -> None:
        prev = self.entry(stage)
        self.d["stages"][stage] = {"status": "running", "hash": hash_,
                                   "started": now_iso(), "meta": prev.get("meta", {})}
        self.save()

    def mark_done(self, stage: str, hash_: str, meta: dict | None = None) -> None:
        prev = self.entry(stage)
        self.d["stages"][stage] = {"status": "done", "hash": hash_,
                                   "started": prev.get("started"), "finished": now_iso(),
                                   "meta": meta or {}}
        self.save()

    def mark_failed(self, stage: str, err: str) -> None:
        e = self.d["stages"].setdefault(stage, {})
        e.update(status="failed", error=str(err)[:500], failed_at=now_iso())
        self.save()

    def mark_stopped(self, stage: str) -> None:
        """Record a stopped stage (not a failure)."""
        e = self.d["stages"].setdefault(stage, {})
        e.update(status="stopped", stopped_at=now_iso())
        e.pop("error", None)
        self.save()

    def clear(self, stage: str) -> None:
        self.d["stages"].pop(stage, None)
        self.d.pop("progress::" + stage, None)
        self.save()

    def progress(self, stage: str) -> dict:
        return self.d.setdefault("progress::" + stage, {})

    def set_progress(self, stage: str, key: str, value: Any) -> None:
        self.progress(stage)[key] = value
        self.save()


class Registry:
    """Training runs and exports."""

    def __init__(self) -> None:
        self.d: Dict[str, Any] = read_json(P.registry, {}) or {}
        self.d.setdefault("items", [])

    def save(self) -> None:
        write_json(P.registry, self.d)

    def add(self, id_: str, kind: str, path, parent: str | None = None,
            metrics: dict | None = None, note: str = "") -> dict:
        item = {"id": id_, "kind": kind, "path": str(path), "parent": parent,
                "created": now_iso(), "metrics": metrics or {}, "note": note,
                "active": False}
        self.d["items"] = [i for i in self.d["items"] if i["id"] != id_]
        self.d["items"].append(item)
        self.save()
        return item

    def update(self, id_: str, **kw) -> None:
        for i in self.d["items"]:
            if i["id"] == id_:
                i.update(kw)
        self.save()

    def get(self, id_: str) -> Optional[dict]:
        return next((i for i in self.d["items"] if i["id"] == id_), None)

    def by_kind(self, kind: str) -> List[dict]:
        return sorted([i for i in self.d["items"] if i["kind"] == kind],
                      key=lambda x: x["created"])

    def active(self, kind: str) -> Optional[dict]:
        items = self.by_kind(kind)
        for i in items:
            if i.get("active"):
                return i
        return items[-1] if items else None

    def promote(self, id_: str) -> Optional[dict]:
        it = self.get(id_)
        if not it:
            return None
        for i in self.d["items"]:
            if i["kind"] == it["kind"]:
                i["active"] = (i["id"] == id_)
        self.save()
        LOG.info("promoted %s (kind=%s) to active", id_, it["kind"])
        return it

    def resolve(self, ref: str) -> Optional[dict]:
        """Find an item by id or unique substring."""
        if not ref:
            return None
        it = self.get(ref)
        if it:
            return it
        hits = [i for i in self.d["items"] if ref in i["id"]]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            LOG.error("ambiguous ref %r -> %s", ref, [h["id"] for h in hits])
        return None

    def prune_missing(self) -> None:
        keep = [i for i in self.d["items"] if Path(i["path"]).exists()]
        if len(keep) != len(self.d["items"]):
            self.d["items"] = keep
            self.save()
