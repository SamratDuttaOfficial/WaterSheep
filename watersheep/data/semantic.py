"""Domain-neutral semantic graph -> WaterSheep decision records."""
from __future__ import annotations
import random
from typing import Dict, List

from ..core.util import stable_int
from .schema import make, validate


def _text(obj: dict, *keys: str) -> str:
    for k in keys:
        v = obj.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def node_label(node: dict) -> str:
    """Compact human-readable option text for a semantic node."""
    title = _text(node, "title", "label", "name", "id")
    formula = _text(node, "formula")
    return title if not formula else "%s — %s" % (title, formula)


def _state(node: dict, edge: dict) -> str:
    parts = ["Source concept: " + _text(node, "title", "label", "name", "id")]
    if _text(node, "summary"):
        parts.append("Summary: " + _text(node, "summary"))
    if _text(node, "formula"):
        parts.append("Formula: " + _text(node, "formula"))
    if _text(node, "cluster", "type"):
        parts.append("Class: " + _text(node, "cluster", "type"))
    parts.append("Relation: " + (_text(edge, "relation") or "related_to"))
    if _text(edge, "kind"):
        parts.append("Relation kind: " + _text(edge, "kind"))
    return "\n".join(parts)


def graph_records(nodes: List[dict], edges: List[dict], cfg=None, *, negatives: int = 3,
                  seed: int = 1234, source_name: str = "semantic-graph") -> List[dict]:
    """Generate deterministic relation-completion records from a typed graph.

    The supplied graph remains authoritative. This function only projects known edges
    into decision examples; it does not infer or promote new semantic claims.
    """
    by_id: Dict[str, dict] = {str(n.get("id")): n for n in nodes if n.get("id") is not None}
    ids = sorted(by_id)
    out: List[dict] = []
    seen_keys = set()
    for edge in edges:
        sid, tid = str(edge.get("source", "")), str(edge.get("target", ""))
        if sid not in by_id or tid not in by_id or sid == tid:
            continue
        src, tgt = by_id[sid], by_id[tid]
        tgt_class = _text(tgt, "cluster", "type")
        pool = [i for i in ids if i not in (sid, tid) and
                (not tgt_class or _text(by_id[i], "cluster", "type") == tgt_class)]
        if len(pool) < negatives:
            pool += [i for i in ids if i not in (sid, tid) and i not in pool]
        relation = _text(edge, "relation") or "related_to"
        edge_key = "%s|%s|%s" % (sid, relation, tid)
        if edge_key in seen_keys:
            continue
        seen_keys.add(edge_key)
        source_edge_id = str(edge.get("id") or "")
        edge_id = ("%s-" % source_edge_id if source_edge_id else "") + ("%016x" % stable_int(edge_key))[:10]
        rng = random.Random(stable_int("%s:%s" % (seed, edge_key)))
        chosen = rng.sample(pool, min(max(0, negatives), len(pool)))
        option_ids = [tid] + chosen
        rng.shuffle(option_ids)
        options = [node_label(by_id[i]) for i in option_ids]
        if len(options) < 2 or len({o.casefold() for o in options}) != len(options):
            continue
        rec = make(
            "sem:" + edge_id,
            "semantic:" + source_name,
            "real",
            "choice",
            _state(src, edge),
            "Which concept best completes the supplied typed relation from the source concept?",
            options,
            option_ids.index(tid),
            cfg,
            meta={
                "semantic_task": "relation_completion",
                "edge_id": source_edge_id,
                "edge_key": edge_key,
                "source_node_id": sid,
                "target_node_id": tid,
                "relation": relation,
                "relation_kind": _text(edge, "kind"),
                "directed": bool(edge.get("directed", True)),
            },
        )
        ok, _ = validate(rec, getattr(cfg, "max_options", 10) if cfg else 10)
        if ok:
            out.append(rec)
    return out
