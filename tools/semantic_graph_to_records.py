#!/usr/bin/env python
"""Convert a typed semantic graph into WaterSheep NDJSON decision records."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from watersheep.data.semantic import graph_records
from watersheep.data.schema import validate


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True, type=Path)
    ap.add_argument("--edges", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--negatives", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--source-name", default="semantic-graph")
    ns = ap.parse_args(argv)

    records = graph_records(_load(ns.nodes), _load(ns.edges), negatives=ns.negatives,
                            seed=ns.seed, source_name=ns.source_name)
    bad = [(r.get("id"), validate(r)[1]) for r in records if not validate(r)[0]]
    if bad:
        raise SystemExit("generated invalid records: %r" % (bad[:5],))
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    with ns.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"records": len(records), "invalid": 0, "out": str(ns.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
