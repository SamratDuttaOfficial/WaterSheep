"""Helpers shared by the results scripts."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def model_arg(description: str):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model", help="export folder or name (default: the newest export)")
    ap.add_argument("what", nargs="*", help="parts to produce (default: all)")
    return ap.parse_args()


def export_dir(model=None) -> Path:
    from watersheep.infer import latest_export
    if model:
        p = Path(model)
        return p if p.is_absolute() or p.exists() else ROOT / "out" / "export" / p.name
    d = latest_export()
    if d is None:
        raise SystemExit("no exported model found in out/export")
    return d


def export_meta(model=None) -> dict:
    return json.loads((export_dir(model) / "watersheep.json").read_text(encoding="utf-8"))


def bench_dir(model=None) -> Path:
    return ROOT / "out" / "benchmarks" / export_dir(model).name
