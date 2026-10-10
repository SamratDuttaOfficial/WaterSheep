"""Error against coverage: answer only above a confidence threshold, on the test and held-out splits."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np                                                     # noqa: E402

from watersheep.core.util import iter_jsonl                            # noqa: E402

OUT = ROOT / "results" / "data"
BUILD = ROOT / "data" / "build"
SPLITS = (("test", "test"), ("heldout", "zeroshot"))
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97)


def confidence(ws, recs, logits):
    """Calibrated confidence and correctness per record. A multi-label answer is correct when every
    option is right, and is as confident as its least certain option."""
    from watersheep.metrics import sigmoid, softmax
    conf, corr = [], []
    for r, z in zip(recs, logits):
        T = ws.temps.get(r["type"], 1.0)
        if r["type"] == "multi":
            p = sigmoid(z, T)
            y = np.zeros(len(p), bool)
            y[list(r["answer"])] = True
            pred = p >= 0.5
            conf.append(float(np.where(pred, p, 1 - p).min()))
            corr.append(bool((pred == y).all()))
        else:
            p = softmax(z, T)
            conf.append(float(p.max()))
            corr.append(int(p.argmax()) == r["answer"])
    return np.array(conf), np.array(corr, bool)


def curve(conf, corr) -> dict:
    order = np.argsort(-conf, kind="stable")
    wrong = np.cumsum(~corr[order])
    n = len(conf)
    points = []
    for c in range(5, 101, 5):
        k = max(1, round(n * c / 100))
        points.append([c / 100, float(wrong[k - 1] / k)])
    at = []
    for th in THRESHOLDS:
        m = conf >= th
        at.append({"threshold": th, "coverage": float(m.mean()),
                   "error": float((~corr[m]).mean()) if m.any() else None})
    return {"n": n, "acc": float(corr.mean()), "curve": points, "thresholds": at}


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import export_dir, model_arg
    from watersheep.benchmark import logits_for
    from watersheep.infer import WaterSheep
    args = model_arg("Error against coverage on the test and held-out splits (newest export unless --model).")
    ws = WaterSheep.load(str(export_dir(args.model)))
    res = {"model": ws.name}
    for name, split in SPLITS:
        f = BUILD / (split + ".jsonl")
        if not f.exists():
            raise SystemExit("%s not found - run the build stage first" % f)
        recs = list(iter_jsonl(f))
        logits = logits_for(ws, recs)
        if logits is None:
            return 1
        res[name] = curve(*confidence(ws, recs, logits))
        print("%-8s n=%d accuracy %.4f" % (name, res[name]["n"], res[name]["acc"]))
        for a in res[name]["thresholds"]:
            print("   confidence >= %.2f: coverage %5.1f%%  error %s" % (
                a["threshold"], 100 * a["coverage"],
                "-" if a["error"] is None else "%.1f%%" % (100 * a["error"])))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coverage.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUT / "coverage.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
