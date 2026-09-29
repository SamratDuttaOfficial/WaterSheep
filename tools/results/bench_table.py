"""Benchmark and speed tables from saved benchmark results."""
from __future__ import annotations
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, bench_dir, model_arg                          # noqa: E402

from watersheep import benchmark as B                                  # noqa: E402

OUT = ROOT / "results" / "tables"
DATA = ROOT / "results" / "data"
ROWS = [
    ("xstest", "XSTest", "no"),
    ("prompt_injection", "Prompt injection (deepset)", "train split"),
    ("legal_cuad_audit_rights", "CUAD audit rights", "no"),
    ("legal_contract_nli_confidentiality_of_agreement", "ContractNLI confidentiality", "no"),
]
SUITE = {b.name: b.suite for b in B.BENCHMARKS}
TYPE = {"binary": "B", "choice": "C", "multi": "M"}


def pct(x):
    return "%.1f" % (100 * x)


def main():
    args = model_arg("Benchmark tables (newest export unless --model).")
    sd = bench_dir(args.model)
    teacher = B.teacher_results()
    lines, summ, group = [], {}, None
    for name, label, note in ROWS:
        s = json.load(open(sd / (name + ".json")))
        t = teacher.get(name)
        summ[name] = {"n": s["n"], "type": s["type"], "majority": s["majority"], "acc": s["acc"],
                      "macro_f1": s["macro_f1"], "ece": s["ece"], "ms": s["ms_per_question"],
                      "teacher": None if t is None else {"acc": t["acc"], "macro_f1": t["macro_f1"],
                                                         "ece": t["ece"], "ms": t["ms_per_question"],
                                                         "n": t["n"]}}
        if SUITE[name] != group:
            if group is not None:
                lines.append(r"\midrule")
            group = SUITE[name]
        tc = "%s & %.3f" % (pct(t["acc"]), t["macro_f1"]) if t else " & "
        lines.append("%s & %s & %s & %s & %s & %s & %s & %.3f & %.3f \\\\" % (
            label, TYPE[s["type"]], format(s["n"], ","), note, pct(s["majority"]), tc,
            pct(s["acc"]), s["macro_f1"], s["ece"]))
    mean = lambda k: statistics.mean(v[k] for v in summ.values())
    lines.append(r"\midrule")
    lines.append("Mean & & & & %s & & & %s & %.3f & %.3f \\\\" % (
        pct(mean("majority")), pct(mean("acc")), mean("macro_f1"), mean("ece")))
    OUT.mkdir(parents=True, exist_ok=True)
    lines.append(r"\bottomrule")  # the rule must be inside the input file
    (OUT / "bench.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    both = {k: v for k, v in summ.items() if v["teacher"]}
    sp = ["%s & %.0f & %.1f & %.0f \\\\" % (label, summ[name]["teacher"]["ms"], summ[name]["ms"],
                                             summ[name]["teacher"]["ms"] / summ[name]["ms"])
          for name, label, _ in ROWS if name in both]
    sp.append(r"\bottomrule")
    (OUT / "speed.tex").write_text("\n".join(sp) + "\n", encoding="utf-8")

    agg = {"model": sd.name, "benchmarks": summ, "mean": {k: mean(k) for k in ("majority", "acc", "macro_f1", "ece")},
           "with_teacher": sorted(both),
           "speedup": {k: v["teacher"]["ms"] / v["ms"] for k, v in both.items()}}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "bench_summary.json").write_text(json.dumps(agg, indent=1), encoding="utf-8")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    sys.exit(main())
