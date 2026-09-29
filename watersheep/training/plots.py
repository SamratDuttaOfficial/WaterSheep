"""Plots from saved reports."""
from __future__ import annotations
from pathlib import Path
from typing import List

from ..core.paths import P
from ..core.util import LOG, iter_jsonl, read_json


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _latest(pattern: str):
    files = sorted(P.reports.glob(pattern), key=lambda p: p.stat().st_mtime)
    return read_json(files[-1], {}) if files else {}


def training(ax, run_id: str) -> bool:
    rows = list(iter_jsonl(P.metrics / (run_id + ".jsonl")))
    tr = [r for r in rows if "loss" in r and not r.get("val")]
    va = [r for r in rows if r.get("val")]
    if not tr:
        return False
    ax.plot([r["step"] for r in tr], [r["loss"] for r in tr], lw=1, label="train loss")
    if va:
        ax.plot([r["step"] for r in va], [r["nll"] for r in va], "o-", lw=1.5, label="val NLL")
        ax2 = ax.twinx()
        ax2.plot([r["step"] for r in va], [r["acc"] for r in va], "s--", color="tab:green",
                 lw=1, label="val accuracy")
        ax2.set_ylabel("accuracy")
        ax2.legend(loc="center right", fontsize=8)
    ax.set_xlabel("optimizer step")
    ax.set_title("training")
    ax.legend(loc="upper right", fontsize=8)
    return True


def reliability(ax, ev: dict) -> bool:
    raw, cal = (ev.get("test_raw") or {}), (ev.get("test_calibrated") or {})
    if not raw.get("reliability"):
        return False
    ax.plot([0, 1], [0, 1], ":", color="gray")
    for name, d in (("raw", raw), ("calibrated", cal)):
        rel = d.get("reliability") or []
        if rel:
            ax.plot([r[0] for r in rel], [r[1] for r in rel], "o-",
                    label="%s (ECE %.3f)" % (name, d.get("ece", 0)))
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("calibration (test split)")
    ax.legend(fontsize=8)
    return True


def per_source(ax, ev: dict, key: str, title: str) -> bool:
    bs = (ev.get(key) or {}).get("by_source") or {}
    if not bs:
        return False
    names = sorted(bs, key=lambda s: bs[s]["acc"])
    y = range(len(names))
    ax.barh(list(y), [bs[s]["acc"] for s in names], color="tab:blue", label="accuracy")
    ax.scatter([bs[s]["chance"] for s in names], list(y), color="tab:red", marker="|", s=80,
               label="chance", zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([s.split(":", 1)[-1][:22] for s in names], fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    return True


def synth_funnel(ax) -> bool:
    s = read_json(P.reports / "synth.json", {}) or {}
    reasons = s.get("reasons") or {}
    if not reasons:
        return False
    items = sorted(reasons.items(), key=lambda kv: -kv[1])
    ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1],
            color=["tab:green" if k == "ok" else "tab:gray" for k, _ in items][::-1])
    ax.set_title("synthetic data: accepted (ok) vs rejected, by reason")
    ax.tick_params(axis="y", labelsize=7)
    return True


def synth_progress(ax) -> bool:
    rows = list(iter_jsonl(P.logs / "synth.jsonl"))
    if len(rows) < 2:
        return False
    t0 = rows[0]["t"]
    ax.plot([(r["t"] - t0) / 3600 for r in rows], [r["accepted"] for r in rows], label="accepted")
    ax.set_xlabel("hours of generation")
    ax.set_ylabel("accepted examples")
    ax2 = ax.twinx()
    ax2.plot([(r["t"] - t0) / 3600 for r in rows], [r.get("tok_s", 0) for r in rows],
             color="tab:orange", lw=0.8, label="teacher tok/s")
    ax2.set_ylabel("tok/s")
    ax.set_title("synthetic generation")
    return True


def audit(ax) -> bool:
    a = read_json(P.reports / "audit.json", {}) or {}
    ps = a.get("per_source") or {}
    if not ps:
        return False
    names = sorted(ps, key=lambda s: ps[s]["agree"])
    y = range(len(names))
    ax.barh(list(y), [ps[s]["agree"] for s in names],
            color=["tab:red" if s in a.get("dropped", []) else "tab:purple" for s in names])
    ax.scatter([ps[s]["chance"] for s in names], list(y), color="black", marker="|", s=80)
    ax.set_yticks(list(y))
    ax.set_yticklabels([s.split(":", 1)[-1][:22] for s in names], fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_title("teacher agreement with public labels (red = dropped)")
    return True


def make(run_id: str = "") -> List[Path]:
    plt = _plt()
    P.plots.mkdir(parents=True, exist_ok=True)
    ev = _latest("eval_*.json")
    run_id = run_id or ev.get("run_id", "")
    panels = [("training", lambda ax: training(ax, run_id)),
              ("calibration", lambda ax: reliability(ax, ev)),
              ("test_by_source", lambda ax: per_source(ax, ev, "test_calibrated", "test accuracy by source")),
              ("zeroshot_by_source", lambda ax: per_source(ax, ev, "zeroshot_calibrated",
                                                          "held-out sources (never trained on)")),
              ("synth_funnel", synth_funnel), ("synth_progress", synth_progress),
              ("audit", audit)]
    files, drawn = [], []
    for name, fn in panels:
        fig, ax = plt.subplots(figsize=(8, 5 if "source" not in name and name != "audit" else 9))
        try:
            ok = fn(ax)
        except Exception as e:
            LOG.debug("plot %s failed: %s", name, e)
            ok = False
        if ok:
            fig.tight_layout()
            f = P.plots / (name + ".png")
            fig.savefig(f, dpi=110)
            files.append(f)
            drawn.append((name, fn))
        plt.close(fig)
    if drawn:
        n = len(drawn)
        cols = 2
        rows = (n + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
        axes = axes.flatten() if n > 1 else [axes]
        for ax, (name, fn) in zip(axes, drawn):
            try:
                fn(ax)
            except Exception:
                pass
        for ax in axes[len(drawn):]:
            ax.axis("off")
        fig.tight_layout()
        f = P.plots / "dashboard.png"
        fig.savefig(f, dpi=90)
        plt.close(fig)
        files.append(f)
    LOG.info("wrote %d plots to %s", len(files), P.plots)
    return files
