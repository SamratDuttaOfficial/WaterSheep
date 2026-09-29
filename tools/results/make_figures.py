"""Figures from saved statistics and benchmark results."""
from __future__ import annotations
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402

from watersheep.core.util import iter_jsonl                            # noqa: E402

DATA = ROOT / "results" / "data"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    "svg.hashsalt": "watersheep",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
# Okabe-Ito palette
C = {"binary": "#0072B2", "choice": "#E69F00", "score": "#009E73", "multi": "#CC79A7",
     "grey": "#7F7F7F", "red": "#D55E00", "sky": "#56B4E9"}
TYPES = ["binary", "choice", "score", "multi"]
TNAME = {"binary": "Binary", "choice": "Choice", "score": "Score", "multi": "Multi"}


def save(fig, name):
    fig.savefig(FIG / (name + ".pdf"), metadata={"CreationDate": None})
    fig.savefig(FIG / (name + ".svg"), pad_inches=0.1, metadata={"Date": None})
    plt.close(fig)
    print("wrote", name)


def fig_synth():
    st = json.load(open(DATA / "data_stats.json"))
    fams = st["synth"]["families"]
    reasons = collections.defaultdict(collections.Counter)
    for f in sorted((ROOT / "data" / "synth").glob("shard_*.jsonl")):
        for row in iter_jsonl(f):
            r = row.get("reason", "?")
            group = ("accepted" if r == "ok" else "teacher_disagrees" if r == "teacher_disagrees" else
                     "low_quality" if r == "low_quality" else "structure")
            reasons[row["type"]][group] += 1
    json.dump({t: dict(v) for t, v in reasons.items()}, open(DATA / "synth_outcomes_by_type.json", "w"), indent=1)

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.35), gridspec_kw={"width_ratios": [1.0, 1.25]})
    groups = [("accepted", "Accepted", "#4C9A6A"), ("teacher_disagrees", "Teacher disagrees", "#E0A458"),
              ("low_quality", "Quality below 3", "#C8553D"), ("structure", "Structural check", "#8C8C8C")]
    x = np.arange(len(TYPES))
    bottom = np.zeros(len(TYPES))
    for key, lab, col in groups:
        tot = np.array([sum(reasons[t].values()) for t in TYPES], float)
        v = np.array([reasons[t][key] for t in TYPES], float) / tot
        a.bar(x, v, 0.62, bottom=bottom, color=col, label=lab, edgecolor="white", linewidth=0.4)
        for i, (vv, bb) in enumerate(zip(v, bottom)):
            if vv > 0.07:
                a.text(i, bb + vv / 2, "%.0f%%" % (100 * vv), ha="center", va="center", fontsize=6.5,
                       color="white" if key != "structure" else "black")
        bottom += v
    a.set_xticks(x)
    a.set_xticklabels([TNAME[t] + "\n(n=%s)" % format(sum(reasons[t].values()), ",") for t in TYPES])
    a.set_ylabel("Share of generated examples")
    a.set_ylim(0, 1)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, handlelength=1.0,
             columnspacing=0.8)
    a.set_title("(a) Verification outcome by type", loc="left")

    rng = np.random.default_rng(0)
    for i, t in enumerate(TYPES):
        vals = sorted(((v["accepted"] / v["tried"], k) for k, v in fams.items() if v["type"] == t))
        ys = np.array([v for v, _ in vals])
        xs = i + rng.uniform(-0.17, 0.17, len(ys))
        b.scatter(xs, ys, s=12, color=C[t], alpha=0.85, edgecolor="white", linewidth=0.3, zorder=3)
        b.plot([i - 0.25, i + 0.25], [np.median(ys)] * 2, color="black", lw=0.9, zorder=4)
        lo, hi = vals[0], vals[-1]
        b.annotate(lo[1].replace("_", " "), (xs[0], lo[0]), xytext=(0, -8), textcoords="offset points",
                   ha="center", fontsize=6, color="#444444")
        b.annotate(hi[1].replace("_", " "), (xs[-1], hi[0]), xytext=(0, 5), textcoords="offset points",
                   ha="center", fontsize=6, color="#444444")
    b.set_xticks(range(len(TYPES)))
    b.set_xticklabels([TNAME[t] + " (%d)" % sum(1 for v in fams.values() if v["type"] == t) for t in TYPES])
    b.set_ylabel("Acceptance rate per family")
    b.set_ylim(0.25, 0.97)
    b.set_xlim(-0.6, 3.6)
    b.set_title("(b) Acceptance rate of the 57 families", loc="left")
    b.grid(axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout(w_pad=2.0)
    save(fig, "synth")


def ema(v, a=0.97):
    out, m = [], None
    for x in v:
        m = x if m is None else a * m + (1 - a) * x
        out.append(m)
    return out


def fig_training():
    st = json.load(open(DATA / "data_stats.json"))["train"]
    curve, val = st["curve"], st["val"]
    best = st["summary"]["best_step"]
    stop = st["summary"]["steps"]
    total = st["summary"]["total_steps"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.2))
    s = [c["step"] for c in curve]
    a.plot(s, [c["loss"] for c in curve], color="#BBBBBB", lw=0.4, label="per 20 steps")
    a.plot(s, ema([c["loss"] for c in curve]), color="black", lw=1.0, label="moving average")
    a.set_xlabel("Optimizer step")
    a.set_ylabel("Training loss")
    a.set_ylim(0, 1.6)
    a.legend(frameon=False, loc="upper right")
    ax2 = a.twinx()
    ax2.plot(s, [c["lr"] for c in curve], color=C["sky"], lw=0.9, ls="--")
    ax2.set_ylabel("Encoder learning rate", color=C["sky"])
    ax2.tick_params(axis="y", colors=C["sky"])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C["sky"])
    ax2.set_ylim(0, 5.5e-5)
    a.set_title("(a) Loss and learning rate", loc="left")

    vs = [v["step"] for v in val]
    for t in TYPES:
        b.plot(vs, [v["by_type"][t] for v in val], color=C[t], label=TNAME[t], lw=1.1)
    b.plot(vs, [v["acc"] for v in val], color="black", lw=1.3, label="All")
    for ax in (a, b):
        ax.axvline(best, color=C["red"], lw=0.7, ls=":")
    b.text(best, 0.24, " selected\n step %s" % format(best, ","), color=C["red"], fontsize=6.5, ha="right")
    b.set_xlabel("Optimizer step")
    b.set_ylabel("Validation accuracy")
    b.set_ylim(0.2, 1.0)
    b.legend(frameon=False, ncol=5, loc="upper center", fontsize=6.8, handlelength=1.2, columnspacing=0.8)
    b.set_title("(b) Validation accuracy (3,000 examples)", loc="left")
    b.grid(axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout(w_pad=1.5)
    save(fig, "training")
    return {"best": best, "stop": stop, "total": total}


def fig_reliability():
    from common import bench_dir
    f = bench_dir(ARGS.model) / "analysis.json"
    if not f.exists():
        raise SystemExit("%s not found - run run-benchmarks.bat (or .sh) for this model first" % f)
    rel = json.load(open(f))["calibration"]
    fig, axs = plt.subplots(1, 4, figsize=(6.5, 1.85), sharey=True)
    for ax, t in zip(axs, TYPES):
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.6, ls="--")
        for tag, mk, fill in (("raw", "o", False), ("calibrated", "o", True)):
            bins = rel[t][tag]["bins"]
            conf = [b[0] for b in bins]
            acc = [b[1] for b in bins]
            n = np.array([b[2] for b in bins], float)
            size = 6 + 40 * n / n.max()
            if fill:
                ax.scatter(conf, acc, s=size, color=C[t], edgecolor="white", linewidth=0.3, zorder=3,
                           label="calibrated")
                ax.plot(conf, acc, color=C[t], lw=0.7, zorder=2)
            else:
                ax.scatter(conf, acc, s=size, facecolor="none", edgecolor="#888888", linewidth=0.5,
                           zorder=2, label="raw")
        ax.text(0.04, 0.93, "ECE %.3f $\\rightarrow$ %.3f" % (rel[t]["raw"]["ece"], rel[t]["calibrated"]["ece"]),
                transform=ax.transAxes, fontsize=6.5, va="top")
        ax.set_title(TNAME[t] + (" (per option)" if t == "multi" else ""), fontsize=8.5)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.5, 1])
        ax.set_aspect("equal")
        ax.set_xlabel("Confidence")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(frameon=False, loc="lower right", fontsize=6.5, handletextpad=0.2)
    fig.tight_layout(w_pad=0.6)
    save(fig, "reliability")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import model_arg
    ARGS = model_arg("Render the figures (newest export unless --model).")
    for w in ARGS.what or ["synth", "training", "reliability"]:
        globals()["fig_" + w]()
