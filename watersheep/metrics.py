"""Accuracy and calibration metrics, and temperature scaling."""
from __future__ import annotations
import math
from typing import Dict, List

import numpy as np

from .model import MULTI, TYPE_NAME


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, np.float64) / max(1e-6, T)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sigmoid(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.clip(np.asarray(z, np.float64) / max(1e-6, T), -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def probs_for(z: np.ndarray, type_id: int, T: float = 1.0) -> np.ndarray:
    return sigmoid(z, T) if type_id == MULTI else softmax(z, T)


def concentration(p) -> float:
    """1 minus normalised entropy."""
    p = np.clip(np.asarray(p, np.float64), 1e-12, 1.0)
    if len(p) < 2:
        return 1.0
    return float(max(0.0, 1.0 + (p * np.log(p)).sum() / math.log(len(p))))


def _reliability(conf: np.ndarray, correct: np.ndarray, bins: int):
    edges = np.linspace(0, 1, bins + 1)
    ece, rel = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
            rel.append([float(conf[m].mean()), float(correct[m].mean()), int(m.sum())])
    return float(ece), rel


def _core(ps: List[np.ndarray], ans: List[int], bins: int = 15) -> dict:
    n = len(ps)
    if n == 0:
        return {"n": 0}
    conf = np.array([p.max() for p in ps])
    pred = np.array([int(p.argmax()) for p in ps])
    correct = (pred == np.array(ans)).astype(np.float64)
    nll = float(np.mean([-math.log(max(1e-12, p[k])) for p, k in zip(ps, ans)]))
    brier = float(np.mean([((p - np.eye(len(p))[k]) ** 2).sum() for p, k in zip(ps, ans)]))
    ece, rel = _reliability(conf, correct, bins)
    return {"n": n, "acc": float(correct.mean()), "nll": nll, "brier": brier,
            "ece": ece, "mean_conf": float(conf.mean()), "reliability": rel}


def _multi_core(rows: List[dict], bins: int = 15) -> dict:
    """Per-option metrics for multi-label questions."""
    if not rows:
        return {"n": 0}
    p = np.concatenate([r["p"] for r in rows])
    y = np.concatenate([np.asarray(r["y"], bool) for r in rows])
    pred = p >= 0.5
    correct = (pred == y).astype(np.float64)
    conf = np.where(pred, p, 1 - p)
    ece, rel = _reliability(conf, correct, bins)
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    nll = float(np.mean([-(np.log(np.clip(r["p"], 1e-12, 1)) * r["y"] +
                           np.log(np.clip(1 - r["p"], 1e-12, 1)) * (1 - np.asarray(r["y"], float))).mean()
                         for r in rows]))
    tp, fp, fn = float((pred & y).sum()), float((pred & ~y).sum()), float((~pred & y).sum())
    exact = float(np.mean([bool(((r["p"] >= 0.5) == np.asarray(r["y"], bool)).all()) for r in rows]))
    return {"n": len(rows), "acc": exact, "option_acc": float(correct.mean()), "nll": nll,
            "brier": float(((pc - y) ** 2).mean()), "ece": ece, "mean_conf": float(conf.mean()),
            "f1": 2 * tp / max(1.0, 2 * tp + fp + fn), "reliability": rel}


def summarize(rows: List[dict]) -> dict:
    """Metrics overall, by type and by source."""
    single = [r for r in rows if r["t"] != MULTI]
    multi = [r for r in rows if r["t"] == MULTI]
    a, b = _core([r["p"] for r in single], [r["a"] for r in single]), _multi_core(multi)
    na, nb = a.get("n", 0), b.get("n", 0)
    n = max(1, na + nb)
    out = {"n": na + nb}
    for k in ("acc", "nll", "brier", "ece", "mean_conf"):
        out[k] = (a.get(k, 0.0) * na + b.get(k, 0.0) * nb) / n
    out["reliability"] = a.get("reliability") if na >= nb else b.get("reliability")
    out["by_type"] = {}
    for tid, name in TYPE_NAME.items():
        sub = [r for r in rows if r["t"] == tid]
        if not sub:
            continue
        d = _multi_core(sub) if tid == MULTI else _core([r["p"] for r in sub], [r["a"] for r in sub])
        d.pop("reliability", None)
        if name == "score":
            d["mae"] = float(np.mean([abs(float((np.arange(len(r["p"])) * r["p"]).sum()) - r["a"])
                                      for r in sub]))
        out["by_type"][name] = d
    by_src: Dict[str, list] = {}
    for r in rows:
        by_src.setdefault(r["s"], []).append(r)
    out["by_source"] = {}
    for s, sub in sorted(by_src.items()):
        if all(r["t"] == MULTI for r in sub):
            d = _multi_core(sub)
            d["chance"] = float(np.mean([max(np.mean(r["y"]), 1 - np.mean(r["y"])) for r in sub]))
        else:
            sub = [r for r in sub if r["t"] != MULTI]
            d = _core([r["p"] for r in sub], [r["a"] for r in sub])
            d["chance"] = float(np.mean([1.0 / len(r["p"]) for r in sub]))
        d.pop("reliability", None)
        out["by_source"][s] = d
    return out


def _golden(f) -> float:
    a, b = math.log(0.2), math.log(8.0)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return float(math.exp((a + b) / 2))


def fit_temperature(logits: List[np.ndarray], ans: List[int]) -> float:
    """Temperature that minimises NLL."""
    if not logits:
        return 1.0
    return _golden(lambda logT: float(np.mean([-math.log(max(1e-12, softmax(z, math.exp(logT))[k]))
                                               for z, k in zip(logits, ans)])))


def fit_temperature_multi(logits: List[np.ndarray], truth: List[np.ndarray]) -> float:
    """Temperature that minimises per-option BCE."""
    if not logits:
        return 1.0
    z = np.concatenate(logits)
    y = np.concatenate([np.asarray(t, float) for t in truth])

    def bce(logT: float) -> float:
        p = np.clip(sigmoid(z, math.exp(logT)), 1e-12, 1 - 1e-12)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    return _golden(bce)
