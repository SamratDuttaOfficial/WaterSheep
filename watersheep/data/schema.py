"""Record format shared by sources, synthesis and training.

    {id, source, kind: real | synth, type: binary | choice | score | multi,
     state, question, options, answer, target, meta}
"""
from __future__ import annotations
from typing import List, Optional, Tuple

from ..core.util import clean_text, norm_text

TYPES = ("binary", "choice", "score", "multi")
YESNO = ["yes", "no"]
LETTERS = "ABCDEFGHIJ"


def score_options(lo: int, hi: int) -> List[str]:
    return [str(i) for i in range(int(lo), int(hi) + 1)]


def answers(r: dict) -> List[int]:
    """Indices of the correct options."""
    a = r["answer"]
    return sorted(set(int(x) for x in a)) if isinstance(a, (list, tuple)) else [int(a)]


def is_digit_scale(options: List[str]) -> bool:
    return all(len(o) == 1 and o.isdigit() for o in options)


def noul_question(question: str, yes: str = "", no: str = "") -> str:
    """Yes/no question with its criteria appended."""
    extra = "".join("\n%s = %s" % (k, v.strip()) for k, v in (("yes", yes), ("no", no)) if v and v.strip())
    return question.rstrip() + extra


def make(id_: str, source: str, kind: str, type_: str, state: str, question: str,
         options: List[str], answer, cfg=None, meta: Optional[dict] = None,
         target: Optional[List[float]] = None) -> dict:
    ms = getattr(cfg, "max_state_chars", 3000) if cfg else 3000
    mo = getattr(cfg, "max_option_chars", 160) if cfg else 160
    return {"id": id_, "source": source, "kind": kind, "type": type_,
            "state": clean_text(state, ms),
            "question": clean_text(question, 600),
            "options": [clean_text(o, mo) for o in options],
            "answer": sorted({int(a) for a in answer}) if type_ == "multi" else int(answer),
            "target": target, "meta": meta or {}}


def validate(r: dict, max_options: int = 10) -> Tuple[bool, str]:
    t = r.get("type")
    if t not in TYPES:
        return False, "bad_type"
    opts = r.get("options")
    if not isinstance(opts, list) or not all(isinstance(o, str) and o.strip() for o in opts):
        return False, "bad_options"
    if not 2 <= len(opts) <= max_options:
        return False, "option_count"
    if len({norm_text(o) for o in opts}) != len(opts):
        return False, "duplicate_options"
    a = r.get("answer")
    if t == "multi":
        if not isinstance(a, list) or len(set(a)) != len(a) or not all(
                isinstance(x, int) and not isinstance(x, bool) and 0 <= x < len(opts) for x in a):
            return False, "answer_range"
    elif not isinstance(a, int) or isinstance(a, bool) or not 0 <= a < len(opts):
        return False, "answer_range"
    q = r.get("question") or ""
    if not isinstance(q, str) or len(q.strip()) < 8:
        return False, "short_question"
    if not isinstance(r.get("state", ""), str):
        return False, "bad_state"
    if t == "binary" and [o.lower() for o in opts] != YESNO:
        return False, "binary_options"
    if t == "score":
        if len(opts) > 10:
            return False, "score_levels"
        if is_digit_scale(opts):
            vals = [int(o) for o in opts]
            if vals != list(range(vals[0], vals[0] + len(vals))):
                return False, "score_not_consecutive"
    tg = r.get("target")
    if tg is not None:
        if len(tg) != len(opts) or min(tg) < 0 or max(tg) > 1 + 1e-6:
            return False, "bad_target"
        if t != "multi" and abs(sum(tg) - 1.0) > 1e-3:
            return False, "bad_target"
    return True, "ok"


def labels_for(r: dict, order: List[int]) -> List[str]:
    """Reply labels the teacher uses for a record."""
    if r["type"] == "score":
        return [r["options"][i] for i in order] if is_digit_scale(r["options"]) else \
            [str(i) for i in order]
    return [LETTERS[k] for k in range(len(order))]


def teacher_view(r: dict, order: Optional[List[int]] = None) -> Tuple[str, List[str], List[int]]:
    """Render a record for a blind teacher answer."""
    n = len(r["options"])
    if r["type"] == "score" or order is None:
        order = list(range(n))
    labels = labels_for(r, order)
    state = r.get("state") or "(no additional context)"
    lines = ["CONTEXT:", state, "", "QUESTION: " + r["question"], "", "OPTIONS:"]
    if r["type"] == "score" and is_digit_scale(r["options"]):
        lines.append("Reply with one of: " + ", ".join(labels))
    elif r["type"] == "score":
        lines[-1] = "SCALE (lowest to highest):"
        for k, i in enumerate(order):
            lines.append("%s = %s" % (labels[k], r["options"][i]))
    else:
        if r["type"] == "multi":
            lines[-1] = "OPTIONS (any number of them may apply, or none):"
        for k, i in enumerate(order):
            lines.append("%s) %s" % (labels[k], r["options"][i]))
    return "\n".join(lines), labels, order


def hard_target(r: dict, smoothing: float, neighbor_mass: float) -> List[float]:
    """Smoothed target for a record without a soft label."""
    n, a = len(r["options"]), r["answer"]
    if r["type"] == "multi":
        on = set(answers(r))
        s = min(0.5, smoothing)
        return [1.0 - s if i in on else s for i in range(n)]
    t = [0.0] * n
    if r["type"] == "score" and neighbor_mass > 0 and n > 2:
        nb = [j for j in (a - 1, a + 1) if 0 <= j < n]
        t[a] = 1.0 - neighbor_mass
        for j in nb:
            t[j] += neighbor_mass / len(nb)
    else:
        t[a] = 1.0
    if smoothing > 0:
        t = [(1 - smoothing) * x + smoothing / n for x in t]
    s = sum(t)
    return [x / s for x in t]
