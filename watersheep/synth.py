"""Synthetic example generation and verification."""
from __future__ import annotations
import random
import re
from typing import Dict, List, Optional, Tuple

from .schema import LETTERS, YESNO, answers, make, score_options, teacher_view, validate
from .util import clean_text, norm_text, stable_int

GEN_SYSTEM = (
    "You create realistic training examples for a decision model used inside software. "
    "Each example has a CONTEXT (exactly what the application would pass in), a QUESTION "
    "about it, and answer options. Exactly one option must be clearly best using only the "
    "context. Use concrete, specific details (names, numbers, dates, product names) and "
    "never placeholders like [Name] or {company}. Never state or hint the answer in the "
    "question. Output only the JSON object.")

JUDGE_SYSTEM = (
    "You are a careful, literal decision maker. Read the context and the question, weigh "
    "every option, and reply with only the label of the single best option.")

JUDGE_MULTI_SYSTEM = (
    "You are a careful, literal decision maker. Several options may apply at once, or "
    "none. Judge only the one option you are asked about, using only the context, and "
    "reply with only yes or no.")

QUALITY_SYSTEM = (
    "You audit training data for a decision model. Grade the example from 1 to 5:\n"
    "5 = realistic, self-contained, exactly one clearly best answer, marked answer is "
    "correct, question does not give the answer away\n"
    "4 = good, minor issues\n3 = usable but somewhat ambiguous or unrealistic\n"
    "2 = ambiguous, or the marked answer is debatable\n1 = broken, wrong answer, or "
    "the answer is not determined by the context.\nReply with only the digit.")

_PLACEHOLDER = re.compile(r"\[(?:[A-Z][A-Za-z ]{0,20}|insert[^\]]*|placeholder[^\]]*)\]|"
                          r"\{[a-z_]{2,30}\}|lorem ipsum|<[A-Z_]{3,}>|\bXXX+\b", re.I)
_GIVEAWAY = re.compile(r"\b(the (correct|right) (answer|option) is|answer\s*:\s*\w)", re.I)
_COUNT = re.compile(r"\b(which|select|choose|pick|name|list)\s+(the\s+)?(one|two|three|four|five|six|both|\d+)\b|"
                    r"\b(two|three|four|five|\d+)\s+(of the |of these )?(options|tags|items|policies|"
                    r"requirements|topics|tools|teams|actions|fields|factors|categories)\b", re.I)


def gen_prompt(j: dict) -> Tuple[str, dict]:
    t = j["type"]
    lines = [
        "Write ONE example.",
        "Decision family: %s - %s." % (j["family"].replace("_", " "), j["desc"]),
        "Domain: %s." % j["domain"],
        "Context style: %s." % j["style"],
        "Difficulty: %s (%s)." % (j["difficulty"], j["difficulty_desc"]),
        "",
        'Fields:',
        '- "state": the full context, about %d words. It must contain everything needed to '
        'decide, and must not state the answer outright.' % j["words"],
    ]
    props: Dict[str, dict] = {"state": {"type": "string"}, "question": {"type": "string"}}
    if t == "binary":
        lines += ['- "question": one yes/no question about the context.',
                  '- "answer": the correct answer to your question, which MUST be "%s" - '
                  'write the context so that "%s" is clearly correct.' % (j["target"], j["target"])]
        props["answer"] = {"type": "string", "enum": ["yes", "no"]}
    elif t == "score":
        lines += ['- "question": one question asking for a rating on a %d-%d scale; state in '
                  'the question what %d and %d mean.' % (j["lo"], j["hi"], j["lo"], j["hi"]),
                  '- "answer": the correct rating, which MUST be %d - write the context so that '
                  '%d is clearly the right rating.' % (j["target"], j["target"])]
        props["answer"] = {"type": "integer", "minimum": j["lo"], "maximum": j["hi"]}
    elif t == "multi":
        n, k = j["n_options"], j["n_true"]
        lines += ['- "question": one question asking which of the options apply (any number '
                  'may apply, including none). Do not say how many apply.',
                  '- "options": exactly %d distinct, plausible, concise options (a few words each).' % n,
                  ('- "answer": the 0-based indices of ALL options that apply. Exactly %d of them '
                   'must apply - write the context so that exactly those %d clearly apply and the '
                   'others clearly do not.' % (k, k)) if k else
                  '- "answer": an empty list - write the context so that NONE of the options '
                  'applies, although they are plausible for this kind of case.']
        props["options"] = {"type": "array", "items": {"type": "string"},
                            "minItems": n, "maxItems": n}
        props["answer"] = {"type": "array", "items": {"type": "integer", "minimum": 0,
                                                      "maximum": n - 1},
                           "minItems": k, "maxItems": k}
    else:
        n = j["n_options"]
        lines += ['- "question": one question asking which option is best.',
                  '- "options": exactly %d distinct, plausible, concise options (a few words '
                  'each). Wrong options must be tempting but clearly wrong.' % n,
                  '- "answer": the 0-based index of the single best option.']
        props["options"] = {"type": "array", "items": {"type": "string"},
                            "minItems": n, "maxItems": n}
        props["answer"] = {"type": "integer", "minimum": 0, "maximum": n - 1}
    lines.append('- "why": one sentence explaining why the answer is correct.')
    props["why"] = {"type": "string"}
    schema = {"type": "object", "properties": props,
              "required": list(props.keys()), "additionalProperties": False}
    return "\n".join(lines), schema


def to_record(j: dict, obj, cfg) -> Tuple[Optional[dict], str]:
    """Teacher JSON to a record, or (None, reason)."""
    if not isinstance(obj, dict):
        return None, "bad_json"
    state, question = obj.get("state"), obj.get("question")
    if not isinstance(state, str) or not isinstance(question, str):
        return None, "bad_json"
    t = j["type"]
    if t == "binary":
        a = str(obj.get("answer", "")).strip().lower()
        if a not in YESNO:
            return None, "bad_answer"
        if a != j["target"]:
            return None, "wrong_target"
        options, answer = YESNO, YESNO.index(a)
    elif t == "score":
        try:
            v = int(obj.get("answer"))
        except (TypeError, ValueError):
            return None, "bad_answer"
        if v != j["target"]:
            return None, "wrong_target"
        options, answer = score_options(j["lo"], j["hi"]), v - j["lo"]
    elif t == "multi":
        options, a = obj.get("options"), obj.get("answer")
        if not isinstance(options, list) or not isinstance(a, list) or \
                not all(isinstance(x, int) and not isinstance(x, bool) for x in a):
            return None, "bad_json"
        if len(set(a)) != j["n_true"]:
            return None, "wrong_target"
        options, answer = [str(o) for o in options], sorted(set(a))
    else:
        options = obj.get("options")
        a = obj.get("answer")
        if not isinstance(options, list) or not isinstance(a, int) or isinstance(a, bool):
            return None, "bad_json"
        options, answer = [str(o) for o in options], a
    rec = make("syn:" + j["jid"], "synth:" + j["family"], "synth", t, state, question,
               options, answer, cfg, meta={"family": j["family"], "domain": j["domain"],
                                           "difficulty": j["difficulty"],
                                           "why": clean_text(obj.get("why", ""), 400)})
    ok, why = validate(rec, cfg.max_options)
    if not ok:
        return None, why
    return rec, "ok"


def structure_checks(rec: dict) -> Optional[str]:
    s, q = rec["state"], rec["question"]
    if len(s) < 40:
        return "short_state"
    if _PLACEHOLDER.search(s) or _PLACEHOLDER.search(q) or \
            any(_PLACEHOLDER.search(o) for o in rec["options"]):
        return "placeholder"
    if _GIVEAWAY.search(s) or _GIVEAWAY.search(q):
        return "giveaway"
    if rec["type"] == "multi" and max(len(o) for o in rec["options"]) > 160:
        return "long_option"
    if rec["type"] == "multi" and _COUNT.search(q):
        return "count_in_question"
    if rec["type"] == "choice":
        right = norm_text(rec["options"][rec["answer"]])
        nq = norm_text(q)
        if len(right) >= 12 and right in nq and not any(
                norm_text(o) in nq for i, o in enumerate(rec["options"]) if i != rec["answer"]):
            return "answer_in_question"
        if max(len(o) for o in rec["options"]) > 160:
            return "long_option"
    return None


def blind_probs(teacher, rec: dict, orders: int, rng: random.Random) -> Optional[List[float]]:
    """Teacher probabilities per option, averaged over shuffled orders."""
    n = len(rec["options"])
    if rec["type"] == "score":
        perms = [list(range(n))]
    else:
        perms = []
        for _ in range(max(1, orders)):
            p = list(range(n))
            rng.shuffle(p)
            if perms and p == perms[-1]:
                p = p[::-1]
            perms.append(p)
    acc = [0.0] * n
    used = 0
    for order in perms:
        text, labels, order = teacher_view(rec, order)
        tail = ("\n\nReply with only the number." if rec["type"] == "score"
                else "\n\nReply with only the letter of the best option.")
        probs = teacher.option_probs(JUDGE_SYSTEM, text + tail, labels)
        if probs is None:
            continue
        for k, i in enumerate(order):
            acc[i] += probs[k]
        used += 1
    if not used:
        return None
    return [a / used for a in acc]


def multi_probs(teacher, rec: dict) -> Optional[List[float]]:
    """Teacher probability that each option applies."""
    text, _, _ = teacher_view(rec, list(range(len(rec["options"]))))
    out = []
    for k, opt in enumerate(rec["options"]):
        probs = teacher.option_probs(
            JUDGE_MULTI_SYSTEM, text + '\n\nDoes option %s) "%s" apply? Reply yes or no.'
            % (LETTERS[k], opt), YESNO)
        if probs is None:
            return None
        out.append(probs[0])
    return out


def quality_score(teacher, rec: dict) -> Optional[float]:
    text, _, _ = teacher_view(rec)
    if rec["type"] == "multi":
        marked = "; ".join("%s) %s" % (LETTERS[a], rec["options"][a]) for a in answers(rec)) or \
            "none of the options applies"
    elif rec["type"] == "score":
        marked = rec["options"][rec["answer"]]
    else:
        marked = "%s) %s" % (LETTERS[rec["answer"]], rec["options"][rec["answer"]])
    user = text + "\n\nMARKED ANSWER: " + marked + "\n\nGrade this example (1-5)."
    probs = teacher.option_probs(QUALITY_SYSTEM, user, ["1", "2", "3", "4", "5"])
    if probs is None:
        return None
    return sum((i + 1) * p for i, p in enumerate(probs))


def accept_blind(rec: dict, probs: List[float], min_p: float) -> Tuple[bool, float]:
    a = rec["answer"]
    if rec["type"] == "multi":
        on = set(answers(rec))
        agree = [p if i in on else 1 - p for i, p in enumerate(probs)]
        return min(agree) >= min_p, sum(agree) / len(agree)
    if rec["type"] == "score":
        ev = sum(i * p for i, p in enumerate(probs))
        return abs(ev - a) <= 1.0, probs[a]
    best = max(range(len(probs)), key=lambda i: probs[i])
    return best == a and probs[a] >= min_p, probs[a]


def relabel_multi(rec: dict, probs: List[float], conf: float) -> Optional[dict]:
    """Record with the teacher's labels when it is confident, else None."""
    if not probs or not all(p >= conf or p <= 1 - conf for p in probs):
        return None
    meta = dict(rec.get("meta") or {}, relabeled=True,
                writer_answer=list(rec["answer"]))
    return dict(rec, answer=[i for i, p in enumerate(probs) if p >= 0.5], meta=meta)


def soft_target(rec: dict, probs: List[float], mix: float) -> List[float]:
    n = len(rec["options"])
    if rec["type"] == "multi":
        on = set(answers(rec))
        return [mix * (1.0 if i in on else 0.0) + (1 - mix) * p for i, p in enumerate(probs)]
    t = [(1 - mix) * p for p in probs]
    t[rec["answer"]] += mix
    s = sum(t)
    return [x / s for x in t] if s > 0 else [1.0 / n] * n


def make_example(teacher, j: dict, cfg) -> dict:
    """Generate and verify one example."""
    out = {"jid": j["jid"], "family": j["family"], "type": j["type"], "status": "rejected"}
    user, schema = gen_prompt(j)
    obj, raw = teacher.generate_json(GEN_SYSTEM, user, schema, max_tokens=cfg.synth_max_tokens,
                                     temperature=cfg.synth_temperature, top_p=cfg.synth_top_p,
                                     seed=j["seed"])
    rec, why = to_record(j, obj, cfg)
    if rec is None:
        out.update(reason=why, raw=raw[:400])
        return out
    bad = structure_checks(rec)
    if bad:
        out.update(reason=bad, record=rec)
        return out
    rng = random.Random(stable_int("verify:" + j["jid"]))
    probs = multi_probs(teacher, rec) if rec["type"] == "multi" else \
        blind_probs(teacher, rec, cfg.verify_orders, rng)
    if probs is None:
        out.update(reason="judge_failed", record=rec)
        return out
    ok, p = accept_blind(rec, probs, cfg.verify_min_p)
    if not ok and rec["type"] == "multi":
        fixed = relabel_multi(rec, probs, cfg.verify_relabel_conf)
        if fixed is not None:
            rec = fixed
            ok, p = accept_blind(rec, probs, cfg.verify_min_p)
    out.update(p_answer=round(p, 4), teacher=[round(x, 4) for x in probs])
    if not ok:
        out.update(reason="teacher_disagrees", record=rec)
        return out
    q = quality_score(teacher, rec)
    out["quality"] = None if q is None else round(q, 3)
    if q is None or q < cfg.verify_min_quality:
        out.update(reason="low_quality", record=rec)
        return out
    rec["target"] = soft_target(rec, probs, cfg.soft_label_mix)
    rec["meta"].update(p_answer=round(p, 4), quality=round(q, 3))
    out.update(status="accepted", reason="ok", record=rec)
    return out


def simhash(text: str) -> int:
    toks = norm_text(text).split()
    grams = [" ".join(toks[i:i + 3]) for i in range(max(1, len(toks) - 2))] or [""]
    v = [0] * 64
    for g in grams:
        h = stable_int(g)
        for b in range(64):
            v[b] += 1 if (h >> b) & 1 else -1
    return sum(1 << b for b in range(64) if v[b] > 0)


class Deduper:
    """Near-duplicate filter."""

    def __init__(self, max_bits: int = 3):
        self.max_bits = max_bits
        self.bands: List[Dict[int, List[int]]] = [{} for _ in range(4)]
        self.exact: set = set()

    @staticmethod
    def key(rec: dict) -> str:
        return rec.get("state", "") + " || " + rec.get("question", "")

    def seen(self, rec: dict) -> bool:
        k = self.key(rec)
        ex = stable_int(norm_text(k))
        if ex in self.exact:
            return True
        h = simhash(k)
        for b in range(4):
            for other in self.bands[b].get((h >> (16 * b)) & 0xFFFF, ()):
                if bin(h ^ other).count("1") <= self.max_bits:
                    return True
        return False

    def add(self, rec: dict) -> None:
        k = self.key(rec)
        self.exact.add(stable_int(norm_text(k)))
        h = simhash(k)
        for b in range(4):
            self.bands[b].setdefault((h >> (16 * b)) & 0xFFFF, []).append(h)


def audit_one(teacher, rec: dict, seed: int) -> Optional[dict]:
    """Blind teacher answer for one public record."""
    if rec["type"] == "multi":
        probs = multi_probs(teacher, rec)
        if probs is None:
            return None
        on = set(answers(rec))
        agree = [p if i in on else 1 - p for i, p in enumerate(probs)]
        return {"id": rec["id"], "source": rec["source"], "chance": 0.5,
                "agree": sum(a >= 0.5 for a in agree) / len(agree), "p": sum(agree) / len(agree)}
    rng = random.Random(stable_int("audit:%s:%d" % (rec["id"], seed)))
    probs = blind_probs(teacher, rec, 1, rng)
    if probs is None:
        return None
    best = max(range(len(probs)), key=lambda i: probs[i])
    return {"id": rec["id"], "source": rec["source"], "agree": best == rec["answer"],
            "p": probs[rec["answer"]], "chance": 1.0 / len(rec["options"])}
