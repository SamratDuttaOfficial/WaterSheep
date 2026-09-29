"""Label descriptions, scale levels and yes/no criteria."""
from __future__ import annotations
import collections
import random
from typing import Dict, List, Optional

from . import interrupt
from .paths import P
from .schema import is_digit_scale, noul_question
from .util import LOG, append_jsonl, clean_text, iter_jsonl, read_json, stable_int, write_json

MAX_LABELS = 150
CHUNK = 25

SYSTEM = ("You write short, precise definitions for the labels and scales of a decision model. "
          "Be concrete, never repeat the label word for word, output only the JSON object.")


def _file():
    return P.state / "describe.json"


def _rows():
    return P.state / "describe_rows.jsonl"


def load() -> dict:
    d = read_json(_file(), {}) or {}
    return {"labels": d.get("labels", {}), "levels": d.get("levels", {}), "criteria": d.get("criteria", {})}


def _plan(cfg, names: List[str], synth) -> List[dict]:
    from .data import raw_file
    jobs = []
    for name in names:
        recs = list(iter_jsonl(raw_file(name)))
        if not recs:
            continue
        t = recs[0]["type"]
        if t in ("choice", "multi"):
            cnt = collections.Counter(o for r in recs for o in r["options"])
            if 3 <= len(cnt) <= MAX_LABELS and len(recs) >= 50:
                labels = sorted(cnt)
                for k in range(0, len(labels), CHUNK):
                    jobs.append({"kind": "labels", "key": name, "part": k // CHUNK,
                                 "labels": labels[k:k + CHUNK], "question": recs[0]["question"]})
        elif t == "score":
            seen = set()
            for r in recs:
                key = _scale_key(r)
                if key not in seen:
                    seen.add(key)
                    jobs.append({"kind": "levels", "key": key, "question": r["question"],
                                 "options": r["options"]})
        elif t == "binary":
            qs = collections.Counter(r["question"] for r in recs)
            for q, n in qs.items():
                if n >= 20:
                    jobs.append({"kind": "criteria", "key": q, "question": q})
    for r in synth:
        if r["type"] == "score" and _draw(r, "levels") < cfg.aug_levels:
            jobs.append({"kind": "levels", "key": r["id"], "question": r["question"],
                         "options": r["options"]})
    return jobs


def _scale_key(r: dict) -> str:
    return "%s|%s|%s" % (r["source"], r["question"], ",".join(r["options"]))


def _prompt(j: dict):
    if j["kind"] == "labels":
        user = ("A classifier is asked: \"%s\"\nFor each label below, write in at most 12 words what "
                "the label means as a category, i.e. when it applies. Define each label on its own: "
                "do not rank or compare labels and do not refer to any particular example. If a "
                "label is a name, just say what it names.\nLabels:\n%s"
                % (j["question"], "\n".join("- " + l for l in j["labels"])))
        props = {l: {"type": "string"} for l in j["labels"]}
        return user, {"type": "object", "properties": props, "required": list(props),
                      "additionalProperties": False}, 60 * len(j["labels"]) + 50
    if j["kind"] == "levels":
        n = len(j["options"])
        scale = ("%s (lowest) to %s (highest)" % (j["options"][0], j["options"][-1])
                 if is_digit_scale(j["options"]) else "%d levels" % n)
        user = ("A rating question: \"%s\"\nThe scale goes from %s. Write a short name (2-6 words) "
                "for each of its %d levels, lowest first, saying what that level means for this "
                "question." % (j["question"], scale, n))
        return user, {"type": "object", "properties": {"levels": {
            "type": "array", "items": {"type": "string"}, "minItems": n, "maxItems": n}},
            "required": ["levels"], "additionalProperties": False}, 25 * n + 40
    user = ("A yes/no question asked about many different texts: \"%s\"\nWrite one short sentence "
            "(at most 15 words) saying when the answer is yes, and one saying when it is no."
            % j["question"])
    return user, {"type": "object", "properties": {"yes": {"type": "string"}, "no": {"type": "string"}},
                  "required": ["yes", "no"], "additionalProperties": False}, 90


def _job_id(j: dict) -> str:
    return "%s|%s|%s" % (j["kind"], j["key"], j.get("part", 0))


def _ask(teacher, j: dict) -> Optional[dict]:
    user, schema, max_tokens = _prompt(j)
    try:
        obj, _ = teacher.generate_json(SYSTEM, user, schema, max_tokens=max_tokens, temperature=0.3,
                                       top_p=0.9, seed=stable_int(_job_id(j)) % (2 ** 31))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if j["kind"] == "labels":
        val = {l: clean_text(str(obj.get(l) or ""), 120) for l in j["labels"]}
        val = {l: d for l, d in val.items() if d}
    elif j["kind"] == "levels":
        lv = [clean_text(str(x), 60) for x in (obj.get("levels") or [])]
        ok = len(lv) == len(j["options"]) and all(lv) and len({x.lower() for x in lv}) == len(lv)
        val = lv if ok else None
    else:
        y, n = clean_text(str(obj.get("yes") or ""), 160), clean_text(str(obj.get("no") or ""), 160)
        val = {"yes": y, "no": n} if y and n else None
    return {"id": _job_id(j), "kind": j["kind"], "key": j["key"], "value": val} if val else None


def run(cfg, teacher, names: List[str], synth) -> dict:
    from .pool import pool_map
    jobs = _plan(cfg, names, list(synth))
    done = {r["id"] for r in iter_jsonl(_rows())}
    todo = [j for j in jobs if _job_id(j) not in done]
    LOG.info("describe: %d items (%d labels groups, %d scales, %d yes/no questions), %d already done",
             len(jobs), sum(j["kind"] == "labels" for j in jobs), sum(j["kind"] == "levels" for j in jobs),
             sum(j["kind"] == "criteria" for j in jobs), len(jobs) - len(todo))
    for k, res in enumerate(pool_map(lambda j: _ask(teacher, j), todo, teacher.workers), 1):
        if res:
            append_jsonl(_rows(), res)
        if k % 200 == 0:
            LOG.info("  describe %d/%d", k, len(todo))
    interrupt.check("describe stopped - finished items are kept")
    out: Dict[str, dict] = {"labels": {}, "levels": {}, "criteria": {}}
    for r in iter_jsonl(_rows()):
        if r["kind"] == "labels":
            out["labels"].setdefault(r["key"], {}).update(r["value"])
        else:
            out[r["kind"]][r["key"]] = r["value"]
    write_json(_file(), out)
    return {"label_sets": len(out["labels"]), "scales": len(out["levels"]),
            "criteria": len(out["criteria"])}


def _draw(r: dict, what: str) -> float:
    return (stable_int("aug:%s:%s" % (what, r["id"])) % 100000) / 100000.0


def augment(r: dict, D: dict, cfg) -> dict:
    """The record as shown to the model."""
    t = r["type"]
    if t in ("choice", "multi"):
        labs = D["labels"].get(r["source"])
        if not labs or not all(o in labs for o in r["options"]):
            return r
        r = dict(r)
        if t == "choice" and r["kind"] == "real" and len(r["options"]) >= 3 and \
                _draw(r, "multi") < cfg.aug_to_multi:
            r = _to_multi(r, labs)
        if _draw(r, "desc") < cfg.aug_descriptions:
            r["options"] = [clean_text("%s: %s" % (o, labs[o]), cfg.max_option_chars)
                            for o in r["options"]]
        return r
    if t == "score":
        lv = D["levels"].get(r["id"]) or D["levels"].get(_scale_key(r))
        if lv and len(lv) == len(r["options"]) and _draw(r, "levels") < cfg.aug_levels:
            return dict(r, options=list(lv))
        return r
    if t == "binary":
        c = D["criteria"].get(r["question"])
        if c and _draw(r, "criteria") < cfg.aug_criteria:
            return dict(r, question=noul_question(r["question"], c["yes"], c["no"]))
    return r


def _to_multi(r: dict, labs: dict) -> dict:
    """Turn a single-answer record into a select-all question."""
    rng = random.Random(stable_int("tomulti:" + r["id"]))
    opts, a = list(r["options"]), r["answer"]
    ans = [a]
    if rng.random() < 0.3:
        pool = [l for l in labs if l not in opts]
        if pool:
            opts[a] = rng.choice(pool)
        else:
            opts.pop(a)
        ans = []
    q = r["question"].rstrip() + " Select every option that applies (possibly none)."
    return dict(r, type="multi", options=opts, answer=ans, question=q, target=None,
                source=r["source"])
