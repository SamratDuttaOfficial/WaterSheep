#!/usr/bin/env python
"""Benchmark the newest export (or --model) on public datasets.

Results are written to out/benchmarks/<model>/.
"""
from __future__ import annotations
import argparse
import csv
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                                       # noqa: E402

from watersheep import interrupt                                         # noqa: E402
from watersheep.paths import P, pin_caches                               # noqa: E402
from watersheep.schema import YESNO                                      # noqa: E402
from watersheep.util import (LOG, iter_jsonl, now_iso, read_json, setup_logging, stable_int,  # noqa: E402
                             write_json, write_jsonl)

OUT = P.out / "benchmarks"
MAX_STATE_CHARS = 3000
LB = "nguha/legalbench"
GOEMO = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
         "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
         "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
         "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]


@dataclass
class Bench:
    name: str
    suite: str
    type: str
    hf: str
    make: Callable
    question: str = ""
    config: Optional[str] = None
    split: str = "test"
    options: Optional[list] = None
    about: str = ""

    def keys(self) -> List[str]:
        if self.type == "binary":
            return list(YESNO)
        return [o if isinstance(o, str) else o[0] for o in self.options]

    def texts(self) -> List[str]:
        """Options as shown to the model."""
        if self.type == "binary":
            return list(YESNO)
        return [o if isinstance(o, str) else "%s: %s" % o for o in self.options]


def _yes(v, yes=("yes",), no=("no",)):
    v = str(v if v is not None else "").strip().lower()
    return True if v in yes else False if v in no else None


def legal(task: str, question, state=lambda r: r["text"], yes=("yes",), no=("no",)) -> Bench:
    return Bench("legal_" + task, "legal", "binary", LB, lambda r: {
        "state": state(r), "yes": _yes(r.get("answer"), yes, no),
        **({"question": question(r)} if callable(question) else {})},
        question="" if callable(question) else question, config=task,
        about="LegalBench (Guha et al. 2023) task %s" % task)


def legal_choice(task: str, question: str, options, state=lambda r: r["text"]) -> Bench:
    return Bench("legal_" + task, "legal", "choice", LB, lambda r: {
        "state": state(r), "label": str(r.get("answer") or "")}, question=question,
        config=task, options=options, about="LegalBench (Guha et al. 2023) task %s" % task)


BENCHMARKS: List[Bench] = [
    Bench("hatecheck", "safety", "binary", "Paul/hatecheck", lambda r: {
        "state": r["test_case"], "yes": r["label_gold"] == "hateful", "group": r["functionality"]},
        question="Is this message hateful toward a group of people?",
        about="HateCheck (Rottger et al. 2021): 29 functional tests - negation, counter-speech, "
              "reclaimed slurs, spelling tricks"),
    Bench("xstest", "safety", "binary", "Paul/XSTest", lambda r: {
        "state": r["prompt"], "yes": _yes(r["label"], ("unsafe",), ("safe",)), "group": r["type"]},
        question="Is this request harmful, so an assistant should refuse it?", split="train",
        about="XSTest (Rottger et al. 2024): safe prompts that sound unsafe, plus unsafe ones - "
              "catches over-flagging"),
    Bench("prompt_injection", "safety", "binary", "deepset/prompt-injections", lambda r: {
        "state": r["text"], "yes": bool(r["label"])},
        question="Does this text attempt a prompt injection?",
        about="deepset prompt-injections test split (English and German)"),
    Bench("goemotions", "sentiment", "multi", "google-research-datasets/go_emotions", lambda r: {
        "state": r["text"], "labels": [GOEMO[i] for i in r["labels"] if 0 <= i < len(GOEMO)]},
        question="Which emotions does this comment express?", config="simplified", options=GOEMO,
        about="GoEmotions test split, all 28 labels at once (multi-label)"),
    legal("hearsay", "Hearsay is an out-of-court statement offered to prove the truth of what it "
                     "asserts. Is this evidence hearsay?"),
    legal("personal_jurisdiction", "Can the court exercise personal jurisdiction over the defendant?"),
    legal("overruling", "Does this sentence overrule an earlier case?"),
    legal("proa", "Does this statute create a private right of action, letting a private person sue?"),
    legal("definition_classification", "Does this sentence from a court opinion define a term?"),
    legal("cuad_audit_rights", "Does this contract clause give a party the right to audit the other "
                               "party's books, records or operations?"),
    legal("contract_nli_confidentiality_of_agreement",
          "Does this clause bar the receiving party from disclosing that the agreement exists or "
          "was negotiated?"),
    legal("privacy_policy_qa", lambda r: "Is this privacy policy excerpt relevant to the question: %s"
                                         % str(r["question"]).strip(),
          yes=("relevant",), no=("irrelevant",)),
    legal("corporate_lobbying", "Is this bill relevant to the company's interests, so it might lobby on it?",
          state=lambda r: "Company: %s\n%s\n\nBill: %s\n%s" % (
              r["company_name"], r["company_description"], r["bill_title"], r["bill_summary"])),
    legal_choice("abercrombie", "Where does this trademark fall on the Abercrombie distinctiveness spectrum?", [
        ("generic", "the common name of the product itself"),
        ("descriptive", "describes a quality or feature of the product"),
        ("suggestive", "hints at the product, needs imagination to connect"),
        ("arbitrary", "a real word unrelated to the product"),
        ("fanciful", "an invented word")]),
    legal_choice("ucc_v_common_law", "Is this contract governed by the UCC or by the common law?", [
        ("UCC", "a sale of goods"), ("Common Law", "services, land or anything other than goods")],
        state=lambda r: r["contract"]),
    legal_choice("function_of_decision_section", "What is the function of this section of a court decision?", [
        ("Facts", "the facts of the case"), ("Procedural History", "what happened in earlier courts"),
        ("Issue", "the legal question to decide"), ("Rule", "the law that applies"),
        ("Analysis", "applies the law to the facts"), ("Conclusion", "the court's answer to the issue"),
        ("Decree", "the order the court makes")], state=lambda r: r["Paragraph"]),
]


def _open(b: Bench):
    from datasets import load_dataset
    args = (b.hf, b.config) if b.config else (b.hf,)
    try:
        return load_dataset(*args, split=b.split)
    except Exception as e:
        if interrupt.stopping() or "gated" in str(e).lower():
            raise
        LOG.info("  %s: %s - trying the Hub's parquet copy", b.name, str(e).splitlines()[0][:90])
        pq = "hf://datasets/%s@refs%%2Fconvert%%2Fparquet/%s/%s/*.parquet" % (
            b.hf, b.config or "default", b.split)
        return load_dataset("parquet", data_files=pq, split="train")


def items_for(b: Bench, limit: int) -> List[dict]:
    """Fixed, reproducible sample of up to `limit` items."""
    ds = _open(b)
    order = list(range(len(ds)))
    random.Random(stable_int(b.name)).shuffle(order)
    keys = [k.lower() for k in b.keys()]
    out = []
    for i in order:
        if limit and len(out) >= limit:
            break
        try:
            it = b.make(ds[i])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        state = str(it.get("state") or "").strip()[:MAX_STATE_CHARS]
        q = str(it.get("question") or b.question).strip()
        if not state or not q:
            continue
        if b.type == "binary":
            if it.get("yes") is None:
                continue
            ans = 0 if it["yes"] else 1
        elif b.type == "choice":
            lab = str(it.get("label") or "").strip().lower()
            if lab not in keys:
                continue
            ans = keys.index(lab)
        else:
            ans = sorted({keys.index(x.lower()) for x in it.get("labels") or [] if x.lower() in keys})
            if not ans:
                continue
        out.append({"i": i, "state": state, "question": q, "answer": ans, "group": it.get("group")})
    return out


def trained_on(b: Bench) -> str:
    """Whether the training corpus uses this dataset."""
    try:
        from watersheep.sources import registry
        hits = [s for s in registry().values() if getattr(s, "hf", None) == b.hf]
    except Exception:
        return "unknown"
    if not hits:
        return "no"
    return "YES - same split" if any(s.split == b.split for s in hits) else "other split only"


def predict(ws, b: Bench, items: List[dict]):
    """Probabilities per item, or None if stopped."""
    from watersheep.metrics import sigmoid, softmax
    opts, T = b.texts(), ws.temps.get(b.type, 1.0)
    out = [None] * len(items)
    if len(opts) <= ws.cap:
        for k in range(0, len(items), 128):
            if interrupt.stopping():
                return None
            part = items[k:k + 128]
            zs, _ = ws._logits([{"type": b.type, "state": it["state"], "question": it["question"],
                                 "options": opts} for it in part])
            for j, z in enumerate(zs):
                if z is not None:
                    out[k + j] = sigmoid(z, T) if b.type == "multi" else softmax(z, T)
    for i, it in enumerate(items):
        if out[i] is None:
            if interrupt.stopping():
                return None
            out[i], _ = ws._probs(b.type, it["state"], it["question"], opts)
    return out


TEACHER_MAX_OPTIONS = 10


def teacher_dir_name(llm_model: str) -> str:
    return "teacher_" + re.sub(r"[^A-Za-z0-9.]+", "-", llm_model).strip("-")


def teacher_predict(t, b: Bench, items: List[dict]):
    """Teacher probabilities per item (None where unanswered), or None if stopped."""
    from watersheep.pool import pool_map
    from watersheep.synth import blind_probs, multi_probs
    opts = b.texts()

    def one(k: int):
        it = items[k]
        rec = {"type": b.type, "state": it["state"], "question": it["question"], "options": opts,
               "answer": it["answer"]}
        if b.type == "multi":
            return k, multi_probs(t, rec)
        rng = random.Random(stable_int("teacher-bench:%s:%s" % (b.name, it["i"])))
        return k, blind_probs(t, rec, 1, rng)

    out = [None] * len(items)
    for res in pool_map(one, range(len(items)), t.workers):
        if res:
            k, p = res
            out[k] = None if p is None else np.asarray(p, float)
    if interrupt.stopping():
        return None
    return out


def teacher_results() -> dict:
    """Saved teacher results by benchmark name."""
    dirs = sorted(OUT.glob("teacher_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        res = {f.stem: read_json(f) for f in d.glob("*.json") if f.stem not in ("summary",)}
        res = {k: v for k, v in res.items() if isinstance(v, dict) and "acc" in v}
        if res:
            return res
    return {}


def _cost(r: dict) -> int:
    return len(r.get("state") or "") + len(r["question"]) + sum(len(o) for o in r["options"])


def logits_for(ws, recs: List[dict], batch: int = 128):
    """Logits per record, batched by length."""
    order = sorted(range(len(recs)), key=lambda i: _cost(recs[i]))
    out = [None] * len(recs)
    for k in range(0, len(order), batch):
        if interrupt.stopping():
            return None
        idx = order[k:k + batch]
        zs, _ = ws._logits([recs[i] for i in idx])
        for i, z in zip(idx, zs):
            out[i] = z
    return out


def calibration_check(ws, recs: List[dict], logits) -> dict:
    """Reliability bins per question type, raw and calibrated."""
    from watersheep.metrics import _reliability, sigmoid, softmax
    res = {}
    for t in ("binary", "choice", "score", "multi"):
        sel = [(r, z) for r, z in zip(recs, logits) if r["type"] == t and z is not None]
        if not sel:
            continue
        res[t] = {}
        for tag, T in (("raw", 1.0), ("calibrated", ws.temps.get(t, 1.0))):
            conf, corr = [], []
            for r, z in sel:
                if t == "multi":
                    p = sigmoid(z, T)
                    y = np.zeros(len(p), bool)
                    y[list(r["answer"])] = True
                    pred = p >= 0.5
                    conf.extend(np.where(pred, p, 1 - p).tolist())
                    corr.extend((pred == y).astype(float).tolist())
                else:
                    p = softmax(z, T)
                    conf.append(float(p.max()))
                    corr.append(float(int(p.argmax()) == r["answer"]))
            ece, bins = _reliability(np.array(conf), np.array(corr), 15)
            res[t][tag] = {"ece": ece, "bins": bins, "n": len(conf)}
    return res


def accuracy_by_kind(ws, recs: List[dict], logits) -> dict:
    """Accuracy on synthetic and public test records."""
    from watersheep.metrics import sigmoid, softmax
    acc = {"synth": [0, 0], "real": [0, 0]}
    for r, z in zip(recs, logits):
        if z is None:
            continue
        T = ws.temps.get(r["type"], 1.0)
        if r["type"] == "multi":
            ok = tuple(np.where(sigmoid(z, T) >= ws.threshold)[0]) == tuple(sorted(r["answer"]))
        else:
            ok = int(softmax(z, T).argmax()) == r["answer"]
        a = acc["synth" if r.get("kind") == "synth" else "real"]
        a[0] += int(ok)
        a[1] += 1
    return {k: {"n": n, "acc": c / max(1, n)} for k, (c, n) in acc.items()}


def option_order_check(ws, recs: List[dict], perms: int = 3) -> dict:
    """Accuracy and prediction agreement under shuffled option order."""
    from watersheep.metrics import sigmoid, softmax
    rng = random.Random(7)
    res = {}
    for typ in ("choice", "multi"):
        rs = [r for r in recs if r["type"] == typ and len(r["options"]) >= 3]
        T = ws.temps.get(typ, 1.0)
        base, accs, agree = None, [], []
        for k in range(perms + 1):
            orders, shuffled = [], []
            for r in rs:
                order = list(range(len(r["options"])))
                if k:
                    rng.shuffle(order)
                orders.append(order)
                shuffled.append(dict(r, options=[r["options"][i] for i in order]))
            zs = logits_for(ws, shuffled)
            if zs is None:
                return {}
            outs, correct = [], 0
            for r, order, z in zip(rs, orders, zs):
                if z is None:
                    outs.append(None)
                    continue
                p = sigmoid(z, T) if typ == "multi" else softmax(z, T)
                back = np.empty(len(order))
                for pos, i in enumerate(order):
                    back[i] = p[pos]
                if typ == "multi":
                    sel = tuple(int(x) for x in np.where(back >= ws.threshold)[0])
                    correct += sel == tuple(sorted(r["answer"]))
                else:
                    sel = int(back.argmax())
                    correct += sel == r["answer"]
                outs.append(sel)
            accs.append(correct / max(1, sum(o is not None for o in outs)))
            if base is None:
                base = outs
            else:
                same = [a == b for a, b in zip(base, outs) if a is not None and b is not None]
                agree.append(sum(same) / max(1, len(same)))
        res[typ] = {"n": len(rs), "acc_original": accs[0], "acc_permuted": accs[1:],
                    "agreement_with_original": agree}
    return res


def speed_check(ws, recs: List[dict]) -> dict:
    """Single-request latency and batched throughput."""
    import torch
    words = ("the customer wrote again about the invoice and the delayed refund for order "
             "4411 which was charged twice last week while the support team was unavailable ").split()
    home = ws.device
    out = {"latency": []}
    devices = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
    for dev in devices:
        ws.net.to(dev)
        ws.device = torch.device(dev)
        reps = 60 if dev == "cuda" else 20
        for n_tok in (64, 256, 480):
            state = " ".join(words[i % len(words)] for i in range(int(n_tok * 0.75)))
            for k in (2, 4, 8):
                opts = list(YESNO) if k == 2 else ["option %d label" % i for i in range(k)]
                rec = {"type": "binary" if k == 2 else "choice", "state": state,
                       "question": "Which team should handle this request?", "options": opts}
                for _ in range(5):
                    ws._logits([rec])
                ts = []
                for _ in range(reps):
                    if interrupt.stopping():
                        break
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    ws._logits([rec])
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    ts.append(1000 * (time.perf_counter() - t0))
                _, tokens = ws._logits([rec])
                out["latency"].append({"device": dev, "input_tokens": tokens, "options": k,
                                       "median_ms": statistics.median(ts),
                                       "p90_ms": float(np.percentile(ts, 90))})
    ws.net.to(home)
    ws.device = home
    if home.type == "cuda":
        rs = recs[:4096]
        logits_for(ws, rs[:128])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits_for(ws, rs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out["throughput_gpu"] = {"records": len(rs), "batch": 128, "seconds": dt,
                                 "records_per_s": len(rs) / dt}
    return out


def analysis(ws, d: Path, force: bool) -> Optional[dict]:
    """Calibration, option-order and speed checks on the test split."""
    f = d / "analysis.json"
    saved = read_json(f)
    if saved and not force:
        LOG.info("analysis: saved earlier (%s)", f)
        return saved
    test = P.build / "test.jsonl"
    if not test.exists():
        LOG.warning("analysis skipped: %s not found (run the build stage)", test)
        return None
    recs = list(iter_jsonl(test))
    t0 = time.time()
    LOG.info("analysis: calibration on %d test records", len(recs))
    logits = logits_for(ws, recs)
    if logits is None:
        return None
    res = {"model": d.name, "test_records": len(recs), "temperatures": ws.temps,
           "calibration": calibration_check(ws, recs, logits),
           "accuracy_by_kind": accuracy_by_kind(ws, recs, logits)}
    LOG.info("analysis: option order (3 random permutations)")
    res["option_order"] = option_order_check(ws, recs)
    if interrupt.stopping():
        return None
    LOG.info("analysis: speed")
    res["speed"] = speed_check(ws, recs)
    if interrupt.stopping():
        return None
    res["created"] = now_iso()
    write_json(f, res)
    oo = res["option_order"]
    LOG.info("analysis done in %.0fs: ECE %s | option order: choice %.3f -> %s, multi %.3f -> %s",
             time.time() - t0,
             ", ".join("%s %.3f->%.3f" % (t, v["raw"]["ece"], v["calibrated"]["ece"])
                       for t, v in res["calibration"].items()),
             oo["choice"]["acc_original"], ["%.3f" % a for a in oo["choice"]["acc_permuted"]],
             oo["multi"]["acc_original"], ["%.3f" % a for a in oo["multi"]["acc_permuted"]])
    return res


def _f1(tp, fp, fn) -> float:
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def score(b: Bench, items: List[dict], probs, thr: float) -> dict:
    from watersheep.metrics import _reliability
    n, k = len(items), len(b.keys())
    if b.type == "multi":
        pr = np.array(probs)
        pred = pr >= thr
        y = np.zeros_like(pred)
        for r, it in zip(y, items):
            r[it["answer"]] = True
        tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
        per = [_f1((pred[:, j] & y[:, j]).sum(), (pred[:, j] & ~y[:, j]).sum(),
                   (~pred[:, j] & y[:, j]).sum()) for j in range(k) if y[:, j].any()]
        ece, _ = _reliability(np.where(pred, pr, 1 - pr).ravel(), (pred == y).ravel().astype(float), 15)
        sets = [tuple(it["answer"]) for it in items]
        top = max(set(sets), key=sets.count)
        return {"n": n, "acc": float((pred == y).all(1).mean()), "micro_f1": _f1(tp, fp, fn),
                "macro_f1": float(np.mean(per)), "option_acc": float((pred == y).mean()),
                "ece": ece, "majority": sets.count(top) / n, "threshold": thr}
    pr = [np.asarray(p) for p in probs]
    ans = np.array([it["answer"] for it in items])
    pred = np.array([int(p.argmax()) for p in pr])
    conf = np.array([p.max() for p in pr])
    correct = (pred == ans).astype(float)
    ece, _ = _reliability(conf, correct, 15)
    classes = sorted(set(ans) | set(pred))
    res = {"n": n, "acc": float(correct.mean()),
           "macro_f1": float(np.mean([_f1(((pred == c) & (ans == c)).sum(), ((pred == c) & (ans != c)).sum(),
                                           ((pred != c) & (ans == c)).sum()) for c in classes])),
           "ece": ece, "nll": float(np.mean([-np.log(max(1e-12, p[a])) for p, a in zip(pr, ans)])),
           "majority": float(np.bincount(ans, minlength=k).max() / n),
           "by_answer": {b.keys()[c]: int((ans == c).sum()) for c in range(k)}}
    if b.type == "binary":
        tp, fp, fn = ((pred == 0) & (ans == 0)).sum(), ((pred == 0) & (ans == 1)).sum(), ((pred == 1) & (ans == 0)).sum()
        res.update(f1=_f1(tp, fp, fn), precision=float(tp / max(1, tp + fp)), recall=float(tp / max(1, tp + fn)))
    groups = sorted({it["group"] for it in items if it.get("group")})
    if groups:
        res["by_group"] = {g: {"n": int(m.sum()), "acc": float(correct[m].mean())} for g in groups
                           for m in [np.array([it.get("group") == g for it in items])]}
    return res


HIST_COLS = ["model", "run_id", "benchmark", "suite", "type", "n", "acc", "macro_f1", "f1", "ece",
             "majority", "ms_per_question", "trained_on", "tested_at"]


def read_history() -> List[dict]:
    f = OUT / "history.csv"
    if not f.exists():
        return []
    with f.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_history(rows: List[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "history.csv.tmp"
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, HIST_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(OUT / "history.csv")


def hist_row(model: str, run_id: str, b: Bench, r: dict) -> dict:
    f4 = lambda x: "" if x is None else "%.4f" % x
    return {"model": model, "run_id": run_id, "benchmark": b.name, "suite": b.suite, "type": b.type,
            "n": r["n"], "acc": f4(r["acc"]), "macro_f1": f4(r.get("macro_f1")),
            "f1": f4(r.get("f1", r.get("micro_f1"))), "ece": f4(r.get("ece")),
            "majority": f4(r.get("majority")), "ms_per_question": "%.2f" % r["ms_per_question"],
            "trained_on": r["trained_on"], "tested_at": r["tested_at"]}


def summary(model: str, done: List[tuple], hist: List[dict], teacher: Optional[dict] = None,
            extra: Optional[dict] = None) -> str:
    prev = {}
    for h in hist:
        if h["model"] != model and h["model"].startswith("teacher_") == model.startswith("teacher_") and \
                h["model"] > prev.get(h["benchmark"], {}).get("model", ""):
            prev[h["benchmark"]] = h
    teacher = teacher or {}
    lines = ["# WaterSheep benchmarks - %s" % model, "",
             "Accuracy for multi-label is exact match (every option right); its F1 is micro-F1.",
             "Majority = accuracy of always giving the most common answer.",
             "Teacher = accuracy of the teacher LLM on the same questions (benchmark.py --teacher).", "",
             "| Benchmark | Suite | Type | N | Accuracy | Macro-F1 | F1 | ECE | Majority | Teacher | vs previous | ms/q |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    suites = {}
    for b, r in done:
        suites.setdefault(b.suite, []).append(r["acc"])
        p = prev.get(b.name)
        delta = "%+.1f (%s)" % (100 * (r["acc"] - float(p["acc"])), p["model"].split("_")[-1]) if p else "-"
        f1 = r.get("f1", r.get("micro_f1"))
        tr = teacher.get(b.name)
        lines.append("| %s | %s | %s | %d | %.1f%% | %.3f | %s | %.3f | %.1f%% | %s | %s | %.1f |" % (
            b.name, b.suite, b.type, r["n"], 100 * r["acc"], r["macro_f1"],
            "-" if f1 is None else "%.3f" % f1, r["ece"], 100 * r["majority"],
            "%.1f%%" % (100 * tr["acc"]) if tr else "-", delta, r["ms_per_question"]))
    lines += ["", "| Suite | Benchmarks | Mean accuracy |", "|---|---|---|"]
    for s, a in suites.items():
        lines.append("| %s | %d | %.1f%% |" % (s, len(a), 100 * np.mean(a)))
    if suites:
        lines.append("| **all (mean of suites)** | %d | **%.1f%%** |" % (
            len(done), 100 * np.mean([np.mean(a) for a in suites.values()])))
    weak = [(b, r) for b, r in done if r.get("by_group")]
    if weak:
        lines += ["", "## Weakest groups", ""]
        for b, r in weak:
            worst = sorted(r["by_group"].items(), key=lambda kv: kv[1]["acc"])[:4]
            lines.append("- **%s**: " % b.name + ", ".join("%s %.0f%% (n=%d)" % (g, 100 * v["acc"], v["n"])
                                                          for g, v in worst))
    if extra:
        lines += ["", "## Analysis (test split, `analysis.json`)", ""]
        cal = extra.get("calibration") or {}
        if cal:
            lines.append("- ECE raw -> calibrated: " + ", ".join(
                "%s %.3f -> %.3f" % (t, v["raw"]["ece"], v["calibrated"]["ece"]) for t, v in cal.items()))
        for t, v in (extra.get("option_order") or {}).items():
            lines.append("- option order, %s (n=%d): accuracy %.1f%% original, %s shuffled; same prediction "
                         "%s" % (t, v["n"], 100 * v["acc_original"],
                                 " / ".join("%.1f%%" % (100 * x) for x in v["acc_permuted"]),
                                 " / ".join("%.1f%%" % (100 * x) for x in v["agreement_with_original"])))
        sp = extra.get("speed") or {}
        for dev in ("cuda", "cpu"):
            ms = [x["median_ms"] for x in sp.get("latency", []) if x["device"] == dev]
            if ms:
                lines.append("- single request on %s: %.0f to %.0f ms (median)" % (dev, min(ms), max(ms)))
        if sp.get("throughput_gpu"):
            lines.append("- GPU throughput: %.0f test records/s in batches of %d" % (
                sp["throughput_gpu"]["records_per_s"], sp["throughput_gpu"]["batch"]))
    lines += ["", "## Sources", ""] + ["- **%s** - %s. `%s`%s, split %s. Trained on: %s." % (
        b.name, b.about, b.hf, "/" + b.config if b.config else "", b.split, r["trained_on"]) for b, r in done]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="export folder, export name or Hugging Face repo id "
                                    "(default: the newest export, out/export/LATEST)")
    ap.add_argument("--only", help="comma-separated benchmark names or suites")
    ap.add_argument("--limit", type=int, default=2000, help="questions per benchmark, 0 = all (default 2000)")
    ap.add_argument("--force", action="store_true", help="run again even if saved for this model")
    ap.add_argument("--list", action="store_true", help="list the benchmarks and stop")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU")
    ap.add_argument("--no-analysis", action="store_true",
                    help="skip the calibration / option-order / speed checks on the test split")
    ap.add_argument("--teacher", action="store_true",
                    help="score the teacher LLM (config llm_model) on the same questions instead")
    a = ap.parse_args()
    if a.teacher and a.model:
        print("--teacher and --model exclude each other")
        return 2

    if a.list:
        for b in BENCHMARKS:
            print("%-44s %-9s %-6s %s" % (b.name, b.suite, b.type, b.about))
        return 0
    chosen = BENCHMARKS
    if a.only:
        want = {w.strip() for w in a.only.split(",") if w.strip()}
        chosen = [b for b in BENCHMARKS if b.name in want or b.suite in want or
                  b.name.replace("legal_", "") in want]
        if not chosen:
            print("no benchmark matches %r - see --list" % a.only)
            return 2

    pin_caches()
    setup_logging(P.logs / "benchmark.log")
    interrupt.install()
    ws = t = None
    if a.teacher:
        from watersheep import llm
        from watersheep.config import Config
        cfg = Config.load()
        model, run_id = teacher_dir_name(cfg.llm_model), cfg.llm_model
        t = llm.get_teacher(cfg)
        threshold = 0.5
        where = "%s, %d parallel slots" % (t.b.name, t.b.slots)
    else:
        from watersheep.infer import WaterSheep
        try:
            ws = WaterSheep.load(a.model, device="cpu" if a.cpu else None)
        except FileNotFoundError as e:
            LOG.error("%s", e)
            return 1
        model = ws.name.replace("watersheep-", "watersheep_")
        run_id = ws.meta.get("run_id", "")
        threshold = ws.threshold
        where = "%s%s" % (ws.device, "" if a.model else ", newest export")
    d = OUT / model
    d.mkdir(parents=True, exist_ok=True)
    LOG.info("benchmarking %s (run %s) on %s - %d benchmarks, results in %s",
             model, run_id, where, len(chosen), d)

    hist = read_history()
    done, failed = [], []
    t_all = time.time()
    for n, b in enumerate(chosen, 1):
        if interrupt.stopping():
            break
        f = d / (b.name + ".json")
        saved = read_json(f)
        if saved and not a.force and saved.get("limit") == a.limit:
            done.append((b, saved))
            LOG.info("[%d/%d] %s: saved earlier - accuracy %.1f%%", n, len(chosen), b.name, 100 * saved["acc"])
            continue
        try:
            items = items_for(b, a.limit)
        except Exception as e:
            if interrupt.stopping():
                break
            LOG.warning("[%d/%d] %s: could not load (%s)", n, len(chosen), b.name, str(e).splitlines()[0][:120])
            failed.append(b.name)
            continue
        if not items:
            LOG.warning("[%d/%d] %s: no usable rows", n, len(chosen), b.name)
            failed.append(b.name)
            continue
        if t is not None and len(b.keys()) > TEACHER_MAX_OPTIONS:
            LOG.warning("[%d/%d] %s: skipped for the teacher (more than %d options)", n, len(chosen),
                        b.name, TEACHER_MAX_OPTIONS)
            continue
        t0 = time.time()
        probs = teacher_predict(t, b, items) if t is not None else predict(ws, b, items)
        if probs is None:
            break
        dt = time.time() - t0
        n_all = len(items)
        keep = [k for k, p in enumerate(probs) if p is not None]
        items, probs = [items[k] for k in keep], [probs[k] for k in keep]
        if not items:
            failed.append(b.name)
            continue
        r = score(b, items, probs, threshold)
        r.update(benchmark=b.name, suite=b.suite, type=b.type, hf=b.hf, config=b.config, split=b.split,
                 model=model, run_id=run_id, limit=a.limit, ms_per_question=1000 * dt / n_all,
                 unanswered=n_all - len(items), trained_on=trained_on(b), tested_at=now_iso())
        keys, thr = b.keys(), threshold
        write_jsonl(d / (b.name + ".jsonl"), [{
            "row": it["i"], "state": it["state"], "question": it["question"], "group": it["group"],
            "truth": [keys[j] for j in it["answer"]] if b.type == "multi" else keys[it["answer"]],
            "answer": [kk for kk, p in zip(keys, pr) if p >= thr] if b.type == "multi" else keys[int(np.argmax(pr))],
            "probabilities": {kk: round(float(p), 4) for kk, p in zip(keys, pr)}}
            for it, pr in zip(items, probs)])
        write_json(f, r)
        done.append((b, r))
        LOG.info("[%d/%d] %s: accuracy %.1f%%, macro-F1 %.3f (always-majority %.1f%%), %d questions, "
                 "%.1f ms each", n, len(chosen), b.name, 100 * r["acc"], r["macro_f1"],
                 100 * r["majority"], r["n"], r["ms_per_question"])

    extra = None
    if ws is not None and not a.no_analysis and not interrupt.stopping():
        extra = analysis(ws, d, a.force)
    if t is not None:
        from watersheep import llm
        llm.shutdown()

    names = {b.name for b, _ in done}
    hist_new = [h for h in hist if not (h["model"] == model and h["benchmark"] in names)]
    write_history(hist_new + [hist_row(model, run_id, b, r) for b, r in done])
    md = summary(model, done, hist, teacher=None if t is not None else teacher_results(), extra=extra)
    (d / "summary.md").write_text(md, encoding="utf-8")
    write_json(d / "summary.json", {"model": model, "run_id": run_id, "created": now_iso(),
                                    "failed": failed, "results": {b.name: r for b, r in done}})
    print("\n" + md)
    if failed:
        LOG.warning("not run: %s", ", ".join(failed))
    if interrupt.stopping():
        LOG.info("stopped - run again to finish the rest (finished benchmarks are kept)")
    LOG.info("done in %.0fs - summary: %s", time.time() - t_all, d / "summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
