#!/usr/bin/env python
"""Offline end-to-end test on toy data.

  python tests/selftest.py
  python tests/selftest.py --real-teacher
"""
from __future__ import annotations
import os
import shutil
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
HOME = HERE / ".selftest"
os.environ["WATERSHEEP_HOME"] = str(HOME)  # set before importing watersheep

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def unit_tests():
    print("\n== unit tests")
    from watersheep.data import sources as S
    from watersheep.synth.llm import label_probs, loose_json
    from watersheep.data.schema import validate
    from watersheep.synth.generate import Deduper, accept_blind, soft_target, to_record
    from watersheep.core.config import Config
    cfg = Config()

    c = S.Ctx("t", 1, 0, {"label": ["Company", "EducationalInstitution", "Artist", "Athlete"]}, {})
    m = S.REGISTRY["hf:dbpedia"].map({"title": "MIT", "content": "A university.", "label": 1}, c)
    check(m and m["options"][m["answer"]] == "Educational Institution",
          "dbpedia maps label 1 -> educational institution")
    m = S.REGISTRY["hf:boolq"].map({"question": "is the sky blue", "passage": "It is.",
                                    "answer": False}, c)
    check(m and m["answer"] == 1 and m["question"].endswith("?"), "boolq False -> 'no'")
    m = S.REGISTRY["hf:qnli"].map({"question": "Where did it sit?", "sentence": "It sat on the mat.",
                                   "label": 0}, c)
    check(m and m["options"][m["answer"]] == "yes", "qnli entailment -> yes")
    m = S.REGISTRY["hf:mnli"].map({"premise": "p", "hypothesis": "h", "label": 2}, c)
    check(m and m["options"][m["answer"]] == "false", "nli contradiction -> false")
    m = S.REGISTRY["kg:phone_reviews"].map({"Reviews": "meh", "Rating": 1}, c)
    check(m and m["options"][m["answer"]] == "1", "phone_reviews rating 1 -> 1 star")
    m = S.REGISTRY["hf:civil_comments"].map({"text": "x", "toxicity": 0.3}, c)
    check(m is None, "civil_comments skips the ambiguous band")
    c2 = S.Ctx("t", 2, 0, {}, {"label_text": ["card_arrival", "top_up_failed", "lost_card",
                                             "pin_blocked", "refund", "fx_rate", "cash", "atm",
                                             "fees", "transfer"]})
    m = S.REGISTRY["hf:banking77"].map({"text": "where is my card", "label_text": "card_arrival"}, c2)
    check(m and len(m["options"]) == 8 and m["options"][m["answer"]] == "card arrival",
          "banking77 subsets to 8 options incl. the right one")
    uf = {"prompt": "hi", "chosen": [{"role": "user", "content": "hi"},
                                     {"role": "assistant", "content": "GOOD"}],
          "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "BAD"}],
          "score_chosen": 8.0, "score_rejected": 3.0}
    m = S.REGISTRY["hf:preference"].map(uf, c)
    good_first = m["state"].index("GOOD") < m["state"].index("BAD")
    check(m and (m["answer"] == 0) == good_first, "preference points at the chosen response")

    check(loose_json('```json\n{"a": 1,}\n```') == {"a": 1}, "loose_json repairs fences/commas")
    check(loose_json('noise {"a": {"b": "}"}} tail') == {"a": {"b": "}"}}, "loose_json brackets")
    r = {"content": "B", "completion_probabilities": [{"top_logprobs": [
        {"token": "B", "logprob": -0.1}, {"token": "A", "logprob": -2.5},
        {"token": " C", "logprob": -4.0}, {"token": "The", "logprob": -5}]}]}
    p = label_probs(r, ["A", "B", "C"])
    check(p and abs(sum(p) - 1) < 1e-6 and p[1] > 0.85, "label_probs reads llama-server top_logprobs")

    job = {"jid": "t1", "type": "choice", "family": "f", "domain": "d", "difficulty": "easy",
           "n_options": 3}
    rec, why = to_record(job, {"state": "Some long enough context about a thing." * 2,
                               "question": "Which is best here?", "options": ["a", "b", "c"],
                               "answer": 2, "why": "x"}, cfg)
    check(rec is not None and validate(rec)[0], "to_record builds a valid choice record")
    ok, pa = accept_blind(rec, [0.1, 0.2, 0.7], 0.5)
    check(ok and abs(pa - 0.7) < 1e-9, "blind check accepts when teacher agrees")
    ok, _ = accept_blind(rec, [0.6, 0.1, 0.3], 0.5)
    check(not ok, "blind check rejects when teacher disagrees")
    t = soft_target(rec, [0.1, 0.2, 0.7], 0.6)
    check(abs(sum(t) - 1) < 1e-6 and t[2] > 0.85, "soft target mixes one-hot and teacher")

    d = Deduper(3)
    a = {"state": "The printer on floor 3 is jammed again and nobody can print invoices today.",
         "question": "Which team?"}
    b = dict(a, state=a["state"] + " ")
    d.add(a)
    check(d.seen(b), "dedup catches a near-duplicate")
    check(not d.seen({"state": "Completely different text about payroll dates and taxes.",
                      "question": "Is it late?"}), "dedup keeps different text")


def watch_and_stop(step: int):
    """Request a stop once training passes `step`."""
    from watersheep.core import interrupt
    from watersheep.core.paths import P
    from watersheep.core.util import iter_jsonl

    def loop():
        while not interrupt.stopping():
            for f in P.metrics.glob("*.jsonl"):
                if any(r.get("step", 0) >= step for r in iter_jsonl(f)):
                    print("\n  [selftest] simulating Ctrl+C at step >= %d" % step, flush=True)
                    interrupt.STOP.set()
                    return
            time.sleep(0.2)
    threading.Thread(target=loop, daemon=True).start()


def pipeline(real_teacher: bool):
    print("\n== pipeline (in %s)" % HOME)
    from watersheep import run
    from watersheep.core import interrupt
    from watersheep.core.paths import P
    from watersheep.core.util import read_json
    import torch
    sets = ["sources=toy:stock,toy:cheapest,toy:delay,toy:tags", "use_kaggle=false",
            "per_source_max=500", "heldout_sources=toy:delay", "audit_per_source=10",
            "audit_drop_below_chance=false", "webui=false",
            "llm_backend=%s" % ("auto" if real_teacher else "fake"),
            "synth_target=%d" % (8 if real_teacher else 80), "synth_shard=%d" % (8 if real_teacher else 40),
            "encoder=tiny-test", "max_len=128", "max_question_tokens=48", "max_option_tokens=16",
            "epochs=6", "eval_every=40", "ckpt_every=10", "target_batch_examples=8",
            "early_stop_patience=100", "auto_install=false",
            "device=%s" % ("cuda" if torch.cuda.is_available() else "cpu")]
    run.main(sum([["--set", s] for s in sets], []))

    watch_and_stop(1)
    rc = run.main(["--until", "train"])
    check(rc == 130, "simulated Ctrl+C pauses the run (rc=130, got %s)" % rc)
    man = read_json(P.manifest, {})
    check(man["stages"]["train"]["status"] == "stopped", "train marked 'stopped', not failed")
    last = next(P.ckpt.glob("*/last.pt"), None)
    check(last is not None, "checkpoint written at the pause")
    paused = torch.load(last, map_location="cpu", weights_only=False)["state"] if last else {}
    interrupt.reset()

    rc = run.main([])
    check(rc == 0, "resumed run completes (rc=%s)" % rc)
    st = torch.load(next(P.ckpt.glob("*/last.pt")), map_location="cpu", weights_only=False)["state"]
    check(0 < paused.get("step", 0) < st["step"],
          "training resumed from step %s and finished at step %d" % (paused.get("step"), st["step"]))
    check(len(list(P.ckpt.glob("run_*"))) == 1, "the resume reused the same run (no restart)")
    summ = read_json(next(P.ckpt.glob("run_*/summary.json")), {})
    check(summ.get("steps") == summ.get("total_steps"),
          "training ran its full LR schedule (%s/%s steps)" % (summ.get("steps"), summ.get("total_steps")))
    man = read_json(P.manifest, {})
    check(all(man["stages"][s]["status"] == "done" for s in
              ("sources", "audit", "synth", "build", "train", "calibrate", "evaluate", "export")),
          "every stage done")
    syn = read_json(P.reports / "synth.json", {})
    check(syn.get("accepted", 0) >= (8 if real_teacher else 80) or real_teacher,
          "synthetic target reached (%s accepted)" % syn.get("accepted"))
    ev = next(P.reports.glob("eval_*.json"), None)
    check(ev is not None, "evaluation report written")
    check((P.plots / "dashboard.png").exists(), "dashboard plot rendered")

    from watersheep.infer import WaterSheep
    ws = WaterSheep.load()
    r1 = ws.decide("Warehouse stock: 50 cables. New order: 10 cables.",
                   "Can the order be shipped from current stock?")
    r2 = ws.decide("Quotes received: Contoso $40; Adatum $12.", "Which supplier has the lowest quote?",
                   ["Contoso", "Adatum"])
    r3 = ws.decide("The parcel was delivered 7 days late.",
                   "Rate the delay from 1 (under 2 days) to 5 (9 or more days).", list("12345"))
    check(r1["type"] == "binary" and 0 <= r1["p_yes"] <= 1, "binary decision: %s" % r1["answer"])
    check(r2["type"] == "choice" and abs(sum(r2["probs"].values()) - 1) < 1e-4,
          "choice decision: %s" % r2["answer"])
    check(r3["type"] == "score" and 1 <= r3["expected"] <= 5, "score decision: %.2f" % r3["expected"])
    r4 = ws.decide("Packing slip for box 7: cables, desks.", "Which of these items are in the box?",
                   ["cables", "chairs", "desks"], "multi")
    check(r4["type"] == "multi" and all(0 <= v <= 1 for v in r4["probs"].values()),
          "multi-label decision: %s" % r4["answer"])

    many = {"label %02d" % k: "description %d" % k for k in range(40)}
    res = ws.ask({"state": {"customer": "Alex", "message": "Charged twice for order 4411"},
                  "questions": {
                      "urgent": {"type": "noul", "instructions": "Does this need attention today?",
                                 "criteria": {"true": "money or access is blocked", "false": "it can wait"}},
                      "team": {"type": "choice", "instructions": "Which team should handle this?",
                               "criteria": {"billing": "payments, refunds", "technical": "bugs",
                                            "sales": "pricing"}},
                      "mood": {"type": "score", "instructions": "How frustrated is the customer?",
                               "criteria": ["calm", "annoyed", "frustrated", "furious"]},
                      "tags": {"type": "multi", "instructions": "Which tags apply?",
                               "criteria": ["refund", "double charge", "shipping"]},
                      "big": {"type": "choice", "instructions": "Which label fits?", "criteria": many},
                      "bad": {"type": "choice", "instructions": "x", "criteria": ["only one"]}}})
    a = res.get("answers", {})
    check(set(res) == {"model", "answers", "usage"} and res["usage"]["input_tokens"] > 0,
          "response envelope (model, answers, usage)")
    check(a["urgent"]["type"] == "noul" and 0 <= a["urgent"]["noul"] <= 1, "noul answer")
    check(a["team"]["choice"] in ("billing", "technical", "sales") and
          abs(sum(a["team"]["probabilities"].values()) - 1) < 1e-3 and 0 <= a["team"]["confidence"] <= 1,
          "choice answer")
    check(0 <= a["mood"]["score"] <= 3 and a["mood"]["legend"]["3"] == "furious" and
          len(a["mood"]["probabilities"]) == 4, "score answer (legend, fractional level)")
    check(a["tags"]["type"] == "multi" and set(a["tags"]["probabilities"]) ==
          {"refund", "double charge", "shipping"}, "multi-label answer")
    check(len(a["big"]["probabilities"]) == 40 and abs(sum(a["big"]["probabilities"].values()) - 1) < 1e-3,
          "40-label choice via tournament")
    check("error" in a["bad"], "invalid question reports an error instead of failing the request")

    before = syn.get("accepted", 0)
    if not real_teacher:
        want = before + 60
        rc = run.main(["--set", "synth_target=%d" % want, "--from", "synth", "--until", "build"])
        syn = read_json(P.reports / "synth.json", {})
        check(rc == 0 and syn.get("accepted", 0) >= want,
              "raising synth_target continues generation (%d -> %d)" % (before, syn.get("accepted", 0)))
        n_done = len(list(P.synth.glob("shard_*.done")))
        check(n_done == len(list(P.synth.glob("shard_*.jsonl"))) and not list(P.synth.glob("*.part")),
              "every synthetic shard closed cleanly (%d shards)" % n_done)


def main():
    real = "--real-teacher" in sys.argv
    if HOME.exists():
        shutil.rmtree(HOME, ignore_errors=True)
    t0 = time.time()
    unit_tests()
    pipeline(real)
    print("\n%s in %.0fs" % ("ALL PASSED" if not FAILS else "%d FAILED" % len(FAILS), time.time() - t0))
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
