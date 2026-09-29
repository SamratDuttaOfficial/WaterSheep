"""Corpus, synthetic-data and training statistics."""
from __future__ import annotations
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np                                                     # noqa: E402

from watersheep.core.util import iter_jsonl                            # noqa: E402

OUT = ROOT / "results" / "data"
PRIVATE = {"ex_per_s", "gpu_mem_gb", "t", "batch_tokens", "train_seconds", "best", "last"}

CATEGORY = {
    "Safety, abuse and fraud": """aegis_categories aegis_safety civil_attributes civil_comments hate_offensive jailbreak
        jigsaw_toxic_types moderation_categories prompt_injection prosocial spml_injection
        toxic_conversations cyberbullying clickbait sms_spam spam_messages youtube_spam phishing
        job_fraud""",
    "Grounding, retrieval and fact checking": """boolq squad_v2 esci halueval cosqa qnli vitaminc liar2 pubmedqa""",
    "Agents, tools and customer support": """tool_choice support_intent support_routing banking77 clinc massive_intent massive_scenario snips
        hwu64 counterfactual""",
    "Judging and preference": """preference hh_rlhf reward_bench mt_bench_human orca_pairs preference_rubric rubric_score
        helpsteer helpsteer1""",
    "Reasoning, inference and knowledge": """bigbench copa wsc mnli snli scitail qasc winogrande quartz folio truthfulqa mmlu mmlu_pro
        arc_challenge arc_easy openbookqa commonsense_qa medmcqa medqa paws""",
    "Law, medicine and finance": """case_hold ledgar unfair_tos unfair_tos_types echr_alleged echr_violations medical_abstracts
        symptom_diagnosis medical_specialty pubmed_topics fin_sentiment fin_topics""",
    "Ethics": "ethics_commonsense ethics_deontology ethics_justice ethics_utility ethics_virtue moral_stories",
    "Sentiment, emotion and ratings": """go_emotions go_emotions_multi poem_sentiment clothing_fit clothing_rating clothing_recommend
        entity_sentiment food_reviews phone_reviews sarcasm review_aspects""",
    "Topic and document classification": """dbpedia student_questions arxiv_fields stackoverflow_tags ecommerce news_category
        news_aggregator movie_genres resume_category so_quality cf_media_audience cf_media_bias
        cf_media_message""",
}
CAT_OF = {n: c for c, names in CATEGORY.items() for n in names.split()}


def short(name: str) -> str:
    return name.split(":", 1)[1]


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import export_meta, model_arg
    args = model_arg("Corpus, synthetic-data and training statistics (newest export unless --model).")
    run = export_meta(args.model)["run_id"]
    reg = json.load(open(ROOT / "state/registry.json"))
    tok_dir = Path(next(i["tok_dir"] for i in reg["items"] if i["id"] == run))
    src = json.load(open(ROOT / "out/reports/sources.json"))
    audit_all = json.load(open(ROOT / "out/reports/audit.json"))
    stats = json.load(open(ROOT / "data/build/stats.json"))
    per_build = stats["per_source"]

    import watersheep.data.sources as S
    reg = S.REGISTRY
    audit = {k: v for k, v in audit_all["per_source"].items() if k in reg}
    rows = []
    missing = []
    for name, n in sorted(src["counts"].items()):
        s = reg.get(name)
        if s is None:
            continue
        cat = CAT_OF.get(short(name))
        if cat is None:
            missing.append(name)
        rows.append({"name": name, "type": s.type if s else "?", "records": n,
                     "origin": s.origin if s else "?", "ref": s.ref if s else "",
                     "category": cat, "audit_agree": (audit.get(name) or {}).get("agree"),
                     "audit_chance": (audit.get(name) or {}).get("chance"),
                     "in_build": per_build.get(name, 0)})
    assert not missing, missing
    cats = collections.OrderedDict()
    for c in CATEGORY:
        rs = [r for r in rows if r["category"] == c]
        cats[c] = {"sources": len(rs), "records": sum(r["records"] for r in rs),
                   "by_type": dict(collections.Counter(r["type"] for r in rs))}

    fam = collections.defaultdict(lambda: collections.Counter())
    fam_type = {}
    diff = collections.defaultdict(collections.Counter)
    typ = collections.defaultdict(collections.Counter)
    quality, p_ans, words, relabeled = [], [], [], 0
    domains = collections.Counter()
    for f in sorted((ROOT / "data/synth").glob("shard_*.jsonl")):
        for row in iter_jsonl(f):
            fam[row["family"]][row["status"]] += 1
            fam_type[row["family"]] = row["type"]
            typ[row["type"]][row["status"]] += 1
            rec = row.get("record") or {}
            meta = rec.get("meta") or {}
            if meta.get("difficulty"):
                diff[meta["difficulty"]][row["status"]] += 1
            if row["status"] == "accepted":
                quality.append(meta.get("quality"))
                p_ans.append(meta.get("p_answer"))
                words.append(len(rec.get("state", "").split()))
                domains[meta.get("domain")] += 1
                relabeled += bool(meta.get("relabeled"))
    families = {k: {"type": fam_type[k], "tried": sum(v.values()), "accepted": v["accepted"]}
                for k, v in fam.items()}
    synth = {"families": families,
             "by_type": {k: {"tried": sum(v.values()), "accepted": v["accepted"]} for k, v in typ.items()},
             "by_difficulty": {k: {"tried": sum(v.values()), "accepted": v["accepted"]} for k, v in diff.items()},
             "quality_mean": statistics.mean(q for q in quality if q is not None),
             "quality_median": statistics.median(q for q in quality if q is not None),
             "p_answer_mean": statistics.mean(p for p in p_ans if p is not None),
             "state_words_mean": statistics.mean(words), "state_words_median": statistics.median(words),
             "relabeled_multi": relabeled, "domains": len(domains),
             "report": json.load(open(ROOT / "out/reports/synth.json"))}

    z = np.load(tok_dir / "train.npz")
    lens = z["lens"]
    nopt = np.diff(z["poff"])
    tok = {"train_examples": int(len(lens)), "tokens_total": int(lens.sum()),
           "len_mean": float(lens.mean()), "len_median": float(np.median(lens)),
           "len_p95": float(np.percentile(lens, 95)), "truncated_share": float((lens >= 512).mean()),
           "options_mean": float(nopt.mean())}

    curve, val = [], []
    for r in iter_jsonl(ROOT / "logs/metrics" / (run + ".jsonl")):
        if r.get("val"):
            val.append({k: r[k] for k in ("step", "acc", "nll", "ece", "brier", "by_type")})
        else:
            curve.append({k: r.get(k) for k in ("step", "epoch", "loss", "lr", "grad_norm", "ex_per_s",
                                                  "gpu_mem_gb", "t")})
    summ = json.load(open(ROOT / "checkpoints" / run / "summary.json"))
    exs = [c["ex_per_s"] for c in curve if c["step"] > 20]
    train = {"summary": summ, "ex_per_s_median": statistics.median(exs),
             "gpu_mem_peak_gb": max(c["gpu_mem_gb"] for c in curve), "curve": curve, "val": val}

    out = {"sources": rows, "categories": cats, "build": {k: stats[k] for k in
                                                         ("splits", "types", "synthetic_train",
                                                          "duplicates", "invalid")},
           "synth": synth, "tokens": tok, "train": train,
           "audit": {"sources": len(audit), "dropped": [s for s in audit_all["dropped"] if s in reg],
                     "agree_median": statistics.median(v["agree"] for v in audit.values()),
                     "agree_mean": statistics.mean(v["agree"] for v in audit.values()),
                     "items": sum(v["n"] for v in audit.values())}}
    public = dict(out, train={"summary": {k: v for k, v in summ.items() if k not in PRIVATE},
                              "curve": [{k: v for k, v in c.items() if k not in PRIVATE} for c in curve],
                              "val": val})
    del public["tokens"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data_stats.json").write_text(json.dumps(public, indent=1), encoding="utf-8")
    print(json.dumps({"categories": cats, "synth": {k: v for k, v in synth.items() if k != "families"},
                      "tokens": tok, "audit": out["audit"],
                      "train": {k: v for k, v in train.items() if k not in ("curve", "val")}}, indent=1))


if __name__ == "__main__":
    main()
