"""Public datasets mapped to decision records."""
from __future__ import annotations
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import interrupt
from .schema import YESNO, answers, make, score_options, validate
from .util import LOG, stable_int


def val(row: dict, col: str):
    v = row.get(col)
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def txt(row: dict, *cols) -> str:
    for c in cols:
        v = val(row, c)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


class Ctx:
    def __init__(self, name: str, idx, seed: int, names: dict, values: dict, config=None):
        self.rng = random.Random(stable_int("%s:%s:%s" % (name, idx, seed)))
        self.names, self.values, self.config = names, values, config

    def pick(self, templates):
        return self.rng.choice(templates) if isinstance(templates, (list, tuple)) else templates

    def label_name(self, col: str, v) -> Optional[str]:
        names = self.names.get(col)
        if names and isinstance(v, int) and not isinstance(v, bool) and 0 <= v < len(names):
            return names[v]
        return None if v is None else str(v)


@dataclass
class Src:
    name: str
    type: str
    map: Callable
    hf: str = ""
    config: Optional[str] = None
    split: str = "train"
    kaggle: str = ""
    url: str = ""
    files: tuple = ()
    all_files: bool = False
    csv_kw: dict = field(default_factory=dict)
    collect: tuple = ()
    configs: object = ()
    stream: bool = False
    scale: float = 1.0
    prep: Optional[Callable] = None
    version: int = 1
    note: str = ""

    @property
    def is_kaggle(self) -> bool:
        return bool(self.kaggle)

    @property
    def origin(self) -> str:
        return ("kaggle" if self.kaggle else "web" if self.url else
                "huggingface" if self.hf else "built-in")

    @property
    def ref(self) -> str:
        return self.hf or self.kaggle or self.url


def subset(correct: str, pool: List[str], k: int, rng: random.Random) -> List[str]:
    if len(pool) <= k:
        return list(pool)
    others = [o for o in pool if o != correct]
    opts = rng.sample(others, k - 1) + [correct]
    rng.shuffle(opts)
    return opts


def humanize(s: str) -> str:
    s = str(s).replace("_", " ").replace("-", " ").strip()
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def binary(text_cols, label_col, pos, questions, state=None):
    """Mapper for yes/no datasets."""
    def m(row, c):
        v = val(row, label_col)
        if v is None:
            return None
        yes = pos(v) if callable(pos) else (v == pos)
        if yes is None:
            return None
        s = state(row) if state else txt(row, *text_cols)
        if not s:
            return None
        return {"state": s, "question": c.pick(questions), "options": YESNO,
                "answer": 0 if yes else 1}
    return m


def classes(text_cols, label_col, questions, *, from_values=False, rename=None,
            skip=(), max_opts=8, state=None, keep=None):
    def m(row, c):
        v = val(row, label_col)
        if v is None:
            return None
        if from_values:
            pool, name = c.values.get(label_col) or [], str(v)
        else:
            pool, name = c.names.get(label_col) or [], c.label_name(label_col, v)
        if keep:
            pool = [n for n in pool if keep(n)]
        if not pool or name is None or name in skip or (keep and not keep(name)):
            return None
        pretty = (lambda x: rename(x)) if callable(rename) else \
                 (lambda x: (rename or {}).get(x, humanize(x)))
        opts_all = [pretty(n) for n in pool if n not in skip]
        correct = pretty(name)
        if correct not in opts_all:
            return None
        s = state(row) if state else txt(row, *text_cols)
        if not s:
            return None
        opts = subset(correct, opts_all, max_opts, c.rng)
        return {"state": s, "question": c.pick(questions), "options": opts,
                "answer": opts.index(correct)}
    return m


NLI_OPTS = ["true", "impossible to tell", "false"]
NLI_QS = ["Given the premise, is this hypothesis true, false, or impossible to tell?\n"
          "Hypothesis: {h}",
          "Does the premise make the following statement true or false, or can't we tell?\n"
          "Statement: {h}"]


def nli(p_col, h_col, label_col, order=(0, 1, 2)):
    """Mapper for entailment datasets."""
    def m(row, c):
        v = val(row, label_col)
        if v not in order:
            return None
        p, h = txt(row, p_col), txt(row, h_col)
        if not p or not h:
            return None
        return {"state": "Premise: " + p, "question": c.pick(NLI_QS).format(h=h),
                "options": NLI_OPTS, "answer": order.index(v)}
    return m


def score(text_cols, label_col, lo, hi, questions, conv=None, state=None):
    def m(row, c):
        v = val(row, label_col)
        if v is None:
            return None
        try:
            s_val = conv(v) if conv else int(v)
        except (TypeError, ValueError):
            return None
        if s_val is None or not lo <= s_val <= hi:
            return None
        s = state(row) if state else txt(row, *text_cols)
        if not s:
            return None
        return {"state": s, "question": c.pick(questions),
                "options": score_options(lo, hi), "answer": s_val - lo}
    return m


def mcq_choices(q_col, choices_col, key_col, state_col=None):
    """Options and answer from a choices column."""
    def m(row, c):
        ch = val(row, choices_col) or {}
        texts, labs = list(ch.get("text") or []), [str(x) for x in (ch.get("label") or [])]
        key = str(val(row, key_col) or "")
        if len(texts) < 2 or key not in labs:
            return None
        return {"state": txt(row, state_col) if state_col else "",
                "question": txt(row, q_col), "options": texts, "answer": labs.index(key)}
    return m


def _mmlu(row, c):
    if isinstance(row.get("train"), dict):
        row = row["train"]
    ch = row.get("choices") or []
    a = row.get("answer")
    if len(ch) < 2 or not isinstance(a, int) or not 0 <= a < len(ch):
        return None
    return {"state": "", "question": txt(row, "question"), "options": list(ch), "answer": a}


def _last_assistant(msgs) -> str:
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "assistant":
                return str(m.get("content") or "").strip()
    return ""


def _ultrafeedback(row, c):
    sc, sr = val(row, "score_chosen"), val(row, "score_rejected")
    if sc is None or sr is None or sc == sr:
        return None
    good, bad = _last_assistant(row.get("chosen")), _last_assistant(row.get("rejected"))
    if not good or not bad or good == bad:
        return None
    first_good = c.rng.random() < 0.5
    r1, r2 = (good, bad) if first_good else (bad, good)
    cut = lambda s: s if len(s) <= 1100 else s[:1100].rstrip() + " ..."
    state = ("User request:\n%s\n\nResponse 1:\n%s\n\nResponse 2:\n%s"
             % (cut(txt(row, "prompt")), cut(r1), cut(r2)))
    return {"state": state, "question": c.pick(["Which response is better?",
                                                "Which response answers the request better?"]),
            "options": ["Response 1", "Response 2"], "answer": 0 if first_good else 1}


def _helpsteer(row, c):
    attr = c.rng.choice(["helpfulness", "correctness", "coherence"])
    v = val(row, attr)
    if v is None:
        return None
    rubric = {"helpfulness": "Rate how helpful the assistant response is, from 0 (useless) "
                             "to 4 (perfectly helpful).",
              "correctness": "Rate the factual correctness of the assistant response, from 0 "
                             "(mostly wrong) to 4 (fully correct).",
              "coherence": "Rate how clear and coherent the assistant response is, from 0 "
                           "(incoherent) to 4 (perfectly clear)."}[attr]
    p, r = txt(row, "prompt"), txt(row, "response")
    if not p or not r:
        return None
    return {"state": "User request:\n%s\n\nAssistant response:\n%s" % (p[:1200], r[:1600]),
            "question": rubric, "options": score_options(0, 4), "answer": int(v)}


def _civil(v):
    if v >= 0.5:
        return True
    if v <= 0.1:
        return False
    return None


Q_SPAM = ["Is this message spam?", "Is this an unsolicited promotional or scam message?"]
Q_TOXIC = ["Is this comment toxic (rude, disrespectful or hateful)?",
           "Should this comment be flagged as toxic?"]
Q_STARS = ["How many stars (1-5) did the reviewer give?",
           "Rate this review from 1 (very negative) to 5 (very positive)."]
pair = lambda a, b, la="Sentence A", lb="Sentence B": \
    (lambda r: "%s: %s\n%s: %s" % (la, txt(r, a), lb, txt(r, b)) if txt(r, a) and txt(r, b) else "")

HF: List[Src] = [
    Src("hf:boolq", "binary", lambda r, c: None if r.get("answer") is None else {
        "state": txt(r, "passage"),
        "question": c.pick(["{q}?", "Based on the passage, {q}?", "According to the text, {q}?"])
        .format(q=txt(r, "question").rstrip("?")),
        "options": YESNO, "answer": 0 if r["answer"] else 1}, hf="google/boolq"),



    Src("hf:qnli", "binary", lambda r, c: None if val(r, "label") not in (0, 1) else {
        "state": txt(r, "sentence"),
        "question": 'Does this text answer the question "%s"?' % txt(r, "question"),
        "options": YESNO, "answer": val(r, "label")}, hf="nyu-mll/glue", config="qnli"),





    Src("hf:sms_spam", "binary", binary(("sms",), "label", 1, Q_SPAM), hf="ucirvine/sms_spam"),

    Src("hf:civil_comments", "binary", binary(("text",), "toxicity", _civil, Q_TOXIC),
        hf="google/civil_comments", split="test"),
    Src("hf:toxic_conversations", "binary", binary(("text",), "label", 1, Q_TOXIC),
        hf="mteb/toxic_conversations_50k"),



    Src("hf:paws", "binary", binary((), "label", 1, ["Do these two sentences mean the same thing?",
                                                     "Is sentence B a paraphrase of sentence A?"],
                                    state=pair("sentence1", "sentence2")),
        hf="google-research-datasets/paws", config="labeled_final"),
    Src("hf:scitail", "binary", binary((), "label", "entails",
                                       ["Does the premise support the hypothesis?"],
                                       state=pair("premise", "hypothesis", "Premise", "Hypothesis")),
        hf="allenai/scitail", config="tsv_format"),
    Src("hf:prompt_injection", "binary", binary(("text",), "label", 1,
                                                ["Is this input trying to override the assistant's "
                                                 "instructions (prompt injection)?",
                                                 "Does this text attempt a prompt injection?"]),
        hf="deepset/prompt-injections"),
    Src("hf:jailbreak", "binary", binary(("prompt",), "type", lambda v: {"jailbreak": True, "benign": False}.get(str(v)),
                                         ["Is this prompt a jailbreak attempt?",
                                          "Is the user trying to get around the assistant's safety rules?"]),
        hf="jackhhao/jailbreak-classification"),


    Src("hf:snli", "choice", nli("premise", "hypothesis", "label"), hf="stanfordnlp/snli"),
    Src("hf:mnli", "choice", nli("premise", "hypothesis", "label"), hf="nyu-mll/multi_nli"),


    Src("hf:dbpedia", "choice", classes((), "label", ["What kind of entity does this text describe?"],
                                        state=lambda r: "%s\n%s" % (txt(r, "title"), txt(r, "content"))),
        hf="fancyzhx/dbpedia_14", split="test"),

    Src("hf:go_emotions", "choice", lambda r, c: None if len(r.get("labels") or []) != 1 else
        classes(("text",), "_label", ["Which emotion best describes this comment?"])(
            dict(r, _label=r["labels"][0]), _alias(c, "_label", "labels")),
        hf="google-research-datasets/go_emotions", config="simplified"),


    Src("hf:banking77", "choice", classes(("text",), "label_text",
                                          ["Which customer-service intent matches this banking message?",
                                           "Route this request: which intent is it?"], from_values=True),
        hf="mteb/banking77", collect=("label_text",)),
    Src("hf:clinc", "choice", classes(("text",), "intent", ["What does the user want?",
                                                            "Which intent does this request express?"],
                                      rename=lambda x: "out of scope" if x == "oos" else humanize(x)),
        hf="clinc/clinc_oos", config="plus"),
    Src("hf:massive_intent", "choice", classes(("text",), "label_text",
                                               ["What is the user asking the voice assistant to do?"],
                                               from_values=True),
        hf="SetFit/amazon_massive_intent_en-US", collect=("label_text",)),
    Src("hf:support_routing", "choice", classes(("instruction",), "category",
                                                ["Which support department should handle this message?",
                                                 "Route this customer message to the right team."],
                                                from_values=True, rename=lambda x: humanize(x).lower()),
        hf="bitext/Bitext-customer-support-llm-chatbot-training-dataset", collect=("category",)),
    Src("hf:support_intent", "choice", classes(("instruction",), "intent",
                                               ["What is the customer's intent?"],
                                               from_values=True, rename=lambda x: humanize(x).lower()),
        hf="bitext/Bitext-customer-support-llm-chatbot-training-dataset", collect=("intent",)),




    Src("hf:fin_sentiment", "choice", classes(("text",), "label",
                                              ["Is this financial news bearish, bullish or neutral?"],
                                              from_values=True,
                                              rename=lambda x: {"0": "bearish", "1": "bullish",
                                                                "2": "neutral"}.get(str(x), humanize(x))),
        hf="zeroshot/twitter-financial-news-sentiment", collect=("label",)),

    Src("hf:arc_easy", "choice", mcq_choices("question", "choices", "answerKey"),
        hf="allenai/ai2_arc", config="ARC-Easy"),
    Src("hf:arc_challenge", "choice", mcq_choices("question", "choices", "answerKey"),
        hf="allenai/ai2_arc", config="ARC-Challenge"),
    Src("hf:openbookqa", "choice", mcq_choices("question_stem", "choices", "answerKey"),
        hf="allenai/openbookqa", config="main"),
    Src("hf:commonsense_qa", "choice", mcq_choices("question", "choices", "answerKey"),
        hf="tau/commonsense_qa"),
    Src("hf:mmlu", "choice", _mmlu, hf="cais/mmlu", config="all", split="auxiliary_train"),

    Src("hf:preference", "choice", _ultrafeedback, hf="HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs"),





    Src("hf:helpsteer", "score", _helpsteer, hf="nvidia/HelpSteer2"),
]


def _alias(c: Ctx, new: str, old: str) -> Ctx:
    """Expose the class names of a list column under `new`."""
    if new not in c.names and old in c.names:
        c.names[new] = c.names[old]
    return c


KAGGLE: List[Src] = [
    Src("kg:food_reviews", "score", score((), "Score", 1, 5, Q_STARS,
                                          state=lambda r: "%s\n%s" % (txt(r, "Summary"), txt(r, "Text"))),
        kaggle="snap/amazon-fine-food-reviews", files=(("Reviews.csv", {}),)),
    Src("kg:ecommerce", "choice", classes(("text",), "label", ["Which product category is this listing?"],
                                          from_values=True),
        kaggle="saurabhshahane/ecommerce-text-classification", files=(("ecommerceDataset.csv", {}),),
        csv_kw={"header": None, "names": ["label", "text"]}, collect=("label",)),
    Src("kg:news_category", "choice", classes((), "category", ["Which news category fits this story?"],
                                              from_values=True, rename=lambda x: x.title(),
                                              state=lambda r: "%s\n%s" % (txt(r, "headline"),
                                                                          txt(r, "short_description"))),
        kaggle="rmisra/news-category-dataset", files=(("News_Category_Dataset*.json", {}),),
        collect=("category",)),
    Src("kg:sarcasm", "binary", binary(("headline",), "is_sarcastic", 1,
                                       ["Is this headline sarcastic?", "Is this a satirical headline?"]),
        kaggle="rmisra/news-headlines-dataset-for-sarcasm-detection",
        files=(("Sarcasm_Headlines_Dataset*.json", {}),)),
    Src("kg:job_fraud", "binary", binary((), "fraudulent", 1,
                                         ["Is this job posting fraudulent?", "Is this job ad a scam?"],
                                         state=lambda r: "Title: %s\n%s\n%s" % (
                                             txt(r, "title"), txt(r, "company_profile")[:600],
                                             txt(r, "description")[:1800])),
        kaggle="shivamb/real-or-fake-fake-jobposting-prediction", files=(("fake_job_postings.csv", {}),)),
    Src("kg:phishing", "binary", binary(("Email Text",), "Email Type",
                                        lambda v: {"Phishing Email": True, "Safe Email": False}.get(str(v)),
                                        ["Is this email a phishing attempt?", "Is this email trying to scam the reader?"]),
        kaggle="subhajournal/phishingemails", files=(("Phishing_Email.csv", {}),)),
]

_REGISTRY: Optional[Dict[str, Src]] = None


def registry() -> Dict[str, Src]:
    """All sources by name."""
    global _REGISTRY
    if _REGISTRY is None:
        from .sources_more import HF_MORE, KAGGLE_MORE, WEB
        _REGISTRY = {s.name: s for s in HF + HF_MORE + KAGGLE + KAGGLE_MORE + WEB}
    return _REGISTRY


def __getattr__(name):
    if name == "REGISTRY":
        return registry()
    raise AttributeError(name)


TOY: Dict[str, Src] = {n: Src(n, "choice", lambda r, c: None, note="offline self-test data")
                       for n in ("toy:stock", "toy:cheapest", "toy:delay", "toy:tags")}


def _toy(src: Src, cfg, want: int, seed: int) -> List[dict]:
    rng = random.Random(stable_int("%s:%d" % (src.name, seed)))
    items = ["laptops", "chairs", "printers", "cables", "monitors", "desks"]
    shops = ["Northwind", "Contoso", "Fabrikam", "Tailspin", "Litware", "Adatum"]
    out = []
    for i in range(want):
        if src.name == "toy:stock":
            a, b = rng.randint(0, 90), rng.randint(1, 90)
            it = rng.choice(items)
            r = make("%s:%d" % (src.name, i), src.name, "real", "binary",
                     "Warehouse stock: %d %s. New order: %d %s." % (a, it, b, it),
                     "Can the order be shipped from current stock?", YESNO,
                     0 if a >= b else 1, cfg)
        elif src.name == "toy:cheapest":
            k = rng.randint(2, 5)
            names = rng.sample(shops, k)
            prices = rng.sample(range(10, 99), k)
            state = "Quotes received: " + "; ".join("%s $%d" % x for x in zip(names, prices)) + "."
            r = make("%s:%d" % (src.name, i), src.name, "real", "choice", state,
                     "Which supplier has the lowest quote?", names,
                     prices.index(min(prices)), cfg)
        elif src.name == "toy:tags":
            opts = rng.sample(items, rng.randint(3, 5))
            inside = [k for k in range(len(opts)) if rng.random() < 0.4]
            extra = [x for x in items if x not in opts][:1]
            packed = [opts[k] for k in inside] + extra
            rng.shuffle(packed)
            r = make("%s:%d" % (src.name, i), src.name, "real", "multi",
                     "Packing slip for box %d: %s." % (i, ", ".join(packed) or "empty"),
                     "Which of these items are in the box?", opts, inside, cfg)
        else:
            d = rng.randint(0, 11)
            v = 1 if d < 2 else 2 if d < 4 else 3 if d < 6 else 4 if d < 9 else 5
            r = make("%s:%d" % (src.name, i), src.name, "real", "score",
                     "The parcel was delivered %d days late." % d,
                     "Rate the delay from 1 (under 2 days) to 5 (9 or more days).",
                     score_options(1, 5), v - 1, cfg)
        out.append(r)
    return out


def enabled(cfg) -> List[Src]:
    names = cfg.sources or list(registry())
    out = []
    for n in names:
        s = registry().get(n) or TOY.get(n)
        if s is None:
            LOG.warning("unknown source %r (see --list-sources)", n)
            continue
        if n in cfg.exclude_sources or (s.is_kaggle and not cfg.use_kaggle):
            continue
        out.append(s)
    return out


def _class_names(features) -> dict:
    out = {}
    for col, feat in features.items():
        if hasattr(feat, "names"):
            out[col] = list(feat.names)
        elif hasattr(getattr(feat, "feature", None), "names"):
            out[col] = list(feat.feature.names)
    return out


def _balanced(src: Src, rows, names, values, cfg, want: int, seed: int,
              config=None) -> List[dict]:
    """Map rows and take a label-balanced sample."""
    return _round_robin(_bucket(src, rows, names, values, cfg, seed, config, {}), want)


def _bucket(src: Src, rows, names, values, cfg, seed: int, config, buckets: dict) -> dict:
    for idx, row in rows:
        c = Ctx(src.name, idx, seed, names, values, config)
        try:
            m = src.map(row, c)
        except Exception:
            m = None
        if not m:
            continue
        rec = make("%s:%s" % (src.name, idx), src.name, "real", m.get("type", src.type),
                   m.get("state", ""), m["question"], m["options"], m["answer"], cfg)
        ok, _ = validate(rec, cfg.max_options)
        if not ok:
            continue
        key = m.get("key") or "|".join(rec["options"][a] for a in answers(rec)).lower() or "(none)"
        buckets.setdefault(key, []).append(rec)
    return buckets


def _round_robin(buckets: dict, want: int) -> List[dict]:
    out: List[dict] = []
    lists = [b for b in buckets.values() if b]
    k = 0
    while len(out) < want and lists:
        nxt = []
        for b in lists:
            if k < len(b):
                out.append(b[k])
                if len(out) >= want:
                    break
                nxt.append(b)
        lists, k = nxt, k + 1
    return out


def _hf_open(src: Src, config):
    """Load one Hugging Face config, with a parquet fallback."""
    from datasets import load_dataset
    args = (src.hf, config) if config else (src.hf,)
    try:
        return load_dataset(*args, split=src.split, streaming=src.stream)
    except Exception as e:
        if interrupt.stopping() or "doesn't exist" in str(e) or "gated" in str(e).lower():
            raise
        pq = "hf://datasets/%s@refs%%2Fconvert%%2Fparquet/%s/%s/*.parquet" % (
            src.hf, config or "default", src.split)
        LOG.info("  %s: %s - trying the Hub's parquet copy", src.name, str(e).splitlines()[0][:80])
        return load_dataset("parquet", data_files=pq, split="train", streaming=src.stream)


def _distinct(src: Src, names: dict, column_values) -> dict:
    out = {}
    for col in src.collect:
        vals = column_values(col)
        if vals is None:
            continue
        if col in names:
            vals = [names[col][v] for v in vals if isinstance(v, int)]
        out[col] = sorted({str(v) for v in vals if v is not None and str(v) != "nan"})
    return out


def _hf_rows(src: Src, config, want: int, seed: int):
    """(rows, class names, collected values) of a Hugging Face source."""
    ds = _hf_open(src, config)
    interrupt.check()
    names = _class_names(ds.features) if getattr(ds, "features", None) else {}
    scan = max(want * 6, 20000)
    tag = (config + "/") if config and src.configs else ""
    if src.stream:
        rows = []
        for i, row in enumerate(ds.shuffle(seed=seed, buffer_size=10000)):
            if i >= scan:
                break
            if i % 2000 == 0:
                interrupt.check()
            rows.append((tag + str(i), row))
        return rows, names, _distinct(src, names, lambda col: {r.get(col) for _, r in rows})
    values = _distinct(src, names, lambda col: ds.unique(col) if col in ds.column_names else None)
    n = len(ds)
    idx = sorted(random.Random(seed).sample(range(n), min(n, scan)))
    return [(tag + str(i), row) for i, row in zip(idx, ds.select(idx))], names, values


def load_hf(src: Src, cfg, want: int, seed: int) -> List[dict]:
    if not src.configs:
        rows, names, values = _hf_rows(src, src.config, want, seed)
        return _balanced(src, rows, names, values, cfg, want, seed, src.config)
    configs = list(src.configs)
    if configs == ["*"]:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names(src.hf)
    per = max(2000, want * 2 // max(1, len(configs)))
    buckets: dict = {}
    for k, name in enumerate(configs, 1):
        interrupt.check()
        try:
            rows, names, values = _hf_rows(src, name, per, seed)
        except interrupt.Interrupted:
            raise
        except Exception as e:
            LOG.info("  %s/%s skipped: %s", src.name, name, str(e).splitlines()[0][:100])
            continue
        _bucket(src, rows, names, values, cfg, seed, name, buckets)
        if len(configs) > 8 and k % 20 == 0:
            LOG.info("  %s: %d/%d configs read", src.name, k, len(configs))
    return _round_robin(buckets, want)


def _read_table(path: Path, kw: dict):
    import pandas as pd
    if path.suffix.lower() == ".json" or path.suffix.lower() == ".jsonl":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    if path.suffix.lower() == ".tsv" and "sep" not in kw:
        kw = dict(kw, sep="\t")
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", **kw)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("could not decode %s" % path)


def _from_files(src: Src, root: Path, cfg, want: int, seed: int) -> List[dict]:
    import pandas as pd
    frames = []
    for pattern, extra in src.files:
        for f in sorted(root.rglob(pattern)):
            df = _read_table(f, src.csv_kw)
            for k, v in extra.items():
                df[k] = v
            frames.append(df)
            if not src.all_files:
                break
    if not frames:
        raise RuntimeError("no file matching %s in %s" % ([p for p, _ in src.files], root))
    df = pd.concat(frames, ignore_index=True)
    if src.prep:
        df = src.prep(df)
    values = {col: sorted({str(v) for v in df[col].dropna().unique()})
              for col in src.collect if col in df.columns}
    n = len(df)
    scan = min(n, max(want * 6, 20000))
    idx = sorted(random.Random(seed).sample(range(n), scan))
    recs = df.iloc[idx].to_dict("records")
    return _balanced(src, zip(idx, recs), {}, values, cfg, want, seed)


def load_kaggle(src: Src, cfg, want: int, seed: int) -> List[dict]:
    try:
        import kagglehub
    except ImportError:
        raise RuntimeError("kagglehub is not installed (the env stage installs it)")
    return _from_files(src, Path(kagglehub.dataset_download(src.kaggle)), cfg, want, seed)


def _extract(path: Path, into: Path) -> None:
    import tarfile
    import zipfile
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            z.extractall(into)
    elif path.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path) as t:
            t.extractall(into)
    else:
        return
    for inner in list(into.rglob("*.zip")):
        if inner != path:
            _extract(inner, inner.parent)
            inner.unlink()


def load_url(src: Src, cfg, want: int, seed: int) -> List[dict]:
    """Download, unpack and read a file source."""
    from urllib.parse import unquote
    from .download import download
    from .paths import P
    root = P.downloads / src.name.replace(":", "__")
    ready = root / ".ready"
    if not ready.exists():
        root.mkdir(parents=True, exist_ok=True)
        dest = root / unquote(src.url.split("?")[0].rstrip("/").split("/")[-1])
        if not download(src.url, dest, src.name):
            raise RuntimeError("download failed: %s" % src.url)
        _extract(dest, root)
        if dest.suffix.lower() in (".zip", ".tgz", ".tar") and dest.exists():
            dest.unlink()
        ready.write_text("ok")
    return _from_files(src, root, cfg, want, seed)


def load(src: Src, cfg, want: int, seed: int) -> List[dict]:
    if src.name.startswith("toy:"):
        return _toy(src, cfg, want, seed)
    want = int(want * src.scale)
    if src.url:
        return load_url(src, cfg, want, seed)
    return load_kaggle(src, cfg, want, seed) if src.is_kaggle else load_hf(src, cfg, want, seed)
