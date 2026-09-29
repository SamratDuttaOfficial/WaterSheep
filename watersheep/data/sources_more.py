"""Additional public datasets."""
from __future__ import annotations
import html
import re

from .schema import YESNO, score_options
from .sources import (Q_SPAM, Q_STARS, Q_TOXIC, Src, binary, classes, humanize, nli, score,
                      subset, txt, val, mcq_choices, _helpsteer)

Q_BETTER = ["Which response is better?", "Which response answers the request better?"]
Q_INJECT = ["Is this input trying to override the assistant's instructions (prompt injection)?",
            "Does this message try to hijack the assistant?"]


def pick(opts, ans, c, k=8):
    """Up to k options that include the right one, or None."""
    opts = [str(o).strip() for o in (opts or [])]
    if len(opts) < 2 or not isinstance(ans, int) or not 0 <= ans < len(opts) or not all(opts):
        return None
    if len({o.lower() for o in opts}) != len(opts):
        return None
    right = opts[ans]
    if len(opts) > k:
        opts = subset(right, opts, k, c.rng)
    return opts, opts.index(right)


def mc(state, question, opts, ans, c, k=8):
    got = pick(opts, ans, c, k)
    if not got or not question:
        return None
    return {"state": state or "", "question": question, "options": got[0], "answer": got[1]}


def pairwise(prompt, good, bad, c, questions, cut=1100, labels=("Response 1", "Response 2"),
             head="User request"):
    good, bad = (good or "").strip(), (bad or "").strip()
    if not good or not bad or good == bad:
        return None
    first = c.rng.random() < 0.5
    r1, r2 = (good, bad) if first else (bad, good)
    clip = lambda s: s if len(s) <= cut else s[:cut].rstrip() + " ..."
    parts = (["%s:\n%s" % (head, clip(prompt.strip()))] if prompt and prompt.strip() else []) + \
        ["%s:\n%s" % (labels[0], clip(r1)), "%s:\n%s" % (labels[1], clip(r2))]
    return {"state": "\n\n".join(parts), "question": c.pick(questions),
            "options": list(labels), "answer": 0 if first else 1}


VERDICT = ["supported", "refuted", "not enough information"]


def verdict(evidence, claim, v, opts=VERDICT):
    if v not in opts or not evidence or not claim:
        return None
    return {"state": "Evidence: " + evidence,
            "question": "Does the evidence support or refute the claim, or is there not enough "
                        "information?\nClaim: " + claim,
            "options": list(opts), "answer": opts.index(v)}


def multi_labels(pairs, questions, state=None, text_col="text", k=8, none_rate=0.15):
    """Multi-label mapper."""
    def m(row, c):
        got = pairs(row, c)
        if not got:
            return None
        pool, true = [str(x) for x in got[0]], [str(x) for x in got[1]]
        if not pool or len(set(pool)) < 2 or not set(true) <= set(pool) or len(true) > k - 1:
            return None
        s = state(row) if state else txt(row, text_col)
        if not s:
            return None
        if len(pool) <= k:
            opts = list(pool)
            c.rng.shuffle(opts)
        else:
            others = [p for p in pool if p not in true]
            keep = [] if (true and c.rng.random() < none_rate) else list(true)
            opts = keep + c.rng.sample(others, min(len(others), k - len(keep)))
            c.rng.shuffle(opts)
        on = [i for i, o in enumerate(opts) if o in true]
        return {"state": s, "question": c.pick(questions), "options": opts, "answer": on,
                "type": "multi", "key": str(min(len(on), 3))}
    return m


def _bigbench(row, c):
    opts, sc = row.get("multiple_choice_targets") or [], row.get("multiple_choice_scores") or []
    if len(opts) < 2 or len(sc) != len(opts) or sum(1 for x in sc if x) != 1:
        return None
    ans = max(range(len(sc)), key=lambda i: sc[i])
    return mc(txt(row, "inputs"), c.pick(["Which answer is correct?", "Choose the best answer."]),
              opts, ans, c)


def _qasc(row, c):
    m = mcq_choices("question", "choices", "answerKey")(row, c)
    if m:
        m["state"] = "%s %s" % (txt(row, "fact1"), txt(row, "fact2"))
    return m


def _split_hh(t):
    i = t.rfind("\n\nAssistant:")
    return (t[:i], t[i + len("\n\nAssistant:"):].strip()) if i >= 0 else ("", "")


def _hh(row, c):
    p1, good = _split_hh(str(row.get("chosen") or ""))
    p2, bad = _split_hh(str(row.get("rejected") or ""))
    if not p1 or p1 != p2:
        return None
    convo = p1.replace("\n\nHuman:", "\nUser:").replace("\n\nAssistant:", "\nAssistant:").strip()
    return pairwise(convo[-1500:], good, bad, c,
                    ["Which final assistant reply is more helpful and harmless?",
                     "Which reply should the assistant send?"], head="Conversation")


def _mt_bench(row, c):
    w = val(row, "winner")
    ca, cb = row.get("conversation_a") or [], row.get("conversation_b") or []
    if val(row, "turn") != 1 or w not in ("model_a", "model_b") or len(ca) < 2 or len(cb) < 2:
        return None
    a, b = str(ca[1].get("content") or ""), str(cb[1].get("content") or "")
    good, bad = (a, b) if w == "model_a" else (b, a)
    return pairwise(str(ca[0].get("content") or ""), good, bad, c, Q_BETTER)


def _feedback(row, c):
    try:
        s = int(float(txt(row, "orig_score")))
    except ValueError:
        return None
    if not 1 <= s <= 5:
        return None
    rubric = "\n".join("%d: %s" % (k, txt(row, "orig_score%d_description" % k)) for k in range(1, 6))
    state = "Criterion: %s\n%s\n\nInstruction:\n%s\n\nResponse:\n%s" % (
        txt(row, "orig_criteria"), rubric, txt(row, "orig_instruction")[:800],
        txt(row, "orig_response")[:1400])
    return {"state": state, "question": "Using the criterion and rubric, score the response from 1 to 5.",
            "options": score_options(1, 5), "answer": s - 1}


def _pref_collection(row, c):
    p = txt(row, "orig_preference").upper().replace("RESPONSE", "").strip()
    if p not in ("A", "B"):
        return None
    a, b = txt(row, "orig_response_A"), txt(row, "orig_response_B")
    good, bad = (a, b) if p == "A" else (b, a)
    r = pairwise(txt(row, "orig_instruction")[:800], good, bad, c, Q_BETTER, cut=900)
    if r:
        r["question"] = "Which response better meets this criterion? %s" % txt(row, "orig_criteria")[:400]
    return r


_FN = re.compile(r'^\{\s*"name":\s*"([^"]+)",\s*"description":\s*"([^"]*)"', re.M)
_CALL = re.compile(r'<functioncall>\s*\{"name":\s*"([^"]+)"')
NO_TOOL = "no tool yet - reply or ask for missing details"


def _tool_choice(row, c):
    """Tool-choice records from function-calling data."""
    tools = _FN.findall(str(row.get("system") or ""))
    turns = str(row.get("chat") or "").split("\n\n\n")
    if not tools or not turns[0].startswith("USER:"):
        return None
    user = turns[0][len("USER:"):].strip()
    reply = next((t for t in turns[1:] if t.startswith("ASSISTANT:")), "")
    names = [n for n, _ in tools]
    if not user or not reply or len(set(names)) != len(names):
        return None
    m = _CALL.search(reply)
    if m and m.group(1) not in names:
        return None
    opts = names + [NO_TOOL]
    state = "Available tools:\n%s\n\nUser: %s" % ("\n".join("- %s: %s" % t for t in tools), user)
    return {"state": state, "options": opts, "answer": opts.index(m.group(1)) if m else len(names),
            "question": c.pick(["Which tool should the assistant call next?",
                                "Should the assistant call a tool for this message, and which one?"]),
            "key": "tool" if m else "none"}


def _halueval(row, c):
    good_bad = {"qa": ("right_answer", "hallucinated_answer"),
                "dialogue": ("right_response", "hallucinated_response"),
                "summarization": ("right_summary", "hallucinated_summary")}.get(c.config)
    if not good_bad:
        return None
    ok = c.rng.random() < 0.5
    resp = txt(row, good_bad[0] if ok else good_bad[1])
    if c.config == "qa":
        state = "Knowledge: %s\nQuestion: %s\nAnswer: %s" % (txt(row, "knowledge"), txt(row, "question"), resp)
        q = "Is the answer supported by the knowledge, with nothing made up?"
    elif c.config == "dialogue":
        state = "Knowledge: %s\nConversation: %s\nResponse: %s" % (
            txt(row, "knowledge"), txt(row, "dialogue_history"), resp)
        q = "Is the response faithful to the knowledge and the conversation?"
    else:
        state = "Document: %s\n\nSummary: %s" % (txt(row, "document")[:2300], resp)
        q = "Is the summary faithful to the document (no invented facts)?"
    if not resp:
        return None
    return {"state": state, "question": q, "options": YESNO, "answer": 0 if ok else 1}


def _esci(row, c):
    labels = {"e": "exact match", "s": "substitute", "c": "complement", "i": "irrelevant"}
    lab = labels.get(txt(row, "esci_label")[:1].lower())
    if not lab or txt(row, "product_locale") not in ("us", ""):
        return None
    opts = list(labels.values())
    return {"state": "Product: %s\n%s" % (txt(row, "product_title"), txt(row, "product_bullet_point")[:800]),
            "question": 'Shopping query: "%s"\nHow well does this product match the query?' % txt(row, "query"),
            "options": opts, "answer": opts.index(lab)}


def _copa(row, c):
    q = {"cause": "What was the cause?", "effect": "What happened as a result?"}.get(txt(row, "question"))
    if not q or val(row, "label") not in (0, 1):
        return None
    return {"state": txt(row, "premise"), "question": q,
            "options": [txt(row, "choice1"), txt(row, "choice2")], "answer": val(row, "label")}


def _utilitarian(row, c):
    r = pairwise("", txt(row, "baseline"), txt(row, "less_pleasant"), c,
                 ["In which situation would the person be happier?",
                  "Which situation is more pleasant for the person?"],
                 labels=("Situation 1", "Situation 2"))
    return r


def _virtue(row, c):
    s, _, trait = txt(row, "scenario").partition("[SEP]")
    if not trait.strip() or val(row, "label") not in (0, 1):
        return None
    return {"state": s.strip(), "question": 'Does the person show the trait "%s"?' % trait.strip(),
            "options": YESNO, "answer": 0 if val(row, "label") == 1 else 1}


def _moral(row, c):
    a, b = txt(row, "moral_action"), txt(row, "immoral_action")
    if not a or not b or "not specified" in (a, b):
        return None
    opts = [a, b]
    c.rng.shuffle(opts)
    return {"state": "Situation: %s\nIntention: %s" % (txt(row, "situation"), txt(row, "intention")),
            "question": c.pick(["Which action is the right thing to do?",
                                "Which action is socially and morally acceptable?"]),
            "options": opts, "answer": opts.index(a)}


LIAR = ["pants on fire", "false", "barely true", "half true", "mostly true", "true"]
MED_ABS = ["neoplasms", "digestive system diseases", "nervous system diseases",
           "cardiovascular diseases", "general pathological conditions"]
FIN_TOPICS = ["Analyst Update", "Fed | Central Banks", "Company | Product News",
              "Treasuries | Corporate Debt", "Dividend", "Earnings", "Energy | Oil", "Financials",
              "Currencies", "General News | Opinion", "Gold | Metals | Materials", "IPO",
              "Legal | Regulation", "M&A | Investments", "Macro", "Markets", "Politics",
              "Personnel Change", "Stock Commentary", "Stock Movement"]
ENTITY_SENT = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral",
               "Irrelevant": "not about it"}


_CIVIL = {"toxicity": "toxic", "severe_toxicity": "severely toxic", "obscene": "obscene",
          "threat": "threat", "insult": "insult", "identity_attack": "identity attack",
          "sexual_explicit": "sexually explicit"}
_ECHR = {"2": "Art. 2 right to life", "3": "Art. 3 prohibition of torture",
         "5": "Art. 5 right to liberty and security", "6": "Art. 6 right to a fair trial",
         "8": "Art. 8 private and family life", "9": "Art. 9 freedom of religion",
         "10": "Art. 10 freedom of expression", "11": "Art. 11 freedom of assembly",
         "14": "Art. 14 prohibition of discrimination", "P1-1": "Protocol 1 Art. 1 protection of property"}


def _go_emotions_multi(r, c):
    names = c.names.get("labels") or []
    true = [names[i] for i in (r.get("labels") or []) if 0 <= i < len(names)]
    return names, true


def _unfair_types(r, c):
    names = c.names.get("labels") or []
    return names, [names[i] for i in (r.get("labels") or []) if 0 <= i < len(names)]


def _civil_attrs(r, c):
    vals = {name: val(r, col) for col, name in _CIVIL.items()}
    if any(v is None for v in vals.values()):
        return None
    if any(0.2 < v < 0.5 for v in vals.values()):
        return None
    return list(_CIVIL.values()), [n for n, v in vals.items() if v >= 0.5]


def _aegis_categories(r, c):
    if txt(r, "prompt") in ("", "REDACTED"):
        return None
    split = lambda s: [x.strip() for x in str(s or "").split(",") if x.strip() and x.strip() != "Needs Caution"]
    vocab = sorted({x for v in c.values.get("violated_categories", []) for x in split(v)})
    return vocab, [x for x in split(val(r, "violated_categories")) if x in vocab]


def _echr(r, c):
    names = c.names.get("labels") or []
    return list(_ECHR.values()), [_ECHR[names[i]] for i in (r.get("labels") or [])
                                  if 0 <= i < len(names) and names[i] in _ECHR]


_MODERATION = {"S": "sexual content", "H": "hate", "V": "violence", "HR": "harassment", "SH": "self-harm",
               "S3": "sexual content involving minors", "H2": "hateful threats", "V2": "graphic violence"}
_JIGSAW = {"toxic": "toxic", "severe_toxic": "severely toxic", "obscene": "obscene", "threat": "threat",
           "insult": "insult", "identity_hate": "identity-based hate"}
_MESH = {"A": "anatomy", "B": "organisms", "C": "diseases", "D": "chemicals and drugs",
         "E": "diagnostic and therapeutic techniques", "F": "psychiatry and psychology",
         "G": "biological phenomena and processes", "H": "disciplines and occupations",
         "I": "social sciences and education", "J": "technology, industry and agriculture",
         "L": "information science", "M": "named groups of people", "N": "health care",
         "Z": "geographic locations"}
_ARXIV = {"cs": "computer science", "math": "mathematics", "stat": "statistics", "physics": "physics (general)",
          "astro-ph": "astrophysics", "cond-mat": "condensed matter physics", "hep-ph": "particle physics (phenomenology)",
          "hep-th": "high energy theory", "hep-ex": "high energy experiment", "hep-lat": "lattice field theory",
          "gr-qc": "general relativity and cosmology", "quant-ph": "quantum physics", "nucl-th": "nuclear theory",
          "nucl-ex": "nuclear experiment", "math-ph": "mathematical physics", "nlin": "nonlinear sciences",
          "q-bio": "quantitative biology", "q-fin": "quantitative finance", "econ": "economics",
          "eess": "electrical engineering and systems science"}
_SO_TAGS = ("javascript python java c# php android html jquery c++ css ios sql mysql r reactjs node.js arrays c "
            "asp.net json ruby-on-rails .net sql-server swift python-3.x objective-c django angular excel regex "
            "pandas ruby iphone ajax linux xml vba spring asp.net-mvc typescript database wordpress string git bash "
            "windows postgresql wpf oracle xcode vb.net eclipse multithreading list mongodb laravel scala numpy "
            "docker amazon-web-services azure kotlin flutter dataframe spring-boot firebase react-native "
            "unit-testing tensorflow api rest image forms function algorithm visual-studio csv performance loops "
            "go selenium macos matplotlib shell sorting powershell dictionary html5 datetime jsp").split()
_GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy",
           "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "Thriller", "War", "Western"]


def _flags(mapping):
    """(labels, labels set) from 0/1 flag columns."""
    def pairs(r, c):
        vals = {name: val(r, col) for col, name in mapping.items()}
        if any(v is None for v in vals.values()):
            return None
        return list(mapping.values()), [n for n, v in vals.items() if int(v) == 1]
    return pairs


def _arxiv_fields(r, c):
    codes = " ".join(r.get("categories") or []).split()
    true = sorted({_ARXIV[x.split(".")[0]] for x in codes if x.split(".")[0] in _ARXIV})
    return (list(_ARXIV.values()), true) if true else None


def _so_tags(r, c):
    if val(r, "PostTypeId") != 1:
        return None
    true = [t for t in (r.get("Tags") or []) if t in _SO_TAGS]
    return (list(_SO_TAGS), true) if true else None


def _movie_genres(r, c):
    import ast
    try:
        g = [d["name"] for d in ast.literal_eval(str(val(r, "genres") or "[]"))]
    except (ValueError, SyntaxError, TypeError, KeyError):
        return None
    true = [x for x in g if x in _GENRES]
    return (_GENRES, true) if true and len(txt(r, "overview")) > 60 else None


def _absa(r, c):
    """Aspect-sentiment records."""
    a = r.get("aspects") or {}
    terms, pol = list(a.get("term") or []), list(a.get("polarity") or [])
    seen, opts, sent = set(), [], {}
    for t, p in zip(terms, pol):
        if t.lower() not in seen and p in ("positive", "negative", "neutral"):
            seen.add(t.lower())
            opts.append(t)
            sent[t] = p
    if len(opts) < 2 or len(set(sent.values())) < 2:
        return None
    want = c.rng.choice(["negative", "positive"])
    c.rng.shuffle(opts)
    q = ("Which of these does the reviewer complain about?" if want == "negative" else
         "Which of these does the reviewer praise?")
    on = [i for i, o in enumerate(opts) if sent[o] == want]
    return {"state": txt(r, "text"), "question": q, "options": opts, "answer": on, "type": "multi",
            "key": str(min(len(on), 3))}


HF_MORE = [
    Src("hf:moderation_categories", "multi", multi_labels(_flags(_MODERATION),
                                                          ["Which moderation categories does this text fall under?"],
                                                          text_col="prompt"),
        hf="mmathys/openai-moderation-api-evaluation"),
    Src("hf:jigsaw_toxic_types", "multi", multi_labels(_flags(_JIGSAW), ["Which of these describe the comment?"],
                                                       text_col="comment_text"),
        hf="thesofakillers/jigsaw-toxic-comment-classification-challenge", stream=True),
    Src("hf:review_aspects", "multi", _absa, hf="jakartaresearch/semeval-absa",
        configs=("laptop", "restaurant")),
    Src("hf:echr_alleged", "multi", multi_labels(
        _echr, ["Which articles of the European Convention on Human Rights does the applicant say were violated?"],
        state=lambda r: "\n".join(r.get("text") or [])[:3000]),
        hf="coastalcph/lex_glue", config="ecthr_b", scale=0.6),
    Src("hf:arxiv_fields", "multi", multi_labels(_arxiv_fields, ["Which research fields does this paper belong to?"],
                                                 state=lambda r: "%s\n%s" % (txt(r, "title"), txt(r, "abstract"))),
        hf="gfissore/arxiv-abstracts-2021", stream=True),
    Src("hf:stackoverflow_tags", "multi", multi_labels(
        _so_tags, ["Which tags apply to this Stack Overflow question?"],
        state=lambda r: "%s\n%s" % (txt(r, "Title"), re.sub(r"\s+", " ", txt(r, "Body"))[:1500])),
        hf="mikex86/stackoverflow-posts", stream=True),
    Src("hf:pubmed_topics", "multi", multi_labels(_flags(_MESH),
                                                  ["Which medical subject areas does this abstract cover?"],
                                                  state=lambda r: "%s\n%s" % (txt(r, "Title"), txt(r, "abstractText"))),
        hf="owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH"),

    Src("hf:go_emotions_multi", "multi", multi_labels(_go_emotions_multi,
                                                      ["Which emotions does this comment express?"]),
        hf="google-research-datasets/go_emotions", config="simplified"),

    Src("hf:unfair_tos_types", "multi", multi_labels(_unfair_types,
                                                     ["Which kinds of unfair term does this clause contain?"]),
        hf="coastalcph/lex_glue", config="unfair_tos"),

    Src("hf:civil_attributes", "multi", multi_labels(_civil_attrs,
                                                     ["Which of these describe the comment?"]),
        hf="google/civil_comments", split="test", scale=0.6),

    Src("hf:aegis_categories", "multi", multi_labels(_aegis_categories,
                                                     ["Which safety categories does this request fall under?"],
                                                     text_col="prompt"),
        hf="nvidia/Aegis-AI-Content-Safety-Dataset-2.0", collect=("violated_categories",)),
    Src("hf:echr_violations", "multi", multi_labels(
        _echr, ["Which articles of the European Convention on Human Rights were violated in this case?"],
        state=lambda r: "\n".join(r.get("text") or [])[:3000]),
        hf="coastalcph/lex_glue", config="ecthr_a", scale=0.6),

    Src("hf:squad_v2", "binary", lambda r, c: {
        "state": txt(r, "context"),
        "question": c.pick(['Can the question "{q}" be answered from this passage?',
                            'Does the passage contain the answer to "{q}"?']).format(q=txt(r, "question")),
        "options": YESNO, "answer": 0 if (r.get("answers") or {}).get("text") else 1},
        hf="rajpurkar/squad_v2"),


    Src("hf:esci", "choice", _esci, hf="tasksource/esci", stream=True),

    Src("hf:halueval", "binary", _halueval, hf="pminervini/HaluEval",
        configs=("qa", "dialogue", "summarization"), split="data"),

    Src("hf:cosqa", "binary", lambda r, c: None if val(r, "label") not in (0, 1) else {
        "state": txt(r, "code")[:2200],
        "question": 'Does this code do what the search query "%s" asks for?' % txt(r, "doc"),
        "options": YESNO, "answer": 0 if val(r, "label") == 1 else 1}, hf="gonglinyuan/CoSQA"),


    Src("hf:vitaminc", "choice", lambda r, c: verdict(
        txt(r, "evidence"), txt(r, "claim"),
        {"SUPPORTS": "supported", "REFUTES": "refuted",
         "NOT ENOUGH INFO": "not enough information"}.get(txt(r, "label"))),
        hf="tals/vitaminc", stream=True),


    Src("hf:liar2", "choice", lambda r, c: None if val(r, "label") not in range(6) else {
        "state": "Statement: %s\nSpeaker: %s\nContext: %s" % (txt(r, "statement"), txt(r, "speaker"),
                                                               txt(r, "context")),
        "question": "How would a fact-checker rate this statement?", "options": LIAR,
        "answer": val(r, "label")}, hf="chengxuphd/liar2"),
    Src("hf:pubmedqa", "binary", lambda r, c: None if txt(r, "final_decision") not in ("yes", "no") else {
        "state": " ".join((r.get("context") or {}).get("contexts") or []),
        "question": txt(r, "question"), "options": YESNO,
        "answer": 0 if txt(r, "final_decision") == "yes" else 1},
        hf="qiaojin/PubMedQA", config="pqa_artificial", stream=True),


    Src("hf:aegis_safety", "binary", lambda r, c: None if txt(r, "prompt") in ("", "REDACTED") else
        binary(("prompt",), "prompt_label", lambda v: {"unsafe": True, "safe": False}.get(str(v)),
               ["Is this user prompt unsafe?", "Does this request need a safety refusal?"])(r, c),
        hf="nvidia/Aegis-AI-Content-Safety-Dataset-2.0"),
    Src("hf:prosocial", "choice", lambda r, c: None if val(r, "response_id") != 0 else
        classes(("context",), "safety_label",
                ["How carefully should an assistant respond to this message?"], from_values=True,
                rename=lambda x: x.strip("_").replace("_", " "))(r, c),
        hf="allenai/prosocial-dialog", collect=("safety_label",), version=2),
    Src("hf:spml_injection", "binary", binary((), "Prompt injection", 1, Q_INJECT,
                                              state=lambda r: "System prompt:\n%s\n\nUser message:\n%s" % (
                                                  txt(r, "System Prompt")[:1500], txt(r, "User Prompt"))),
        hf="reshabhs/SPML_Chatbot_Prompt_Injection"),

    Src("hf:hate_offensive", "choice", classes(("tweet",), "class",
                                               ["Is this tweet hate speech, offensive language, or neither?"]),
        hf="tdavidson/hate_speech_offensive"),


    Src("hf:clickbait", "binary", binary(("title",), "clickbait", 1,
                                         ["Is this headline clickbait?", "Is this headline written as bait?"]),
        hf="marksverdhei/clickbait_title_classification"),
    Src("hf:spam_messages", "binary", binary(("text",), "label", "spam", Q_SPAM),
        hf="Deysi/spam-detection-dataset"),


    Src("hf:tool_choice", "choice", _tool_choice, hf="glaiveai/glaive-function-calling-v2", stream=True,
        version=2),


    Src("hf:snips", "choice", classes(("text",), "category", ["What does the user want the assistant to do?"],
                                      from_values=True, rename=lambda x: humanize(x).lower()),
        hf="benayas/snips", collect=("category",)),
    Src("hf:massive_scenario", "choice", classes(("text",), "label_text",
                                                 ["Which assistant domain does this request belong to?"],
                                                 from_values=True),
        hf="SetFit/amazon_massive_scenario_en-US", collect=("label_text",)),
    Src("hf:student_questions", "choice", classes(("text",), "label_text",
                                                  ["Which school subject is this question from?"],
                                                  from_values=True),
        hf="SetFit/student-question-categories", collect=("label_text",)),

    Src("hf:hh_rlhf", "choice", _hh, hf="Anthropic/hh-rlhf", stream=True),

    Src("hf:reward_bench", "choice", lambda r, c: pairwise(txt(r, "prompt"), txt(r, "chosen"),
                                                            txt(r, "rejected"), c, Q_BETTER),
        hf="allenai/reward-bench", split="filtered"),
    Src("hf:mt_bench_human", "choice", _mt_bench, hf="lmsys/mt_bench_human_judgments", split="human"),
    Src("hf:orca_pairs", "choice", lambda r, c: pairwise(txt(r, "question"), txt(r, "chosen"),
                                                          txt(r, "rejected"), c, Q_BETTER),
        hf="Intel/orca_dpo_pairs"),
    Src("hf:preference_rubric", "choice", _pref_collection, hf="prometheus-eval/Preference-Collection",
        stream=True),
    Src("hf:rubric_score", "score", _feedback, hf="prometheus-eval/Feedback-Collection", stream=True),
    Src("hf:helpsteer1", "score", _helpsteer, hf="nvidia/HelpSteer"),

    Src("hf:bigbench", "choice", _bigbench, hf="tasksource/bigbench", configs="*", scale=3,
        note="about 100 BIG-bench multiple-choice tasks"),

    Src("hf:qasc", "choice", _qasc, hf="allenai/qasc"),
    Src("hf:winogrande", "choice", lambda r, c: None if txt(r, "answer") not in ("1", "2") else {
        "state": txt(r, "sentence"), "question": "Which option correctly fills the blank (_)?",
        "options": [txt(r, "option1"), txt(r, "option2")], "answer": int(txt(r, "answer")) - 1},
        hf="allenai/winogrande", config="winogrande_xl"),
    Src("hf:quartz", "choice", mcq_choices("question", "choices", "answerKey", "para"), hf="allenai/quartz"),
    Src("hf:copa", "choice", _copa, hf="aps/super_glue", config="copa"),


    Src("hf:wsc", "binary", lambda r, c: None if val(r, "label") not in (0, 1) else {
        "state": txt(r, "text"),
        "question": 'In this text, does "%s" refer to "%s"?' % (txt(r, "span2_text"), txt(r, "span1_text")),
        "options": YESNO, "answer": 0 if val(r, "label") == 1 else 1},
        hf="aps/super_glue", config="wsc.fixed"),

    Src("hf:folio", "choice", lambda r, c: None if txt(r, "label") not in ("True", "False", "Uncertain") else {
        "state": txt(r, "premises"),
        "question": "Based only on these premises, is the conclusion true, false or uncertain?\n"
                    "Conclusion: " + txt(r, "conclusion"),
        "options": ["true", "false", "uncertain"],
        "answer": ["True", "False", "Uncertain"].index(txt(r, "label"))}, hf="tasksource/folio"),

    Src("hf:truthfulqa", "choice", lambda r, c: mc(
        "", txt(r, "question"), (r.get("mc1_targets") or {}).get("choices"),
        list((r.get("mc1_targets") or {}).get("labels") or [0]).index(1)
        if 1 in ((r.get("mc1_targets") or {}).get("labels") or []) else -1, c),
        hf="truthfulqa/truthful_qa", config="multiple_choice", split="validation"),
    Src("hf:mmlu_pro", "choice", lambda r, c: mc("", txt(r, "question"), r.get("options"),
                                                 val(r, "answer_index"), c),
        hf="TIGER-Lab/MMLU-Pro", split="test"),
    Src("hf:medmcqa", "choice", lambda r, c: None if txt(r, "choice_type") != "single" else mc(
        "", txt(r, "question"), [txt(r, k) for k in ("opa", "opb", "opc", "opd")], val(r, "cop"), c),
        hf="openlifescienceai/medmcqa"),
    Src("hf:medqa", "choice", lambda r, c: mc(
        "", txt(r, "question"), [(r.get("options") or {}).get(k) for k in "ABCD"],
        "ABCD".find(txt(r, "answer_idx") or "?"), c), hf="GBaker/MedQA-USMLE-4-options"),

    Src("hf:case_hold", "choice", lambda r, c: mc(
        txt(r, "context"), "Which holding belongs where the <HOLDING> placeholder is?",
        r.get("endings"), val(r, "label"), c), hf="coastalcph/lex_glue", config="case_hold"),
    Src("hf:ledgar", "choice", classes(("text",), "label", ["What type of contract clause is this?",
                                                             "Which provision heading fits this clause?"]),
        hf="coastalcph/lex_glue", config="ledgar"),
    Src("hf:unfair_tos", "binary", binary(("text",), "labels", lambda v: len(v) > 0,
                                          ["Is this terms-of-service clause potentially unfair to consumers?"]),
        hf="coastalcph/lex_glue", config="unfair_tos"),

    Src("hf:medical_abstracts", "choice", lambda r, c: None if val(r, "condition_label") not in range(1, 6)
        else {"state": txt(r, "medical_abstract"), "options": MED_ABS,
              "question": "Which kind of condition does this medical abstract describe?",
              "answer": val(r, "condition_label") - 1}, hf="TimSchopf/medical_abstracts"),
    Src("hf:symptom_diagnosis", "choice", classes(("input_text",), "output_text",
                                                  ["What is the most likely diagnosis?"], from_values=True),
        hf="gretelai/symptom_to_diagnosis", collect=("output_text",)),

    Src("hf:fin_topics", "choice", lambda r, c: mc(txt(r, "text"), "What is this financial tweet about?",
                                                   FIN_TOPICS, val(r, "label"), c),
        hf="zeroshot/twitter-financial-news-topic"),


    Src("hf:ethics_commonsense", "binary", binary(("input",), "label", 1,
                                                  ["Did the narrator do something clearly morally wrong?",
                                                   "Is the action described here morally wrong?"]),
        hf="hendrycks/ethics", config="commonsense"),
    Src("hf:ethics_deontology", "binary", binary((), "label", 1, ["Is this a reasonable excuse?"],
                                                 state=lambda r: "Request: %s\nExcuse: %s" % (
                                                     txt(r, "scenario"), txt(r, "excuse"))),
        hf="hendrycks/ethics", config="deontology"),
    Src("hf:ethics_justice", "binary", binary(("scenario",), "label", 1,
                                              ["Is this claim reasonable and fair?"]),
        hf="hendrycks/ethics", config="justice"),
    Src("hf:ethics_virtue", "binary", _virtue, hf="hendrycks/ethics", config="virtue"),
    Src("hf:ethics_utility", "choice", _utilitarian, hf="hendrycks/ethics", config="utilitarianism"),
    Src("hf:moral_stories", "choice", _moral, hf="demelin/moral_stories", config="full"),


    Src("hf:counterfactual", "binary", binary(("text",), "label_text", "counterfactual",
                                              ["Does this review describe something counterfactual "
                                               "(a wish or an imagined outcome, not what happened)?"]),
        hf="mteb/amazon_counterfactual", config="en"),
    Src("hf:poem_sentiment", "choice", classes(("verse_text",), "label", ["What feeling does this verse convey?"],
                                               rename={"no_impact": "no clear feeling"}),
        hf="google-research-datasets/poem_sentiment"),


    Src("hf:cf_media_bias", "binary", binary(("text",), "label", 0,
                                             ["Is this politician's message partisan?"]),
        hf="tasksource/crowdflower", config="political-media-bias"),
    Src("hf:cf_media_message", "choice", classes(("text",), "label",
                                                 ["What is the main purpose of this politician's message?"],
                                                 skip=("other",)),
        hf="tasksource/crowdflower", config="political-media-message"),
    Src("hf:cf_media_audience", "choice", classes(("text",), "label",
                                                  ["Is this message aimed at the politician's own "
                                                   "constituency or a national audience?"]),
        hf="tasksource/crowdflower", config="political-media-audience"),
]


KAGGLE_MORE = [
    Src("kg:movie_genres", "multi", multi_labels(_movie_genres, ["Which genres fit this movie?"],
                                                 state=lambda r: "%s\n%s" % (txt(r, "title"), txt(r, "overview"))),
        kaggle="rounakbanik/the-movies-dataset", files=(("movies_metadata.csv", {}),),
        csv_kw={"low_memory": False}),
    Src("kg:resume_category", "choice", classes(("Resume_str",), "Category",
                                                ["Which job category does this resume fit best?"],
                                                from_values=True, rename=lambda x: humanize(x).title()),
        kaggle="snehaanbhawal/resume-dataset", files=(("Resume.csv", {}),), collect=("Category",)),
    Src("kg:medical_specialty", "choice", classes((), "medical_specialty",
                                                  ["Which medical specialty does this note belong to?"],
                                                  from_values=True, rename=lambda x: x.strip(),
                                                  state=lambda r: "%s\n%s" % (
                                                      txt(r, "description"), txt(r, "transcription")[:2200])),
        kaggle="tboyle10/medicaltranscriptions", files=(("mtsamples.csv", {}),),
        collect=("medical_specialty",)),
    Src("kg:clothing_recommend", "binary", binary((), "Recommended IND", 1,
                                                  ["Would this customer recommend the product?",
                                                   "Does the reviewer recommend this item?"],
                                                  state=lambda r: ("%s\n%s" % (txt(r, "Title"), txt(r, "Review Text"))
                                                                   ).strip() if txt(r, "Review Text") else ""),
        kaggle="nicapotato/womens-ecommerce-clothing-reviews",
        files=(("Womens Clothing E-Commerce Reviews.csv", {}),)),
    Src("kg:clothing_rating", "score", score((), "Rating", 1, 5, Q_STARS,
                                             state=lambda r: ("%s\n%s" % (txt(r, "Title"), txt(r, "Review Text"))
                                                              ).strip() if txt(r, "Review Text") else ""),
        kaggle="nicapotato/womens-ecommerce-clothing-reviews",
        files=(("Womens Clothing E-Commerce Reviews.csv", {}),)),


    Src("kg:so_quality", "choice", classes((), "Y", ["How would moderators judge this Stack Overflow question?"],
                                           from_values=True,
                                           rename={"HQ": "high quality", "LQ_EDIT": "low quality, needs editing",
                                                   "LQ_CLOSE": "low quality, should be closed"},
                                           state=lambda r: "Title: %s\n%s" % (
                                               txt(r, "Title"), re.sub(r"<[^>]+>", " ", txt(r, "Body")))),
        kaggle="imoore/60k-stack-overflow-questions-with-quality-rate", files=(("train.csv", {}),),
        collect=("Y",)),
    Src("kg:cyberbullying", "choice", classes(("tweet_text",), "cyberbullying_type",
                                              ["What kind of cyberbullying, if any, is in this tweet?"],
                                              from_values=True,
                                              rename=lambda x: {"not_cyberbullying": "none",
                                                                "other_cyberbullying": "other"}.get(x, humanize(x))),
        kaggle="andrewmvd/cyberbullying-classification", files=(("cyberbullying_tweets.csv", {}),),
        collect=("cyberbullying_type",)),
    Src("kg:phone_reviews", "score", score(("Reviews",), "Rating", 1, 5, Q_STARS, conv=lambda v: int(float(v))),
        kaggle="PromptCloudHQ/amazon-reviews-unlocked-mobile-phones", files=(("Amazon_Unlocked_Mobile.csv", {}),)),
    Src("kg:clothing_fit", "choice", classes((), "fit", ["Did the item fit as expected, run small, or run large?"],
                                             from_values=True,
                                             rename={"fit": "fit as expected", "small": "ran small",
                                                     "large": "ran large"},
                                             state=lambda r: "%s\n%s" % (txt(r, "review_summary"),
                                                                         txt(r, "review_text"))),
        kaggle="rmisra/clothing-fit-dataset-for-size-recommendation",
        files=(("renttherunway_final_data.json", {}),), collect=("fit",)),


    Src("kg:entity_sentiment", "choice", lambda r, c: None if txt(r, "sentiment") not in ENTITY_SENT else {
        "state": txt(r, "text"), "question": "What is this tweet's sentiment toward %s?" % txt(r, "entity"),
        "options": list(ENTITY_SENT.values()),
        "answer": list(ENTITY_SENT).index(txt(r, "sentiment"))},
        kaggle="jp797498e/twitter-entity-sentiment-analysis", files=(("twitter_training.csv", {}),),
        csv_kw={"header": None, "names": ["id", "entity", "sentiment", "text"]}),
]


def _hwu_prep(df):
    df["_intent"] = (df["scenario"].astype(str) + " " + df["intent"].astype(str)).str.replace("_", " ")
    return df


WEB = [
    Src("web:news_aggregator", "choice", classes(("TITLE",), "CATEGORY",
                                                 ["Which news section does this headline belong to?"],
                                                 from_values=True,
                                                 rename={"b": "business", "t": "science and technology",
                                                         "e": "entertainment", "m": "health"}),
        url="https://archive.ics.uci.edu/static/public/359/news+aggregator.zip",
        files=(("newsCorpora.csv", {}),), collect=("CATEGORY",),
        csv_kw={"sep": "\t", "header": None, "quoting": 3,
                "names": ["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"]},
        note="UCI Machine Learning Repository (Gasparetti, 2016)"),
    Src("web:youtube_spam", "binary", binary(("CONTENT",), "CLASS", 1,
                                             ["Is this YouTube comment spam?", "Is this comment self-promotion or spam?"]),
        url="https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip",
        files=(("Youtube*.csv", {}),), all_files=True,
        note="UCI Machine Learning Repository (Alberto et al., 2015)"),
    Src("web:hwu64", "choice", classes(("answer",), "_intent", ["What does the user want the assistant to do?"],
                                       from_values=True),
        url="https://raw.githubusercontent.com/xliuhw/NLU-Evaluation-Data/master/AnnotatedData/"
            "NLU-Data-Home-Domain-Annotated-All.csv",
        files=(("NLU-Data-Home-Domain-Annotated-All.csv", {}),), csv_kw={"sep": ";"}, prep=_hwu_prep,
        collect=("_intent",), note="HWU64 (Liu et al., 2019) from GitHub"),
]
