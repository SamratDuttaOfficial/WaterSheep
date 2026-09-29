"""Appendix tables of public datasets and synthetic families."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from watersheep.data.taxonomy import FAMILIES                          # noqa: E402

OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TYPE = {"binary": "B", "choice": "C", "score": "S", "multi": "M"}
HELDOUT = {"kg:sarcasm", "kg:news_category", "hf:qasc", "kg:phone_reviews"}


def esc(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#", "$": r"\$",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in s)


def ref(r: dict) -> str:
    x = r["ref"]
    if x.startswith("http"):
        x = x.split("//", 1)[1].split("/")[0]
    return r"\path{%s}" % x


def sources():
    st = json.load(open(ROOT / "results/data/data_stats.json"))
    rows = st["sources"]
    lines = [r"\begin{longtable}{@{}l>{\raggedright\arraybackslash}p{6.4cm}crl@{}}",
             (r"\caption{All %d public datasets. Name is the identifier in our code; the origin is the Hugging Face "
              r"dataset, the Kaggle dataset, or the host of a direct download. Type: B binary, C choice, S score, "
              r"M multi. $^\dagger$ withheld from training.}\label{tab:sources}\\") % len(rows),
             r"\toprule", r"Name & Origin & Type & Records & Category \\", r"\midrule", r"\endfirsthead",
             r"\toprule", r"Name & Origin & Type & Records & Category \\", r"\midrule", r"\endhead",
             r"\bottomrule", r"\endfoot"]
    short = {"Safety, abuse and fraud": "Safety", "Grounding, retrieval and fact checking": "Grounding",
             "Agents, tools and customer support": "Agents and support", "Judging and preference": "Judging",
             "Reasoning, inference and knowledge": "Reasoning", "Law, medicine and finance": "Domain",
             "Ethics": "Ethics", "Sentiment, emotion and ratings": "Sentiment",
             "Topic and document classification": "Topic"}
    for r in sorted(rows, key=lambda r: (r["category"], r["name"])):
        name = esc(r["name"].split(":", 1)[1]) + (r"$^\dagger$" if r["name"] in HELDOUT else "")
        lines.append("%s & %s & %s & %s & %s \\\\" % (name, ref(r), TYPE[r["type"]],
                                                  format(r["records"], ","), short[r["category"]]))
    lines.append(r"\end{longtable}")
    (OUT / "sources.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def families():
    st = json.load(open(ROOT / "results/data/data_stats.json"))
    fams = st["synth"]["families"]
    desc = {name: d for _, name, d in FAMILIES}
    lines = [r"\begin{longtable}{@{}lc>{\raggedright\arraybackslash}p{8.4cm}rr@{}}",
             r"\caption{The 57 synthetic decision families with the description given to the teacher, the number "
             r"of generated examples and the acceptance rate. Type: B binary, C choice, S score, M multi.}"
             r"\label{tab:families}\\",
             r"\toprule", r"Family & Type & Context and decision & Generated & Acceptance \\", r"\midrule",
             r"\endfirsthead",
             r"\toprule", r"Family & Type & Context and decision & Generated & Acceptance \\", r"\midrule",
             r"\endhead", r"\bottomrule", r"\endfoot"]
    for t, name, _ in FAMILIES:
        f = fams[name]
        lines.append("%s & %s & %s & %s & %.0f\\%% \\\\" % (
            esc(name.replace("_", " ")), TYPE[t], esc(desc[name]), format(f["tried"], ","),
            100 * f["accepted"] / f["tried"]))
    lines.append(r"\end{longtable}")
    (OUT / "families.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sources()
    families()
    print("written", OUT)
