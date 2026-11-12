"""Shared plumbing for the LangChain and LlamaIndex integrations.

    c = client("samratduttaofficial/WaterSheep")    # a local model (anything WaterSheep.load takes)
    c = client(url="http://127.0.0.1:8766")         # or a running `watersheep --serve`
    decide(c, text, "Which team?", ["billing", "shipping"])
"""
from __future__ import annotations
import json
import urllib.error
import urllib.request
from typing import Optional, Sequence, Union

YESNO = ["yes", "no"]
TYPES = {"binary": "noul", "noul": "noul", "yes/no": "noul", "boolean": "noul", "choice": "choice",
         "score": "score", "multi": "multi", "multi_choice": "multi", "multilabel": "multi",
         "multi-label": "multi"}


def missing(lib: str, extra: str) -> ImportError:
    return ImportError('%s is not installed - pip install "watersheep[%s]"' % (lib, extra))


class Remote:
    """A running `watersheep --serve` (or any server that answers POST /v1/systemone)."""

    def __init__(self, url: str, timeout: float = 60.0, api_key: Optional[str] = None):
        url = url.rstrip("/")
        if not url.endswith(("/v1/systemone", "/v1/decisions")):
            url += "/v1/systemone"
        self.url, self.timeout, self.api_key, self.name = url, timeout, api_key, url

    def ask(self, request: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer %s" % self.api_key
        req = urllib.request.Request(self.url, json.dumps(request).encode("utf-8"), headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                why = json.loads(e.read()).get("error")
            except Exception:
                why = None
            raise RuntimeError("%s: %s" % (self.url, why or e)) from e


def client(model=None, url: Optional[str] = None, device: Optional[str] = None):
    """A local model or a server. `model` may also be a loaded WaterSheep (anything with `ask`)."""
    if url:
        if model is not None:
            raise ValueError("give a model or a url, not both")
        return Remote(url)
    if hasattr(model, "ask"):
        return model
    from ..infer import WaterSheep
    return WaterSheep.load(model, device)


def infer_type(options) -> str:
    """As watersheep.infer.infer_type, without importing torch (a server client may not need it)."""
    low = [str(o).strip().lower() for o in options or []]
    if not low or low == YESNO:
        return "noul"
    if all(len(o) == 1 and o.isdigit() for o in low) and \
            [int(o) for o in low] == list(range(int(low[0]), int(low[0]) + len(low))):
        return "score"
    return "choice"


def question(question: str, options: Union[Sequence[str], dict, None] = None,
             type: Optional[str] = None, threshold: Optional[float] = None) -> dict:
    """One question in the `ask` format. No options means yes/no; a dict maps labels to descriptions."""
    if not str(question or "").strip():
        raise ValueError("a question is required")
    if type:
        t = TYPES.get(str(type).lower())
        if t is None:
            raise ValueError("unknown question type %r (noul, choice, score or multi)" % type)
    else:
        t = infer_type(options)
    q = {"type": t, "instructions": str(question)}
    if t != "noul":
        q["criteria"] = dict(options) if isinstance(options, dict) else [str(o) for o in options or []]
    if threshold is not None and t == "multi":
        q["threshold"] = float(threshold)
    return q


def decide(c, text, question_: str, options=None, type: Optional[str] = None,
           threshold: Optional[float] = None) -> dict:
    """The answer to one question about `text`, with a probability for every option."""
    res = c.ask({"state": text, "questions": {"q": question(question_, options, type, threshold)}})
    a = res["answers"]["q"]
    if "error" in a:
        raise ValueError(a["error"])
    if a.get("type") == "noul":
        a.setdefault("probabilities", {"yes": a["noul"], "no": round(1 - a["noul"], 4)})
    return a
