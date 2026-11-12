"""WaterSheep for LangChain: a Runnable and an agent tool.

    from watersheep.integrations.langchain import WaterSheepRunnable, watersheep_tool

    route = WaterSheepRunnable("samratduttaofficial/WaterSheep",
                               question="Which team should handle this?", options=["billing", "shipping"])
    route.invoke("I was charged twice.")       # {"type": "choice", "choice": "billing", "probabilities": {...}}

    ws = WaterSheepRunnable(url="http://127.0.0.1:8766")    # a running `watersheep --serve`
    ws.invoke({"text": "...", "question": "Is this spam?"})

    tool = watersheep_tool(url="http://127.0.0.1:8766")     # for agents
"""
from __future__ import annotations
import json
from typing import Any, List, Optional

from . import client, decide, missing

try:
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field
except ImportError as e:
    raise missing("langchain-core", "langchain") from e


class WaterSheepRunnable(Runnable):
    """Text plus a question in, the answer with a probability for every option out.

    Input is the text (a string, uses the question given here) or a dict with `text` (or `state`),
    `question`, and optionally `options`, `type` and `threshold`; dict keys override the defaults.
    No options means yes/no. A dict of options maps labels to descriptions.
    """

    def __init__(self, model=None, *, url: Optional[str] = None, question: Optional[str] = None,
                 options=None, type: Optional[str] = None, threshold: Optional[float] = None,
                 device: Optional[str] = None):
        self.client = client(model, url, device)
        self.defaults = {"question": question, "options": options, "type": type, "threshold": threshold}

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> dict:
        return self._call_with_config(self._decide, input, config)

    def _decide(self, input: Any) -> dict:
        a = dict(self.defaults)
        if isinstance(input, dict):
            a.update({k: v for k, v in input.items() if k in a})
            text = input.get("text", input.get("state"))
        else:
            text = input
        if text is None:
            raise ValueError("no text given (a string, or a dict with 'text')")
        return decide(self.client, text, a["question"], a["options"], a["type"], a["threshold"])


class WaterSheepInput(BaseModel):
    text: str = Field(description="The text to decide about.")
    question: str = Field(description="The question about the text, e.g. 'Is this a refund request?'.")
    options: Optional[List[str]] = Field(
        None, description="The possible answers. Leave empty for a yes/no question.")


def watersheep_tool(model=None, *, url: Optional[str] = None, device: Optional[str] = None,
                    name: str = "watersheep",
                    description: str = ("Answers a yes/no or multiple-choice question about a text and "
                                        "returns a calibrated probability for every option. Use it for "
                                        "classification, routing and yes/no checks.")) -> StructuredTool:
    """A LangChain tool (a BaseTool) that agents can call for yes/no and classification decisions."""
    c = client(model, url, device)

    def run(text: str, question: str, options: Optional[List[str]] = None) -> str:
        return json.dumps(decide(c, text, question, options or None), ensure_ascii=False)

    return StructuredTool.from_function(run, name=name, description=description,
                                        args_schema=WaterSheepInput)
