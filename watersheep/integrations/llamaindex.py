"""WaterSheep for LlamaIndex: a relevance filter for retrieved nodes, and a FunctionTool.

    from watersheep.integrations.llamaindex import WaterSheepRelevanceFilter, watersheep_function_tool

    keep = WaterSheepRelevanceFilter("samratduttaofficial/WaterSheep", threshold=0.5)
    engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[keep])

    keep = WaterSheepRelevanceFilter(url="http://127.0.0.1:8766")    # a running `watersheep --serve`
    tool = watersheep_function_tool(url="http://127.0.0.1:8766")     # for agents
"""
from __future__ import annotations
import json
from typing import Any, List, Optional

from . import client, decide, missing

try:
    from llama_index.core.bridge.pydantic import Field, PrivateAttr
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
    from llama_index.core.tools import FunctionTool
except ImportError as e:
    raise missing("llama-index-core", "llamaindex") from e


class WaterSheepRelevanceFilter(BaseNodePostprocessor):
    """Keeps the retrieved nodes that WaterSheep judges relevant to the query.

    Each node gets a yes/no question (`question`, with `{query}` filled in). Nodes whose probability of
    yes is at least `threshold` are kept, best first, with that probability as their score.
    """

    question: str = Field(default='Does this text help answer the question "{query}"?')
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    _client: Any = PrivateAttr()

    def __init__(self, model=None, *, url: Optional[str] = None, device: Optional[str] = None,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self._client = client(model, url, device)

    @classmethod
    def class_name(cls) -> str:
        return "WaterSheepRelevanceFilter"

    def _postprocess_nodes(self, nodes: List[NodeWithScore],
                           query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("WaterSheepRelevanceFilter needs the query")
        q = self.question.format(query=query_bundle.query_str)
        out = []
        for n in nodes:
            p = decide(self._client, n.node.get_content(metadata_mode=MetadataMode.NONE), q)["noul"]
            if p >= self.threshold:
                out.append(NodeWithScore(node=n.node, score=p))
        return sorted(out, key=lambda n: -n.score)


def watersheep_function_tool(model=None, *, url: Optional[str] = None, device: Optional[str] = None,
                             name: str = "watersheep",
                             description: str = ("Answers a yes/no or multiple-choice question about a "
                                                 "text and returns a calibrated probability for every "
                                                 "option. Leave options empty for yes/no.")) -> FunctionTool:
    """A LlamaIndex FunctionTool that agents can call for yes/no and classification decisions."""
    c = client(model, url, device)

    def watersheep(text: str, question: str, options: Optional[List[str]] = None) -> str:
        """Answer a question about a text, with a probability for every option."""
        return json.dumps(decide(c, text, question, options or None), ensure_ascii=False)

    return FunctionTool.from_defaults(fn=watersheep, name=name, description=description)
