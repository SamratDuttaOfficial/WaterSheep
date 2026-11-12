"""LangChain and LlamaIndex integrations against a fake model (no weights needed).

    python -m unittest tests.test_integrations
"""
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch

from watersheep import cli
from watersheep.integrations import Remote, client, decide, question

try:
    import langchain_core  # noqa: F401
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
try:
    import llama_index.core  # noqa: F401
    HAS_LLAMAINDEX = True
except ImportError:
    HAS_LLAMAINDEX = False


class FakeWaterSheep:
    """Answers like WaterSheep.ask: yes when the text contains "yes", else the first option."""
    name = "fake-watersheep"

    def __init__(self):
        self.requests = []

    def ask(self, request):
        self.requests.append(request)
        text, answers = str(request.get("state")), {}
        for name, q in request["questions"].items():
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.9 if "yes" in text else 0.2}
                continue
            crit = q["criteria"]
            keys = list(crit) if isinstance(crit, dict) else crit
            if len(keys) < 2:
                answers[name] = {"type": q["type"], "error": "needs 2 labels"}
                continue
            probs = {k: (0.7 if i == 0 else round(0.3 / (len(keys) - 1), 4)) for i, k in enumerate(keys)}
            answers[name] = {"type": "choice", "choice": keys[0], "confidence": 0.5,
                             "probabilities": probs}
        return {"model": self.name, "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class CoreTests(unittest.TestCase):
    def test_question_types(self):
        self.assertEqual(question("Spam?"), {"type": "noul", "instructions": "Spam?"})
        self.assertEqual(question("Spam?", ["Yes", "No"])["type"], "noul")
        self.assertEqual(question("Rate it", ["1", "2", "3"])["type"], "score")
        self.assertEqual(question("Team?", ["billing", "shipping"])["criteria"], ["billing", "shipping"])
        q = question("Issues?", ["a", "b"], type="multi", threshold=0.3)
        self.assertEqual((q["type"], q["threshold"]), ("multi", 0.3))
        with self.assertRaises(ValueError):
            question("")
        with self.assertRaises(ValueError):
            question("x", type="essay")

    def test_decide(self):
        ws = FakeWaterSheep()
        self.assertIs(client(ws), ws)
        a = decide(ws, "yes please", "Agree?")
        self.assertEqual(a["probabilities"], {"yes": 0.9, "no": 0.1})
        self.assertEqual(decide(ws, "hi", "Team?", ["billing", "shipping"])["choice"], "billing")
        with self.assertRaises(ValueError):
            decide(ws, "hi", "Team?", ["billing"], type="choice")

    def test_remote_against_server(self):
        ws, port = FakeWaterSheep(), free_port()
        with patch("builtins.print"):
            threading.Thread(target=cli.serve, args=(ws, port), daemon=True).start()
            time.sleep(0.3)
        c = client(url="http://127.0.0.1:%d/" % port)
        self.assertIsInstance(c, Remote)
        self.assertTrue(c.url.endswith("/v1/systemone"))
        self.assertEqual(decide(c, "yes", "Agree?")["noul"], 0.9)
        self.assertEqual(ws.requests[-1]["state"], "yes")
        with self.assertRaises(ValueError):
            client(ws, url="http://127.0.0.1:1")


@unittest.skipUnless(HAS_LANGCHAIN, "langchain-core is not installed")
class LangChainTests(unittest.TestCase):
    def test_runnable(self):
        from langchain_core.runnables import RunnableLambda
        from watersheep.integrations.langchain import WaterSheepRunnable
        ws = FakeWaterSheep()
        route = WaterSheepRunnable(ws, question="Which team?", options=["billing", "shipping"])
        self.assertEqual(route.invoke("I was charged twice.")["choice"], "billing")
        self.assertEqual(route.invoke({"text": "yes", "options": None})["noul"], 0.9)
        self.assertEqual(len(route.batch(["a", "b"])), 2)
        chain = RunnableLambda(lambda s: s.strip()) | route | RunnableLambda(lambda a: a["choice"])
        self.assertEqual(chain.invoke("  hello  "), "billing")
        with self.assertRaises(ValueError):
            route.invoke({"question": "x"})

    def test_tool(self):
        from langchain_core.tools import BaseTool
        from watersheep.integrations.langchain import watersheep_tool
        tool = watersheep_tool(FakeWaterSheep())
        self.assertIsInstance(tool, BaseTool)
        out = json.loads(tool.invoke({"text": "yes", "question": "Agree?"}))
        self.assertEqual(out["noul"], 0.9)
        out = json.loads(tool.invoke({"text": "x", "question": "Team?", "options": ["a", "b"]}))
        self.assertEqual(out["choice"], "a")


@unittest.skipUnless(HAS_LLAMAINDEX, "llama-index-core is not installed")
class LlamaIndexTests(unittest.TestCase):
    def test_relevance_filter(self):
        from llama_index.core.schema import NodeWithScore, TextNode
        from watersheep.integrations.llamaindex import WaterSheepRelevanceFilter
        ws = FakeWaterSheep()
        keep = WaterSheepRelevanceFilter(ws, threshold=0.5)
        nodes = [NodeWithScore(node=TextNode(text=t), score=0.1) for t in ("no", "yes, this", "nope")]
        out = keep.postprocess_nodes(nodes, query_str="refund policy")
        self.assertEqual([n.node.get_content() for n in out], ["yes, this"])
        self.assertEqual(out[0].score, 0.9)
        self.assertIn("refund policy", ws.requests[0]["questions"]["q"]["instructions"])
        self.assertEqual(len(WaterSheepRelevanceFilter(ws, threshold=0.1).postprocess_nodes(
            nodes, query_str="q")), 3)
        with self.assertRaises(ValueError):
            keep.postprocess_nodes(nodes)

    def test_function_tool(self):
        from watersheep.integrations.llamaindex import watersheep_function_tool
        tool = watersheep_function_tool(FakeWaterSheep())
        self.assertEqual(tool.metadata.name, "watersheep")
        out = json.loads(tool.call(text="x", question="Team?", options=["a", "b"]).content)
        self.assertEqual(out["choice"], "a")


if __name__ == "__main__":
    unittest.main()
