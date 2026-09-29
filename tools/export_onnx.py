#!/usr/bin/env python
"""Export a model to ONNX with 8-bit weights and check it against PyTorch.

Writes onnx/model_quantized.onnx into the export. Needs: pip install onnx onnxruntime onnxscript

  python tools/export_onnx.py                          the newest export
  python tools/export_onnx.py --model out/export/NAME  another export
  python tools/export_onnx.py --keep-fp32              also keep onnx/model.onnx
"""
from __future__ import annotations
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from watersheep.infer import WaterSheep, find_export                     # noqa: E402
from watersheep.metrics import sigmoid, softmax                          # noqa: E402
from watersheep.model import encode_records                              # noqa: E402

NAMES = ["input_ids", "attention_mask", "option_positions", "option_mask"]
CASES = [
    ("choice", "I was charged twice for order #4411 and still haven't received a refund.",
     "Which team should handle this?", ["billing", "shipping", "technical support", "sales"]),
    ("binary", "The production database is down and customers cannot check out.",
     "Does this need immediate attention?", ["yes", "no"]),
    ("score", "This is the third time I'm writing. Nobody answers and I'm about to cancel.",
     "How frustrated is the customer?", ["calm", "annoyed", "frustrated", "furious"]),
    ("multi", "The box arrived crushed, one mug was broken and the invoice has the wrong address.",
     "Which problems does the customer report?",
     ["damaged item", "wrong address", "late delivery", "missing item", "billing error"]),
    ("choice", "User: what will the weather be like in Paris tomorrow?",
     "Which tool should the assistant call?", ["web_search", "weather_forecast", "calculator", "calendar"]),
    ("score", "The parcel arrived six days after the promised date. " * 40,
     "Rate the delay.", ["1", "2", "3", "4", "5"]),
]


class Wrapped(torch.nn.Module):
    """WaterSheepNet with integer inputs only."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, input_ids, attention_mask, option_positions, option_mask):
        return self.net(input_ids, attention_mask, option_positions, option_mask > 0)


def batch(ws, recs):
    """Padded int64 arrays for records, like WaterSheep._logits."""
    m = ws.meta
    enc = list(encode_records(ws.tok, recs, m["max_len"], m["max_question_tokens"], m["max_option_tokens"]))
    T, K = max(len(a) for a, _ in enc), max(len(p) for _, p in enc)
    x = np.full((len(enc), T), getattr(ws.tok, "pad", 0), np.int64)
    am, op, om = np.zeros_like(x), np.zeros((len(enc), K), np.int64), np.zeros((len(enc), K), np.int64)
    for b, (ids, pos) in enumerate(enc):
        x[b, :len(ids)], am[b, :len(ids)] = ids, 1
        op[b, :len(pos)], om[b, :len(pos)] = pos, 1
    return [x, am, op, om]


def export(ws, path: Path) -> None:
    model = Wrapped(ws.net).eval()
    args = tuple(torch.from_numpy(a) for a in batch(ws, [dict(type=t, state=s, question=q, options=o)
                                                          for t, s, q, o in CASES[:2]]))
    B, T, K = torch.export.Dim("batch"), torch.export.Dim("tokens", max=4096), torch.export.Dim("options")
    shapes = {"input_ids": {0: B, 1: T}, "attention_mask": {0: B, 1: T},
              "option_positions": {0: B, 1: K}, "option_mask": {0: B, 1: K}}
    fast = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)   # the fused encoder-layer kernel has no ONNX form
    try:
        with torch.no_grad():
            torch.onnx.export(model, args, str(path), input_names=NAMES, output_names=["logits"],
                              dynamic_shapes=shapes, opset_version=18, dynamo=True, external_data=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(fast)


def weight_only_int8(model):
    """Per-channel 8-bit weights with DequantizeLinear; activations stay in float."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    def quantize(w, axis):
        s = np.maximum(np.abs(w).max(axis=1 - axis), 1e-12) / 127.0
        wq = np.clip(np.round(w / (s[None, :] if axis == 1 else s[:, None])), -127, 127).astype(np.int8)
        return wq, s.astype(np.float32)

    g = model.graph
    inits = {t.name: t for t in g.initializer}
    nodes, done = [], set()
    for n in g.node:
        w_in = n.input[0] if n.op_type == "Gather" else n.input[1] if n.op_type in ("MatMul", "Gemm") else None
        w = numpy_helper.to_array(inits[w_in]) if w_in in inits else None
        if w is None or w.ndim != 2 or w.size < 16384:
            nodes.append(n)
            continue
        if w_in in done and n.op_type != "Gather":     # a weight shared by several nodes
            inputs = list(n.input)
            inputs[1] = w_in + "_dq"
            nodes.append(helper.make_node(n.op_type, inputs, list(n.output), name=n.name))
            nodes[-1].attribute.extend(n.attribute)
            continue
        if w_in in done:
            nodes.append(n)
            continue
        done.add(w_in)
        if n.op_type == "Gather":
            wq, s = quantize(w, 0)
            g.initializer.extend([numpy_helper.from_array(wq, w_in + "_q"),
                                  numpy_helper.from_array(s[:, None], w_in + "_s")])
            nodes += [helper.make_node("Gather", [w_in + "_q", n.input[1]], [w_in + "_gq"], axis=0),
                      helper.make_node("Cast", [w_in + "_gq"], [w_in + "_gf"], to=TensorProto.FLOAT),
                      helper.make_node("Gather", [w_in + "_s", n.input[1]], [w_in + "_gs"], axis=0),
                      helper.make_node("Mul", [w_in + "_gf", w_in + "_gs"], list(n.output))]
            continue
        axis = 0 if any(a.name == "transB" and a.i for a in n.attribute) else 1   # output channels
        wq, s = quantize(w, axis)
        g.initializer.extend([numpy_helper.from_array(wq, w_in + "_q"), numpy_helper.from_array(s, w_in + "_s"),
                              numpy_helper.from_array(np.zeros(s.shape, np.int8), w_in + "_z")])
        nodes.append(helper.make_node("DequantizeLinear", [w_in + "_q", w_in + "_s", w_in + "_z"],
                                      [w_in + "_dq"], axis=axis))
        inputs = list(n.input)
        inputs[1] = w_in + "_dq"
        nodes.append(onnx.helper.make_node(n.op_type, inputs, list(n.output), name=n.name))
        nodes[-1].attribute.extend(n.attribute)
    used = {i for n in nodes for i in n.input}
    keep = [t for t in g.initializer if t.name not in done or t.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    del g.node[:]
    g.node.extend(nodes)
    return model


def check(ws, paths) -> bool:
    """Compare each ONNX file with PyTorch on the test cases."""
    import onnxruntime as ort
    ok = True
    sessions = {p.name: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for p in paths}
    for t, s, q, o in CASES:
        rec = dict(type=t, state=s, question=q, options=o)
        (ref,), _ = ws._logits([rec])
        T = ws.temps.get(t, 1.0)
        prob = (lambda z: sigmoid(z, T)) if t == "multi" else (lambda z: softmax(z, T))
        for name, sess in sessions.items():
            z = sess.run(None, dict(zip(NAMES, batch(ws, [rec]))))[0][0][:len(o)]
            dp = float(np.abs(prob(z) - prob(ref)).max())
            same = (prob(z) >= ws.threshold).tolist() == (prob(ref) >= ws.threshold).tolist() if t == "multi" \
                else int(np.argmax(z)) == int(np.argmax(ref))
            ok &= same and dp < 0.1
            print("  %-22s %-7s max prob diff %.4f  %s" % (name, t, dp, "same answer" if same else "DIFFERENT ANSWER"))
    x = batch(ws, [dict(type=t, state=s, question=q, options=o) for t, s, q, o in CASES])
    for name, sess in sessions.items():
        got = sess.run(None, dict(zip(NAMES, x)))[0]
        print("  %-22s batch of %d with padding: output %s" % (name, len(CASES), tuple(got.shape)))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="export folder or export name (default: the newest export)")
    ap.add_argument("--keep-fp32", action="store_true", help="also keep onnx/model.onnx")
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):   # the exporter prints emoji
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        import onnx
        import onnxruntime  # noqa: F401
    except ImportError:
        print("needs: pip install onnx onnxruntime onnxscript")
        return 1
    d = find_export(a.model)
    if not d:
        print("no export found%s" % (" at %r" % a.model if a.model else ""))
        return 1
    ws = WaterSheep.load(d, device="cpu")
    out = d / "onnx"
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = (out if a.keep_fp32 else Path(tmp)) / "model.onnx"
        print("exporting %s ..." % d.name)
        export(ws, fp32)
        q = out / "model_quantized.onnx"
        onnx.save(weight_only_int8(onnx.load(str(fp32))), str(q))
        print("checking against PyTorch:")
        ok = check(ws, [fp32, q])
        for f in [fp32, q]:
            print("  %-22s %.0f MB" % (f.name, f.stat().st_size / 2 ** 20))
    if not ok:
        print("the ONNX model gives different answers - not safe to publish")
        return 1
    print("wrote %s" % q)
    return 0


if __name__ == "__main__":
    sys.exit(main())
