import numpy as np
import torch
from transformers import AutoTokenizer, Pipeline

TYPES = {"noul": "binary", "binary": "binary", "yes/no": "binary", "boolean": "binary",
         "choice": "choice", "score": "score", "multi": "multi", "multi_choice": "multi",
         "multilabel": "multi", "multi-label": "multi"}
TAGS = {"binary": "[yes/no]", "choice": "[choose]", "score": "[rate]", "multi": "[select all]"}
DEFAULT_QUESTION = {"choice": "Which label fits the text?", "multi": "Which labels apply to the text?"}
YESNO = ["yes", "no"]


def _softmax(z, t):
    z = np.asarray(z, np.float64) / max(1e-6, t)
    e = np.exp(z - z.max())
    return e / e.sum()


def _sigmoid(z, t):
    z = np.clip(np.asarray(z, np.float64) / max(1e-6, t), -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def _digits(opts):
    return all(len(o) == 1 and o.isdigit() for o in opts)


def _infer_type(opts):
    low = [o.strip().lower() for o in opts]
    if low == YESNO:
        return "binary"
    if _digits(low):
        v = [int(o) for o in low]
        if v == list(range(v[0], v[0] + len(v))):
            return "score"
    return "choice"


class WaterSheepPipeline(Pipeline):
    _load_tokenizer = False
    _load_processor = False
    _load_image_processor = False
    _load_video_processor = False
    _load_feature_extractor = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model.name_or_path, subfolder="tokenizer",
                revision=getattr(self.model.config, "_commit_hash", None))

    def _sanitize_parameters(self, question=None, options=None, candidate_labels=None, type=None,
                             multi_label=None, threshold=None, hypothesis_template=None):
        pre = {}
        if question is not None:
            pre["question"] = question
        opts = options if options is not None else candidate_labels
        if isinstance(opts, str):
            opts = [o.strip() for o in opts.split(",") if o.strip()]
        if opts is not None:
            pre["options"] = list(opts)
        if type is not None or multi_label:
            pre["type"] = type or "multi"
        return pre, {}, {} if threshold is None else {"threshold": threshold}

    def preprocess(self, text, question=None, options=None, type=None):
        opts = [str(o) for o in (options or YESNO)]
        t = TYPES.get(str(type or "").lower()) or _infer_type(opts)
        if t == "binary":
            opts = list(YESNO)
        question = question or DEFAULT_QUESTION.get(t)
        if not question:
            raise ValueError("a question is required")
        return {"text": "" if text is None else str(text), "question": str(question), "type": t, "options": opts}

    def _forward(self, x):
        return {**x, "probs": self._probs(x["type"], x["text"], x["question"], x["options"])}

    def postprocess(self, x, threshold=None):
        t, opts, p = x["type"], x["options"], x["probs"]
        res = {"type": t, "probs": {o: float(v) for o, v in zip(opts, p)}}
        if t == "multi":
            thr = self.model.config.multi_threshold if threshold is None else float(threshold)
            res.update(answer=[o for o, v in zip(opts, p) if v >= thr],
                       confidence=float(np.mean([max(v, 1 - v) for v in p])))
        else:
            i = int(np.argmax(p))
            res.update(answer=opts[i], index=i, confidence=float(p[i]))
        if t == "binary":
            res["p_yes"] = float(p[0])
        elif t == "score":
            res["expected"] = float(sum(k * v for k, v in enumerate(p))) + (int(opts[0]) if _digits(opts) else 0)
        return res

    def _logits(self, t, text, question, groups):
        cfg, tok = self.model.config, self.tokenizer
        cls = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
        sep = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
        mask, pad = tok.mask_token_id, tok.pad_token_id if tok.pad_token_id is not None else 0
        enc = lambda xs: tok(xs, add_special_tokens=False)["input_ids"]
        q = enc([TAGS[t] + " " + question])[0][:cfg.max_question_tokens]
        s = enc([text])[0]
        flat = enc([o for g in groups for o in g])
        items, k = [], 0
        for g in groups:
            ids, pos = [cls] + q + [sep], []
            for o in flat[k:k + len(g)]:
                pos.append(len(ids))
                ids += [mask] + o[:cfg.max_option_tokens]
            k += len(g)
            ids.append(sep)
            room = cfg.max_len - len(ids) - 1
            if room > 0 and s:
                ids += s[:room]
            ids.append(sep)
            if len(ids) > cfg.max_len:
                ids = ids[:cfg.max_len - 1] + [sep]
                pos = [p for p in pos if p < cfg.max_len - 1]
            items.append((ids, pos))
        out, dev = [], self.device
        bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
        for i in range(0, len(items), 64):
            part = items[i:i + 64]
            n, m = max(len(a) for a, _ in part), max(len(p) for _, p in part)
            x = torch.full((len(part), n), pad, dtype=torch.long)
            am = torch.zeros((len(part), n), dtype=torch.long)
            op = torch.zeros((len(part), m), dtype=torch.long)
            om = torch.zeros((len(part), m), dtype=torch.bool)
            for b, (ids, pos) in enumerate(part):
                x[b, :len(ids)] = torch.tensor(ids)
                am[b, :len(ids)] = 1
                op[b, :len(pos)] = torch.tensor(pos)
                om[b, :len(pos)] = True
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
                z = self.model(x.to(dev), am.to(dev), op.to(dev), om.to(dev)).logits.float().cpu().numpy()
            for (ids, pos), g, zz in zip(part, groups[i:i + 64], z):
                out.append(zz[:len(pos)] if len(pos) == len(g) else None)
        return out

    def _probs(self, t, text, question, opts, size=0):
        cfg = self.model.config
        temp = (cfg.temperatures or {}).get(t, 1.0)
        n = len(opts)
        size = min(size or max(2, min(10, int(cfg.max_options or 10))), n)
        while True:
            if n <= size:
                (z,) = self._logits(t, text, question, [opts])
                if z is not None:
                    return _sigmoid(z, temp) if t == "multi" else _softmax(z, temp)
            else:
                groups = [list(range(k, min(n, k + size))) for k in range(0, n, size)]
                zs = self._logits(t, text, question, [[opts[i] for i in g] for g in groups])
                if all(z is not None for z in zs):
                    break
            if size <= 2:
                raise ValueError("the options do not fit in the model's input - shorten them")
            size = max(2, min(size - 1, 8) if size > 8 else size // 2)
        if t == "multi":
            p = np.zeros(n)
            for g, z in zip(groups, zs):
                p[g] = _sigmoid(z, temp)
            return p
        keep = max(1, size // len(groups))
        local, finalists = {}, []
        for g, z in zip(groups, zs):
            pg = _softmax(z, temp)
            local.update({i: pg[k] for k, i in enumerate(g)})
            finalists += [g[k] for k in np.argsort(-pg)[:keep]]
        pf = self._probs(t, text, question, [opts[i] for i in finalists], size)
        p = np.zeros(n)
        for g in groups:
            top = max((i for i in g if i in finalists), key=lambda i: local[i])
            scale = pf[finalists.index(top)] / max(1e-12, local[top])
            for i in g:
                p[i] = local[i] * scale
        for k, i in enumerate(finalists):
            p[i] = pf[k]
        return p / p.sum()
