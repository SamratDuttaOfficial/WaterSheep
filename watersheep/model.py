"""The decision model: an encoder that scores all options in one pass."""
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

TYPE_TAG = {"binary": "[yes/no]", "choice": "[choose]", "score": "[rate]", "multi": "[select all]"}
TYPE_ID = {"binary": 0, "choice": 1, "score": 2, "multi": 3}
MULTI = TYPE_ID["multi"]
TYPE_NAME = {v: k for k, v in TYPE_ID.items()}


class Tok:
    """Tokenizer wrapper."""

    def __init__(self, hf_tok):
        self.t = hf_tok
        self.cls = hf_tok.cls_token_id if hf_tok.cls_token_id is not None else hf_tok.bos_token_id
        self.sep = hf_tok.sep_token_id if hf_tok.sep_token_id is not None else hf_tok.eos_token_id
        self.mask = hf_tok.mask_token_id
        self.pad = hf_tok.pad_token_id if hf_tok.pad_token_id is not None else 0
        if self.cls is None or self.sep is None or self.mask is None:
            raise ValueError("the encoder's tokenizer needs CLS/SEP/MASK tokens")

    @classmethod
    def load(cls, name_or_dir: str) -> "Tok":
        from transformers import AutoTokenizer
        return cls(AutoTokenizer.from_pretrained(name_or_dir))

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        if not texts:
            return []
        return self.t(list(texts), add_special_tokens=False)["input_ids"]

    def save(self, d) -> None:
        self.t.save_pretrained(str(d))


class CharTok:
    """Byte tokenizer for the offline self-test."""
    cls, sep, mask, pad = 0, 1, 2, 3
    vocab_size = 260

    def encode_batch(self, texts):
        return [[4 + b for b in t.encode("utf-8")] for t in texts]

    def save(self, d) -> None:
        pass


def assemble(tok, q_ids: List[int], opt_ids: List[List[int]], s_ids: List[int],
             max_len: int) -> Tuple[List[int], List[int]]:
    ids = [tok.cls] + q_ids + [tok.sep]
    pos = []
    for o in opt_ids:
        pos.append(len(ids))
        ids += [tok.mask] + o
    ids.append(tok.sep)
    room = max_len - len(ids) - 1
    if room > 0 and s_ids:
        ids += s_ids[:room]
    ids.append(tok.sep)
    if len(ids) > max_len:
        ids = ids[:max_len - 1] + [tok.sep]
        pos = [p for p in pos if p < max_len - 1]
    return ids, pos


def encode_records(tok, recs: List[dict], max_len: int, max_q: int, max_opt: int,
                   chunk: int = 2048):
    """Yield (ids, option positions) per record."""
    for i in range(0, len(recs), chunk):
        part = recs[i:i + chunk]
        qs = tok.encode_batch([TYPE_TAG[r["type"]] + " " + r["question"] for r in part])
        ss = tok.encode_batch([r.get("state") or "" for r in part])
        flat = [o for r in part for o in r["options"]]
        os_ = tok.encode_batch(flat)
        k = 0
        for r, q, s in zip(part, qs, ss):
            n = len(r["options"])
            opts = [o[:max_opt] for o in os_[k:k + n]]
            k += n
            yield assemble(tok, q[:max_q], opts, s, max_len)


def load_encoder(spec, grad_checkpointing: bool = False):
    """Encoder from a model id, a directory or a config dict."""
    from transformers import AutoConfig, AutoModel
    if isinstance(spec, dict):
        from transformers import ModernBertConfig
        enc = AutoModel.from_config(ModernBertConfig(**spec))
    else:
        conf = AutoConfig.from_pretrained(spec)
        if hasattr(conf, "reference_compile"):
            conf.reference_compile = False  # torch.compile needs triton
        enc = AutoModel.from_pretrained(spec, config=conf)
    if grad_checkpointing and hasattr(enc, "gradient_checkpointing_enable"):
        enc.gradient_checkpointing_enable()
    return enc


def encoder_from_dir(d):
    """Encoder architecture from an export, without weights."""
    from transformers import AutoConfig, AutoModel
    conf = AutoConfig.from_pretrained(str(d))
    if hasattr(conf, "reference_compile"):
        conf.reference_compile = False
    return AutoModel.from_config(conf)


class WaterSheepNet(nn.Module):
    def __init__(self, encoder, head_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        h = encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h))
        self.mix: Optional[nn.Module] = None
        if head_layers > 0:
            layer = nn.TransformerEncoderLayer(h, nhead=max(1, h // 64), dim_feedforward=2 * h,
                                               dropout=dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.mix = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        self.out = nn.Linear(h, 1)

    def forward(self, input_ids, attention_mask, opt_pos, opt_mask):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx = opt_pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, hs.size(-1))
        x = self.proj(torch.gather(hs, 1, idx))
        if self.mix is not None:
            x = self.mix(x, src_key_padding_mask=~opt_mask)
        logits = self.out(x).squeeze(-1).float()
        return logits.masked_fill(~opt_mask, -1e4)

    def param_groups(self, lr: float, head_lr: float, wd: float):
        enc_d, enc_nd, head_d, head_nd = [], [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_enc = n.startswith("encoder.")
            decay = p.dim() >= 2 and "norm" not in n.lower() and "embed" not in n.lower()
            (enc_d if is_enc and decay else enc_nd if is_enc else
             head_d if decay else head_nd).append(p)
        groups = [{"params": enc_d, "lr": lr, "weight_decay": wd, "base_lr": lr},
                  {"params": enc_nd, "lr": lr, "weight_decay": 0.0, "base_lr": lr},
                  {"params": head_d, "lr": head_lr, "weight_decay": wd, "base_lr": head_lr},
                  {"params": head_nd, "lr": head_lr, "weight_decay": 0.0, "base_lr": head_lr}]
        return [g for g in groups if g["params"]]


def soft_ce(logits: torch.Tensor, target: torch.Tensor, opt_mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a soft target."""
    logp = torch.log_softmax(logits, dim=-1)
    return -(target * logp).masked_fill(~opt_mask, 0.0).sum(-1)


def decision_loss(logits, target, opt_mask, typ) -> torch.Tensor:
    """Per-example loss: cross-entropy, or per-option BCE for multi."""
    ce = soft_ce(logits, target, opt_mask)
    multi = typ == MULTI
    if not bool(multi.any()):
        return ce
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.clamp(-30, 30), target, reduction="none").masked_fill(~opt_mask, 0.0).sum(-1)
    return torch.where(multi, bce, ce)
