"""Dataset build and packed training arrays."""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import interrupt
from .model import TYPE_ID, encode_records
from .paths import P
from .schema import hard_target, validate
from .util import (LOG, iter_jsonl, norm_text, read_json, rmtree, stable_int, write_json,
                   write_jsonl)

SPLITS = ("train", "val", "test", "zeroshot")


def raw_file(name: str) -> Path:
    return P.raw / (name.replace(":", "__") + ".jsonl")


def synth_records():
    """Accepted synthetic records."""
    for f in sorted(P.synth.glob("shard_*.jsonl")):
        if not f.with_suffix(".done").exists():
            continue
        for row in iter_jsonl(f):
            if row.get("status") == "accepted" and row.get("record"):
                yield row["record"]


def _shuffle_options(r: dict) -> dict:
    if r["type"] not in ("choice", "multi"):
        return r
    rng = random.Random(stable_int("opts:" + r["id"]))
    order = list(range(len(r["options"])))
    rng.shuffle(order)
    r = dict(r)
    r["options"] = [r["options"][i] for i in order]
    r["answer"] = sorted(order.index(a) for a in r["answer"]) if r["type"] == "multi" else \
        order.index(r["answer"])
    if r.get("target"):
        r["target"] = [r["target"][i] for i in order]
    return r


def build(cfg, source_names: List[str], dropped: List[str]) -> dict:
    from .corpus import excluder
    from . import describe
    extras = describe.load()
    heldout = set(cfg.heldout_sources)
    removed_by_user = excluder()
    seen = set()
    out: Dict[str, list] = {s: [] for s in SPLITS}
    per_source: Dict[str, int] = {}
    bad = dup = edited = 0

    def add(r: dict, repeat: int = 1):
        nonlocal bad, dup, edited
        if removed_by_user(r):
            edited += 1
            return
        r = describe.augment(r, extras, cfg)
        ok, _ = validate(r, cfg.max_options)
        if not ok:
            bad += 1
            return
        key = stable_int(norm_text(r.get("state", "")) + "|" + norm_text(r["question"]) + "|" +
                         "|".join(norm_text(o) for o in r["options"]))
        if key in seen:
            dup += 1
            return
        seen.add(key)
        r = _shuffle_options(r)
        if r.get("target") is None:
            r["target"] = hard_target(r, cfg.label_smoothing, cfg.score_neighbor_mass)
        if r["source"] in heldout:
            split = "zeroshot"
        else:
            u = (stable_int("split:%s:%d" % (r["id"], cfg.seed)) % 100000) / 100000.0
            split = ("test" if u < cfg.test_fraction else
                     "val" if u < cfg.test_fraction + cfg.val_fraction else "train")
        for _ in range(repeat if split == "train" else 1):
            out[split].append(r)
        per_source[r["source"]] = per_source.get(r["source"], 0) + 1

    for name in source_names:
        interrupt.check()
        if name in dropped:
            continue
        for r in iter_jsonl(raw_file(name)):
            add(r)
    for r in synth_records():
        add(r, max(1, cfg.synth_repeat))

    rng = random.Random(cfg.seed)
    for s in SPLITS:
        rng.shuffle(out[s])
        write_jsonl(P.build / (s + ".jsonl"), out[s])
    stats = {"splits": {s: len(v) for s, v in out.items()},
             "types": {s: {t: sum(1 for r in v if r["type"] == t) for t in TYPE_ID}
                       for s, v in out.items()},
             "synthetic_train": sum(1 for r in out["train"] if r["kind"] == "synth"),
             "per_source": dict(sorted(per_source.items())),
             "invalid": bad, "duplicates": dup, "removed_in_ui": edited,
             "dropped_sources": dropped}
    write_json(P.build / "stats.json", stats)
    LOG.info("built: %s  (synthetic in train: %d, duplicates removed: %d, invalid: %d, "
             "removed in the web UI: %d)", ", ".join("%s=%d" % (k, v) for k, v in stats["splits"].items()),
             stats["synthetic_train"], dup, bad, edited)
    return stats


def pack(tok, split: str, cfg, out_dir: Path, sources: List[str]) -> Path:
    """Tokenize one split into flat arrays."""
    f = out_dir / (split + ".npz")
    if f.exists():
        return f
    recs = list(iter_jsonl(P.build / (split + ".jsonl")))
    src_id = {s: i for i, s in enumerate(sources)}
    ids, off, pos, poff, tgt, ans, typ, src, lens = [], [0], [], [0], [], [], [], [], []
    for r, (seq, p) in zip(recs, encode_records(tok, recs, cfg.max_len, cfg.max_question_tokens,
                                                 cfg.max_option_tokens)):
        if len(p) != len(r["options"]):
            continue
        ids.extend(seq)
        off.append(len(ids))
        pos.extend(p)
        poff.append(len(pos))
        tgt.extend(r["target"])
        ans.append(-1 if r["type"] == "multi" else r["answer"])
        typ.append(TYPE_ID[r["type"]])
        src.append(src_id.setdefault(r["source"], len(src_id)))
        lens.append(len(seq))
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (split + ".tmp.npz")
    np.savez(tmp, ids=np.asarray(ids, np.int32), off=np.asarray(off, np.int64),
             pos=np.asarray(pos, np.int32), poff=np.asarray(poff, np.int64),
             tgt=np.asarray(tgt, np.float32), ans=np.asarray(ans, np.int16),
             typ=np.asarray(typ, np.int8), src=np.asarray(src, np.int32),
             lens=np.asarray(lens, np.int32))
    tmp.replace(f)
    write_json(out_dir / "sources.json", list(src_id))
    LOG.info("packed %s: %d examples, %.1fM tokens", split, len(ans), len(ids) / 1e6)
    return f


class Packed:
    def __init__(self, path: Path):
        z = np.load(path)
        self.ids, self.off = z["ids"], z["off"]
        self.pos, self.poff = z["pos"], z["poff"]
        self.tgt, self.ans = z["tgt"], z["ans"].astype(np.int64)
        self.typ, self.src, self.lens = z["typ"], z["src"], z["lens"]
        self.sources = read_json(path.parent / "sources.json", []) or []

    def __len__(self) -> int:
        return len(self.ans)

    def n_opts(self, i: int) -> int:
        return int(self.poff[i + 1] - self.poff[i])

    def batch(self, idx: List[int], pad: int, device):
        import torch
        T = int(max(self.lens[i] for i in idx))
        K = int(max(self.n_opts(i) for i in idx))
        B = len(idx)
        x = np.full((B, T), pad, np.int64)
        am = np.zeros((B, T), np.int64)
        op = np.zeros((B, K), np.int64)
        om = np.zeros((B, K), bool)
        tg = np.zeros((B, K), np.float32)
        for b, i in enumerate(idx):
            s = self.ids[self.off[i]:self.off[i + 1]]
            x[b, :len(s)] = s
            am[b, :len(s)] = 1
            p = self.pos[self.poff[i]:self.poff[i + 1]]
            op[b, :len(p)] = p
            om[b, :len(p)] = True
            tg[b, :len(p)] = self.tgt[self.poff[i]:self.poff[i + 1]]
        t = lambda a: torch.from_numpy(a).pin_memory().to(device, non_blocking=True) \
            if device.type == "cuda" else torch.from_numpy(a)
        return {"input_ids": t(x), "attention_mask": t(am), "opt_pos": t(op),
                "opt_mask": t(om), "target": t(tg),
                "answer": torch.as_tensor(self.ans[idx]), "type": torch.as_tensor(self.typ[idx])}


def make_batches(lens: np.ndarray, batch_tokens: int, seed: int, shuffle: bool = True,
                 max_rows: int = 256) -> List[List[int]]:
    """Length-grouped batches under a padded-token budget, deterministic per seed."""
    n = len(lens)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n) if shuffle else np.arange(n)
    batches: List[List[int]] = []
    chunk = 8192
    for c in range(0, n, chunk):
        part = order[c:c + chunk]
        part = part[np.argsort(-lens[part], kind="stable")]
        cur: List[int] = []
        cur_max = 0
        for i in part:
            L = int(lens[i])
            if cur and (max(cur_max, L) * (len(cur) + 1) > batch_tokens or len(cur) >= max_rows):
                batches.append(cur)
                cur, cur_max = [], 0
            cur.append(int(i))
            cur_max = max(cur_max, L)
        if cur:
            batches.append(cur)
    if shuffle:
        perm = rng.permutation(len(batches))
        batches = [batches[i] for i in perm]
    return batches


def clean_tok() -> None:
    rmtree(P.tok)
