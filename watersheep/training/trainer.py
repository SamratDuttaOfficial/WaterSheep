"""Training loop with resumable checkpoints."""
from __future__ import annotations
import gc
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ..core import interrupt
from ..data.dataset import Packed, make_batches
from ..metrics import probs_for, summarize
from ..model import WaterSheepNet, decision_loss, load_encoder, soft_ce
from ..core.util import LOG, append_jsonl, human_num, human_time, now_iso, write_json


def pick_device(want: str) -> torch.device:
    if want.startswith("cuda") and not torch.cuda.is_available():
        LOG.warning("cuda requested but unavailable - using the CPU (slow)")
        return torch.device("cpu")
    return torch.device(want)


def amp_ctx(device: torch.device, name: str):
    if device.type != "cuda":
        return torch.autocast("cpu", enabled=False)
    dt = torch.bfloat16 if name == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dt)


def build_model(cfg, encoder_spec) -> WaterSheepNet:
    enc = load_encoder(encoder_spec, cfg.grad_checkpointing)
    return WaterSheepNet(enc, cfg.head_layers, cfg.head_dropout)


def save_ckpt(path: Path, model, opt, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"model": model.state_dict(), "opt": opt.state_dict() if opt is not None else None,
            "state": state, "saved_at": now_iso(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)


def load_ckpt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


@torch.no_grad()
def predict(model, ds: Packed, idx: List[int], batch_tokens: int, pad: int,
            device, amp: str) -> List[np.ndarray]:
    model.eval()
    idx = list(idx)
    out: Dict[int, np.ndarray] = {}
    lens = ds.lens[idx]
    for b in make_batches(lens, batch_tokens, seed=0, shuffle=False):
        real = [idx[j] for j in b]
        t = ds.batch(real, pad, device)
        with amp_ctx(device, amp):
            logits = model(t["input_ids"], t["attention_mask"], t["opt_pos"], t["opt_mask"])
        lg = logits.float().cpu().numpy()
        for row, i in enumerate(real):
            out[i] = lg[row, :ds.n_opts(i)]
    model.train()
    return [out[i] for i in idx]


def evaluate_rows(model, ds: Packed, idx, batch_tokens, pad, device, amp,
                  temps: Optional[dict] = None) -> List[dict]:
    from ..model import TYPE_NAME
    logits = predict(model, ds, idx, batch_tokens, pad, device, amp)
    rows = []
    for i, z in zip(idx, logits):
        t = int(ds.typ[i])
        T = (temps or {}).get(TYPE_NAME[t], 1.0)
        rows.append({"p": probs_for(z, t, T), "z": z, "a": int(ds.ans[i]), "t": t,
                     "y": ds.tgt[ds.poff[i]:ds.poff[i + 1]] > 0.5,
                     "s": ds.sources[int(ds.src[i])] if ds.sources else str(int(ds.src[i]))})
    return rows


def autotune(model, cfg, device, pad: int) -> int:
    """Largest padded-token budget that fits in GPU memory."""
    if device.type != "cuda":
        return 2048
    total = torch.cuda.get_device_properties(device).total_memory
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reserve = n_params * 4 * 2
    L, K = cfg.max_len, min(cfg.max_options, 10)
    vocab = int(getattr(model.encoder.config, "vocab_size", 1000))

    def attempt(rows: int) -> int:
        """Peak memory of one forward and backward pass."""
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randint(5, max(6, min(vocab - 1, 30000)), (rows, L), device=device)
        am = torch.ones_like(x)
        op = (torch.arange(K, device=device) * 3 + 1).unsqueeze(0).expand(rows, -1).contiguous()
        om = torch.ones(rows, K, dtype=torch.bool, device=device)
        tg = torch.full((rows, K), 1.0 / K, device=device)
        with amp_ctx(device, cfg.amp_dtype):
            logits = model(x, am, op, om)
        soft_ce(logits, tg, om).sum().backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device)

    def measure(rows: int) -> int:
        peak = attempt(rows)
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        return peak

    try:
        p2, p4 = measure(2), measure(4)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        LOG.warning("autotune: even 2 x %d does not fit (%s)", L, str(e).splitlines()[0][:80])
        return 0
    per_row = max(1, (p4 - p2) / 2)
    base = p2 - 2 * per_row
    budget = 0.76 * total - reserve
    rows = int((budget - base) // per_row)
    rows = max(1, min(rows, 65536 // L, cfg.target_batch_examples))
    peak = measure(rows)
    LOG.info("autotune: %d tokens per batch (%d x %d worst case) peaks at %.2f GB + %.2f GB "
             "optimizer of %.2f GB  (%.0f MB per row)", rows * L, rows, L, peak / 1e9,
             reserve / 1e9, total / 1e9, per_row / 1e6)
    return rows * L


def lr_factor(step: int, total: int, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))


def train(*, cfg, run_dir: Path, run_id: str, tok_dir: Path, pad: int, encoder_spec,
          metrics_file: Path, saved_budget: int = 0) -> dict:
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tr, va = Packed(tok_dir / "train.npz"), Packed(tok_dir / "val.npz")
    if len(tr) == 0:
        raise RuntimeError("the training split is empty")
    run_dir.mkdir(parents=True, exist_ok=True)
    last, best_path = run_dir / "last.pt", run_dir / "best.pt"

    model = build_model(cfg, encoder_spec).to(device)
    groups = model.param_groups(cfg.lr, cfg.head_lr, cfg.weight_decay)
    try:
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-6, fused=device.type == "cuda")
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-6)

    st = {"step": 0, "epoch": 0, "pos": 0, "best": float("inf"), "bad_evals": 0,
          "batch_tokens": saved_budget or cfg.batch_tokens, "best_metrics": {}}
    if last.exists():
        blob = load_ckpt(last)
        model.load_state_dict(blob["model"])
        try:
            opt.load_state_dict(blob["opt"])
        except Exception as e:
            LOG.warning("optimizer state incompatible (%s) - fresh optimizer", e)
        st.update(blob.get("state", {}))
        try:
            torch.set_rng_state(blob["torch_rng"])
        except Exception:
            pass
        LOG.info("resuming %s at step %d (epoch %d, batch %d)", run_id, st["step"],
                 st["epoch"], st["pos"])

    if not st["batch_tokens"]:
        bt = autotune(model, cfg, device, pad)
        if not bt and not cfg.grad_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
            LOG.warning("nothing fits - enabling gradient checkpointing")
            model.encoder.gradient_checkpointing_enable()
            bt = autotune(model, cfg, device, pad)
        st["batch_tokens"] = bt or 512
    bt = int(st["batch_tokens"])

    n_tr = len(tr)
    n_epochs = int(math.ceil(cfg.epochs))
    rows_cap = max(1, cfg.target_batch_examples)

    def epoch_batches(e: int):
        return make_batches(tr.lens, bt, seed=cfg.seed * 1000 + e, max_rows=rows_cap)

    def steps_in(batches) -> int:
        s = n = 0
        for i, b in enumerate(batches):
            n += len(b)
            if n >= cfg.target_batch_examples or i == len(batches) - 1:
                s, n = s + 1, 0
        return s

    per_epoch = [steps_in(epoch_batches(e)) for e in range(n_epochs)]
    frac = cfg.epochs - (n_epochs - 1)
    total = max(1, sum(per_epoch[:-1]) + int(round(per_epoch[-1] * frac)))
    warmup = int(cfg.warmup_ratio * total)
    val_idx = list(range(min(len(va), cfg.eval_examples)))
    params = [p for g in groups for p in g["params"]]
    LOG.info("training %s: %s examples, %d steps of ~%d examples, %d tokens/batch, device %s",
             run_id, human_num(n_tr), total, cfg.target_batch_examples, bt, device)
    write_json(run_dir / "run.json", {"run_id": run_id, "encoder": str(encoder_spec),
                                      "train_examples": n_tr, "steps": total,
                                      "batch_tokens": bt, "config": cfg.as_dict()})

    def save_last():
        save_ckpt(last, model, opt, st)

    def do_eval() -> dict:
        rows = evaluate_rows(model, va, val_idx, bt * 2, pad, device, cfg.amp_dtype)
        m = summarize(rows)
        return {"acc": m["acc"], "nll": m["nll"], "ece": m["ece"], "brier": m["brier"],
                "by_type": {k: round(v["acc"], 4) for k, v in m["by_type"].items()}}

    model.train()
    t_win, ex_win, tok_win, t0 = time.time(), 0, 0, time.time()
    step0 = st["step"]
    loss_acc = torch.zeros((), device=device)
    n_acc = 0
    finished = st["step"] >= total or st["bad_evals"] >= cfg.early_stop_patience
    for epoch in range(st["epoch"], n_epochs):
        if finished:
            break
        batches = epoch_batches(epoch)
        start = st["pos"] if epoch == st["epoch"] else 0
        st["epoch"] = epoch
        for bi in range(start, len(batches)):
            b = tr.batch(batches[bi], pad, device)
            with amp_ctx(device, cfg.amp_dtype):
                logits = model(b["input_ids"], b["attention_mask"], b["opt_pos"], b["opt_mask"])
            lv = decision_loss(logits, b["target"], b["opt_mask"], b["type"].to(logits.device))
            lv.sum().backward()
            n_acc += len(batches[bi])
            loss_acc += lv.detach().sum()
            ex_win += len(batches[bi])
            tok_win += int(b["attention_mask"].sum())
            if n_acc < cfg.target_batch_examples and bi + 1 < len(batches):
                continue
            for p in params:
                if p.grad is not None:
                    p.grad.div_(n_acc)
            gn = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
            f = lr_factor(st["step"], total, warmup, cfg.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            opt.step()
            opt.zero_grad(set_to_none=True)
            st["step"] += 1
            st["pos"] = bi + 1
            step = st["step"]
            if step % 20 == 0 or step == 1:
                dt = time.time() - t_win
                loss = float(loss_acc) / max(1, n_acc)
                eta = (time.time() - t0) / max(1, step - step0) * (total - step)
                mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0
                LOG.info("step %d/%d  loss %.4f  lr %.2e  |g| %.2f  %.0f ex/s  %s tok/s  "
                         "mem %.1fGB  eta %s", step, total, loss, opt.param_groups[0]["lr"], gn,
                         ex_win / max(1e-6, dt), human_num(tok_win / max(1e-6, dt)), mem,
                         human_time(eta))
                append_jsonl(metrics_file, {"t": time.time(), "run_id": run_id, "step": step,
                                            "epoch": epoch, "loss": loss, "grad_norm": gn,
                                            "lr": opt.param_groups[0]["lr"],
                                            "ex_per_s": ex_win / max(1e-6, dt), "gpu_mem_gb": mem})
                t_win, ex_win, tok_win = time.time(), 0, 0
            loss_acc.zero_()
            n_acc = 0

            if step % cfg.eval_every == 0 or step == total:
                m = do_eval()
                LOG.info("  val acc %.4f  nll %.4f  ece %.4f  %s", m["acc"], m["nll"], m["ece"],
                         " ".join("%s=%.3f" % kv for kv in m["by_type"].items()))
                append_jsonl(metrics_file, dict({"t": time.time(), "run_id": run_id,
                                                 "step": step, "val": True}, **m))
                if m["nll"] < st["best"]:
                    st.update(best=m["nll"], bad_evals=0, best_metrics=m, best_step=step)
                    save_ckpt(best_path, model, None, dict(st))
                    LOG.info("  new best -> %s", best_path.name)
                else:
                    st["bad_evals"] += 1
                    if st["bad_evals"] >= cfg.early_stop_patience:
                        LOG.info("  no improvement in %d evals - stopping early", st["bad_evals"])
                        finished = True
            if step % cfg.ckpt_every == 0 or finished or step >= total:
                save_last()
            if interrupt.stopping():
                save_last()
                LOG.info("%s paused at step %d/%d - checkpoint saved; re-run to continue",
                         run_id, step, total)
                raise interrupt.Interrupted("training stopped at step %d" % step)
            if step >= total or finished:
                finished = True
                break
        else:
            st["epoch"], st["pos"] = epoch + 1, 0
            save_last()

    if not best_path.exists():
        m = do_eval()
        st.update(best=m["nll"], best_metrics=m, best_step=st["step"])
        save_ckpt(best_path, model, None, dict(st))
    save_last()
    summ = {"run_id": run_id, "steps": st["step"], "total_steps": total,
            "best_step": st.get("best_step"), "best_val": st.get("best_metrics", {}),
            "batch_tokens": bt, "train_examples": n_tr, "train_seconds": time.time() - t0,
            "best": str(best_path), "last": str(last)}
    write_json(run_dir / "summary.json", summ)
    return summ
