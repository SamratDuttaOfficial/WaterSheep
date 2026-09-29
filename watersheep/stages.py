"""Pipeline stages. Each stage is resumable and safe to re-run."""
from __future__ import annotations
import collections
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from . import gpu, interrupt, llm
from .paths import P, ensure_dirs, pin_caches
from .schema import answers
from .state import Manifest, Registry
from .util import (LOG, append_jsonl, human_num, human_time, iter_jsonl,
                   now_ts, read_json, rmtree, sha1, write_json, write_jsonl)


@dataclass
class Stage:
    name: str
    desc: str
    run: Callable
    clean: Callable


TINY_ENCODER = {"vocab_size": 260, "hidden_size": 64, "intermediate_size": 128,
                "num_hidden_layers": 2, "num_attention_heads": 2, "max_position_embeddings": 1024,
                "pad_token_id": 3, "cls_token_id": 0, "sep_token_id": 1, "bos_token_id": 0,
                "eos_token_id": 1}


def encoder_spec(cfg):
    return TINY_ENCODER if cfg.encoder == "tiny-test" else cfg.encoder


def load_tok(cfg):
    from .model import CharTok, Tok
    return CharTok() if cfg.encoder == "tiny-test" else Tok.load(cfg.encoder)


def _pip(*args) -> int:
    return subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                           *args], check=False).returncode


def run_env(cfg, man: Manifest, reg: Registry, h: dict) -> dict:
    ensure_dirs()
    pin_caches()
    need = [("torch", "torch"), ("numpy", "numpy"), ("transformers", "transformers>=4.48"),
            ("datasets", "datasets>=3.0"), ("pandas", "pandas"), ("pyarrow", "pyarrow"),
            ("safetensors", "safetensors"), ("matplotlib", "matplotlib")]
    if cfg.use_kaggle:
        need.append(("kagglehub", "kagglehub"))
    missing = []
    for mod, pkg in need:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing and cfg.auto_install:
        LOG.info("installing missing packages: %s", ", ".join(missing))
        if "torch" in missing:
            idx = "https://download.pytorch.org/whl/cu128" if gpu.has_nvidia() else \
                "https://download.pytorch.org/whl/cpu"
            LOG.info("installing torch from %s (large, one time)", idx)
            _pip("torch", "--index-url", idx)
            missing.remove("torch")
        if missing:
            _pip(*missing)
        interrupt.check("install interrupted - re-run to continue")
    elif missing:
        raise RuntimeError("missing packages: %s (pip install -r requirements.txt)" % missing)

    import torch
    if not torch.cuda.is_available() and gpu.has_nvidia() and cfg.device.startswith("cuda"):
        if cfg.auto_install:
            LOG.warning("torch %s is CPU-only but this machine has an NVIDIA GPU - installing "
                        "the CUDA build (one time)", torch.__version__)
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch"], check=False)
            _pip("torch", "--index-url", "https://download.pytorch.org/whl/cu128")
            raise RuntimeError("torch was reinstalled with CUDA support - run again to "
                               "continue (nothing was lost)")
        raise RuntimeError("CPU-only torch on a CUDA machine; install the CUDA build or "
                           "--set device=cpu")
    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, vram_gb=round(p.total_memory / 1e9, 2),
                    bf16=torch.cuda.is_bf16_supported())
        LOG.info("GPU: %s, %.1f GB, bf16=%s", p.name, p.total_memory / 1e9, info["bf16"])
    else:
        LOG.warning("no CUDA device - training will run on the CPU (very slow)")
    write_json(P.state / "env.json", info)
    return info


def clean_env(cfg) -> None:
    (P.state / "env.json").unlink(missing_ok=True)


def run_teacher(cfg, man, reg, h) -> dict:
    from .pool import pool_map
    from .synth import make_example
    from .taxonomy import job
    t = llm.get_teacher(cfg)
    n = max(4, t.b.slots)
    LOG.info("teacher warm-up: %d examples in parallel to measure throughput", n)
    tok0, t0 = t.tokens, time.time()
    results = [r for r in pool_map(lambda j: make_example(t, j, cfg),
                                   [job(999999, k, cfg.seed) for k in range(n)], t.workers) if r]
    interrupt.check()
    dt = time.time() - t0
    ok = sum(1 for r in results if r.get("status") == "accepted")
    meta = {"backend": t.b.name, "slots": t.b.slots, "tok_per_s": round((t.tokens - tok0) / dt, 1),
            "warmup_accepted": "%d/%d" % (ok, len(results))}
    LOG.info("teacher ready: %s, %d slots, %.0f tok/s, warm-up accepted %s", meta["backend"],
             meta["slots"], meta["tok_per_s"], meta["warmup_accepted"])
    return meta


def clean_teacher(cfg) -> None:
    (P.state / "llm_slots.json").unlink(missing_ok=True)


def _source_key(cfg, name: str) -> str:
    from .sources import REGISTRY
    key = {"v": 1, "name": name, "max": cfg.per_source_max, "s": cfg.max_state_chars,
           "o": cfg.max_option_chars, "k": cfg.max_options, "seed": cfg.seed}
    ver = getattr(REGISTRY.get(name), "version", 1)
    if ver != 1:
        key["ver"] = ver
    return sha1(key)


def run_sources(cfg, man, reg, h) -> dict:
    from . import sources as S
    from .data import raw_file
    pin_caches()
    counts, failed = {}, {}
    todo = S.enabled(cfg)
    LOG.info("%d public sources enabled", len(todo))
    for k, src in enumerate(todo, 1):
        interrupt.check("sources stopped - finished sources are kept")
        f = raw_file(src.name)
        marker = read_json(f.with_suffix(".done"), {}) or {}
        key = _source_key(cfg, src.name)
        if marker.get("key") == key and f.exists():
            counts[src.name] = marker.get("n", 0)
            continue
        LOG.info("[%d/%d] %s  (%s)", k, len(todo), src.name, src.hf or src.kaggle or "built-in")
        t0 = time.time()
        try:
            recs = S.load(src, cfg, cfg.per_source_max, cfg.seed)
        except interrupt.Interrupted:
            raise
        except Exception as e:
            msg = str(e).splitlines()[0][:200] if str(e) else repr(e)
            LOG.warning("  skipped %s: %s", src.name, msg)
            failed[src.name] = msg
            continue
        if not recs:
            LOG.warning("  skipped %s: no usable rows", src.name)
            failed[src.name] = "no usable rows"
            continue
        write_jsonl(f, recs)
        write_json(f.with_suffix(".done"), {"key": key, "n": len(recs), "when": now_ts()})
        counts[src.name] = len(recs)
        by = collections.Counter("+".join(r["options"][a] for a in answers(r)) or "(none)"
                                 for r in recs)
        LOG.info("  %d records in %.0fs  (answers: %s)", len(recs), time.time() - t0,
                 ", ".join("%s=%d" % kv for kv in by.most_common(4)))
    write_json(P.reports / "sources.json", {"counts": counts, "failed": failed})
    if not counts:
        raise RuntimeError("no public source could be loaded - check the network")
    LOG.info("sources: %d loaded (%s records), %d skipped", len(counts),
             human_num(sum(counts.values())), len(failed))
    return {"sources": len(counts), "records": sum(counts.values()), "failed": sorted(failed)}


def clean_sources(cfg) -> None:
    rmtree(P.raw)


def run_audit(cfg, man, reg, h) -> dict:
    from .data import raw_file
    from .pool import pool_map
    from .sources import enabled
    from .synth import audit_one
    names = [s.name for s in enabled(cfg) if raw_file(s.name).exists()]
    if cfg.audit_per_source <= 0 or not names:
        write_json(P.reports / "audit.json", {"per_source": {}, "dropped": []})
        return {"audited": 0}
    rows_f = P.state / "audit_rows.jsonl"
    done = {r["id"]: r for r in iter_jsonl(rows_f)}
    items = []
    for name in names:
        recs = list(iter_jsonl(raw_file(name)))
        random.Random(cfg.seed).shuffle(recs)
        items += recs[:cfg.audit_per_source]
    todo = [r for r in items if r["id"] not in done]
    if todo:
        t = llm.get_teacher(cfg)
        LOG.info("audit: teacher answers %d public examples blind (%d already done)",
                 len(todo), len(items) - len(todo))
        last = time.time()
        for k, res in enumerate(pool_map(lambda r: audit_one(t, r, cfg.seed), todo, t.workers), 1):
            if res:
                append_jsonl(rows_f, res)
                done[res["id"]] = res
            if time.time() - last > 30:
                last = time.time()
                LOG.info("  audit %d/%d", k, len(todo))
        interrupt.check("audit stopped - answered items are kept")
    wanted = {r["id"] for r in items}
    per = collections.defaultdict(list)
    for r in done.values():
        if r["id"] in wanted:
            per[r["source"]].append(r)
    table = {}
    for s, rs in per.items():
        table[s] = {"n": len(rs), "agree": sum(r["agree"] for r in rs) / len(rs),
                    "mean_p": sum(r["p"] for r in rs) / len(rs),
                    "chance": sum(r["chance"] for r in rs) / len(rs)}
    dropped = sorted(s for s, d in table.items()
                     if cfg.audit_drop_below_chance and d["n"] >= 20
                     and d["agree"] < d["chance"] - 0.15)
    write_json(P.reports / "audit.json", {"per_source": table, "dropped": dropped})
    for s in sorted(table, key=lambda x: table[x]["agree"]):
        d = table[s]
        LOG.info("  %-26s agree %.2f  (chance %.2f, n=%d)%s", s, d["agree"], d["chance"], d["n"],
                 "  -> DROPPED (looks mislabelled)" if s in dropped else "")
    return {"audited": len(wanted), "dropped": dropped}


def clean_audit(cfg) -> None:
    (P.state / "audit_rows.jsonl").unlink(missing_ok=True)
    (P.reports / "audit.json").unlink(missing_ok=True)


def _shard(i: int, ext: str) -> Path:
    return P.synth / ("shard_%05d.%s" % (i, ext))


def _salvage_multi(cfg) -> None:
    """Relabel confident multi-label rejections in finished shards (once)."""
    from .synth import relabel_multi, soft_target
    mark = P.state / ("salvaged_multi_%s.json" % str(cfg.verify_relabel_conf).replace(".", ""))
    if mark.exists():
        return
    n = 0
    for f in sorted(P.synth.glob("shard_*.jsonl")):
        rows, changed = list(iter_jsonl(f)), False
        for row in rows:
            if row.get("type") != "multi" or row.get("reason") != "teacher_disagrees" or \
                    not row.get("record") or not row.get("teacher"):
                continue
            rec = relabel_multi(row["record"], row["teacher"], cfg.verify_relabel_conf)
            if rec is None:
                continue
            rec["target"] = soft_target(rec, row["teacher"], cfg.soft_label_mix)
            row.update(status="accepted", reason="ok", record=rec, salvaged=True)
            changed, n = True, n + 1
        if changed:
            write_jsonl(f, rows)
    write_json(mark, {"salvaged": n, "when": now_ts()})
    if n:
        LOG.info("synthetic: %d multi-label examples the teacher was sure about were relabelled "
                 "and accepted", n)


def run_synth(cfg, man, reg, h) -> dict:
    from .pool import pool_map
    from .synth import Deduper, make_example
    from .taxonomy import jobs as make_jobs
    P.synth.mkdir(parents=True, exist_ok=True)
    _salvage_multi(cfg)
    dedup = Deduper(cfg.verify_dedup_bits)
    reasons: collections.Counter = collections.Counter()
    accepted = total = 0

    def count(row):
        nonlocal accepted, total
        total += 1
        reasons[row.get("reason", "?")] += 1
        if row.get("status") == "accepted":
            accepted += 1
            dedup.add(row["record"])

    for d in sorted(P.synth.glob("shard_*.done")):
        for row in iter_jsonl(d.with_suffix(".jsonl")):
            count(row)
    if accepted >= cfg.synth_target:
        LOG.info("synthetic: %d accepted already (target %d)", accepted, cfg.synth_target)
        return _synth_meta(accepted, total, reasons)

    preloaded = {}
    for part in sorted(P.synth.glob("shard_*.part")):
        s = int(part.stem.split("_")[1])
        preloaded[s] = {row["jid"]: row for row in iter_jsonl(part)}
        for row in preloaded[s].values():
            count(row)

    t = llm.get_teacher(cfg)
    LOG.info("synthetic: %d/%d accepted so far; streaming jobs to %d parallel workers",
             accepted, cfg.synth_target, t.workers)
    shards: dict = {}
    retried: set = set()
    skip: set = set()
    errors = 0

    def close(s: int) -> None:
        sh = shards.pop(s)
        missing = [j for j in sh["jobs"] if j["jid"] not in sh["have"]]
        if len(missing) <= 0.03 * len(sh["jobs"]):
            write_jsonl(_shard(s, "jsonl"), [sh["have"][j["jid"]] for j in sh["jobs"]
                                              if j["jid"] in sh["have"]])
            _shard(s, "done").write_text(now_ts(), encoding="utf-8")
            _shard(s, "part").unlink(missing_ok=True)
            write_json(P.reports / "synth.json", _synth_meta(accepted, total, reasons))
            if s % 10 == 0:
                LOG.info("  shard %d closed; outcomes so far: %s", s,
                         ", ".join("%s=%d" % kv for kv in reasons.most_common(7)))
        elif s not in retried:
            retried.add(s)
            preloaded[s] = sh["have"]
        else:
            skip.add(s)
            LOG.warning("  shard %d left open: %d jobs failed twice - they retry on the next run",
                        s, len(missing))

    def feeder():
        """Stream jobs across shards until the target is reached."""
        s = 0
        while s < cfg.synth_max_shards:
            if _shard(s, "done").exists() or s in skip:
                s += 1
                continue
            rate = accepted / total if total >= 20 else 0.5
            in_flight = sum(sh["left"] for sh in list(shards.values()))
            if accepted + rate * in_flight >= cfg.synth_target:
                return
            jobs = make_jobs(s, cfg.synth_shard, cfg.seed, cfg.synth_multi_share)
            have = preloaded.pop(s, {})
            todo = [j for j in jobs if j["jid"] not in have]
            shards[s] = {"jobs": jobs, "have": dict(have), "left": len(todo)}
            for j in todo:
                yield dict(j, shard=s)
            s += 1

    def work(j: dict) -> dict:
        try:
            res = make_example(t, j, cfg)
        except interrupt.Interrupted:
            raise
        except Exception as e:
            res = {"jid": j["jid"], "status": "error", "reason": "error:%s" % type(e).__name__}
        res["shard"] = j["shard"]
        return res

    last_log, tok_last = time.time(), t.tokens
    t_start, acc_start = time.time(), accepted
    while True:
        got = 0
        for res in pool_map(work, feeder(), t.workers):
            if res is None:
                continue
            got += 1
            s = res.pop("shard")
            sh = shards.get(s)
            if sh is None:
                continue
            sh["left"] -= 1
            if res["status"] == "error":
                errors += 1
            else:
                if res["status"] == "accepted" and dedup.seen(res["record"]):
                    res.update(status="rejected", reason="duplicate")
                append_jsonl(_shard(s, "part"), res)
                sh["have"][res["jid"]] = res
                count(res)
            if sh["left"] <= 0:
                close(s)
            if time.time() - last_log > 30:
                dt = time.time() - last_log
                tok_s = (t.tokens - tok_last) / dt
                last_log, tok_last = time.time(), t.tokens
                rate_h = (accepted - acc_start) / max(1e-6, time.time() - t_start) * 3600
                g = gpu.status()
                LOG.info("  synth %d/%d accepted (%.0f%% of %d tried)  %.0f accepted/h  "
                         "%.0f tok/s  GPU %s%%  eta %s", accepted, cfg.synth_target,
                         100.0 * accepted / max(1, total), total, rate_h, tok_s,
                         g["util"] if g else "?",
                         human_time((cfg.synth_target - accepted) / max(1e-6, rate_h) * 3600))
                append_jsonl(P.logs / "synth.jsonl", {"t": time.time(), "accepted": accepted,
                                                      "tried": total, "tok_s": tok_s,
                                                      "gpu_util": g["util"] if g else None})
        for s in [s for s, sh in shards.items() if sh["left"] <= 0]:
            close(s)
        for s in list(shards):
            preloaded[s] = shards.pop(s)["have"]
        if interrupt.stopping() or accepted >= cfg.synth_target or got == 0:
            break
    if errors:
        LOG.warning("  %d generation calls failed (they are retried on the next run)", errors)
    write_json(P.reports / "synth.json", _synth_meta(accepted, total, reasons))
    interrupt.check("synthesis stopped - every finished example is saved")
    if accepted < cfg.synth_target:
        LOG.warning("synthetic: stopped at %d accepted (synth_max_shards reached)", accepted)
    return _synth_meta(accepted, total, reasons)


def _synth_meta(accepted, total, reasons) -> dict:
    return {"accepted": accepted, "tried": total,
            "acceptance": round(accepted / max(1, total), 3), "reasons": dict(reasons)}


def clean_synth(cfg) -> None:
    rmtree(P.synth)
    (P.logs / "synth.jsonl").unlink(missing_ok=True)
    (P.reports / "synth.json").unlink(missing_ok=True)


def run_describe(cfg, man, reg, h) -> dict:
    from . import describe
    from .data import raw_file, synth_records
    from .sources import enabled
    names = [s.name for s in enabled(cfg) if raw_file(s.name).exists()]
    return describe.run(cfg, llm.get_teacher(cfg), names, synth_records())


def clean_describe(cfg) -> None:
    (P.state / "describe.json").unlink(missing_ok=True)
    (P.state / "describe_rows.jsonl").unlink(missing_ok=True)


def run_build(cfg, man, reg, h) -> dict:
    from .data import build, raw_file
    from .sources import enabled
    names = [s.name for s in enabled(cfg) if raw_file(s.name).exists()]
    audit = read_json(P.reports / "audit.json", {}) or {}
    dropped = audit.get("dropped", []) if cfg.audit_drop_below_chance else []
    stats = build(cfg, names, dropped)
    if stats["splits"]["train"] == 0:
        raise RuntimeError("the training split is empty - nothing to train on")
    return {"splits": stats["splits"], "synthetic_train": stats["synthetic_train"]}


def clean_build(cfg) -> None:
    rmtree(P.build)


def _tok_dir(cfg, h: dict) -> Path:
    return P.tok / sha1({"enc": cfg.encoder, "L": cfg.max_len, "q": cfg.max_question_tokens,
                         "o": cfg.max_option_tokens, "build": h.get("build")})


def prepare_packed(cfg, h: dict):
    from .data import SPLITS, pack
    tok = load_tok(cfg)
    d = _tok_dir(cfg, h)
    stats = read_json(P.build / "stats.json", {}) or {}
    sources = sorted((stats.get("per_source") or {}).keys())
    for s in SPLITS:
        interrupt.check()
        pack(tok, s, cfg, d, sources)
    return tok, d


def run_train(cfg, man, reg, h) -> dict:
    from .trainer import train
    llm.shutdown()
    pin_caches()
    tok, tok_dir = prepare_packed(cfg, h)
    prog = man.progress("train")
    rid = prog.get("run_id")
    if not rid or prog.get("sig") != h["train"] or not (P.ckpt / rid).exists():
        rid = "run_%s" % now_ts()
        man.set_progress("train", "run_id", rid)
        man.set_progress("train", "sig", h["train"])
        man.set_progress("train", "batch_tokens", 0)
    rdir = P.ckpt / rid
    summ = train(cfg=cfg, run_dir=rdir, run_id=rid, tok_dir=tok_dir, pad=tok.pad,
                 encoder_spec=encoder_spec(cfg), metrics_file=P.metrics / (rid + ".jsonl"),
                 saved_budget=int(prog.get("batch_tokens") or 0))
    man.set_progress("train", "batch_tokens", summ["batch_tokens"])
    reg.add(rid, "run", rdir / "best.pt", metrics=summ.get("best_val") or {},
            note="tok=%s" % tok_dir.name)
    reg.update(rid, tok_dir=str(tok_dir))
    reg.promote(rid)
    return {"run_id": rid, "steps": summ["steps"], "val": summ.get("best_val")}


def clean_train(cfg) -> None:
    rmtree(P.ckpt)
    rmtree(P.tok)


def _active_model(cfg, reg: Registry, h: dict):
    import torch
    from .data import Packed
    from .trainer import build_model, load_ckpt, pick_device
    run = reg.active("run")
    if not run:
        raise RuntimeError("no trained run - run the train stage first")
    device = pick_device(cfg.device)
    model = build_model(cfg, encoder_spec(cfg))
    model.load_state_dict(load_ckpt(Path(run["path"]))["model"])
    model.to(device).eval()
    tok_dir = Path(run.get("tok_dir") or _tok_dir(cfg, h))
    tok = load_tok(cfg)
    budget = int(read_json(P.manifest, {}).get("progress::train", {}).get("batch_tokens") or 4096)
    return run, model, tok, tok_dir, device, budget * 2, Packed


def run_calibrate(cfg, man, reg, h) -> dict:
    from .metrics import fit_temperature, fit_temperature_multi
    from .model import MULTI, TYPE_NAME
    from .trainer import predict
    llm.shutdown()
    run, model, tok, tok_dir, device, bt, Packed = _active_model(cfg, reg, h)
    va = Packed(tok_dir / "val.npz")
    idx = list(range(len(va)))
    logits = predict(model, va, idx, bt, tok.pad, device, cfg.amp_dtype)
    temps = {}
    for tid, name in TYPE_NAME.items():
        sel = [k for k, i in enumerate(idx) if int(va.typ[i]) == tid]
        if len(sel) >= 30 and tid == MULTI:
            truth = [va.tgt[va.poff[idx[k]]:va.poff[idx[k] + 1]] > 0.5 for k in sel]
            temps[name] = round(fit_temperature_multi([logits[k] for k in sel], truth), 4)
        elif len(sel) >= 30:
            temps[name] = round(fit_temperature([logits[k] for k in sel],
                                                [int(va.ans[idx[k]]) for k in sel]), 4)
        else:
            temps[name] = 1.0
    thr = 0.5
    sel = [k for k, i in enumerate(idx) if int(va.typ[i]) == MULTI]
    if len(sel) >= 30:
        from .metrics import sigmoid
        ps = [sigmoid(logits[k], temps.get("multi", 1.0)) for k in sel]
        ys = [va.tgt[va.poff[idx[k]]:va.poff[idx[k] + 1]] > 0.5 for k in sel]

        def f1(th):
            tp = sum(int(((p >= th) & y).sum()) for p, y in zip(ps, ys))
            fp = sum(int(((p >= th) & ~y).sum()) for p, y in zip(ps, ys))
            fn = sum(int(((p < th) & y).sum()) for p, y in zip(ps, ys))
            return 2 * tp / max(1, 2 * tp + fp + fn)
        thr = max([round(0.2 + 0.05 * k, 2) for k in range(9)], key=f1)
        LOG.info("multi-label threshold %.2f (F1 %.3f; at 0.5: %.3f)", thr, f1(thr), f1(0.5))
    write_json(P.reports / ("calibration_%s.json" % run["id"]), {"run_id": run["id"], "temperatures": temps,
                                                                  "multi_threshold": thr})
    LOG.info("temperatures: %s", ", ".join("%s=%.3f" % kv for kv in temps.items()))
    return {"run_id": run["id"], "temperatures": temps}


def clean_calibrate(cfg) -> None:
    for f in P.reports.glob("calibration_*.json"):
        f.unlink(missing_ok=True)


def run_evaluate(cfg, man, reg, h) -> dict:
    from .metrics import summarize
    from .trainer import evaluate_rows
    llm.shutdown()
    run, model, tok, tok_dir, device, bt, Packed = _active_model(cfg, reg, h)
    temps = (read_json(P.reports / ("calibration_%s.json" % run["id"]), {}) or {}).get("temperatures", {})
    out = {"run_id": run["id"], "when": now_ts(), "temperatures": temps}
    for split in ("test", "zeroshot"):
        ds = Packed(tok_dir / (split + ".npz"))
        if len(ds) == 0:
            continue
        idx = list(range(len(ds)))
        out[split + "_raw"] = summarize(evaluate_rows(model, ds, idx, bt, tok.pad, device, cfg.amp_dtype))
        out[split + "_calibrated"] = summarize(evaluate_rows(model, ds, idx, bt, tok.pad, device,
                                                             cfg.amp_dtype, temps))
    write_json(P.reports / ("eval_%s.json" % run["id"]), out)
    head = {}
    for split in ("test", "zeroshot"):
        c = out.get(split + "_calibrated")
        r = out.get(split + "_raw")
        if not c:
            continue
        LOG.info("%-9s n=%-6d acc %.4f  ECE %.4f -> %.4f calibrated  NLL %.4f", split, c["n"],
                 c["acc"], r["ece"], c["ece"], c["nll"])
        for t, d in c["by_type"].items():
            LOG.info("   %-7s acc %.4f  ECE %.4f%s", t, d["acc"], d["ece"],
                     ("  MAE %.3f" % d["mae"]) if "mae" in d else
                     ("  per-option acc %.3f  F1 %.3f" % (d["option_acc"], d["f1"])) if "f1" in d else "")
        head[split] = {"acc": round(c["acc"], 4), "ece": round(c["ece"], 4)}
    if out.get("zeroshot_calibrated"):
        for s, d in out["zeroshot_calibrated"]["by_source"].items():
            LOG.info("   zero-shot %-22s acc %.3f (chance %.3f, n=%d)", s, d["acc"], d["chance"], d["n"])
    reg.update(run["id"], metrics=dict(run.get("metrics") or {}, **{"test": head.get("test"),
                                                                    "zeroshot": head.get("zeroshot")}))
    return {"run_id": run["id"], **head}


def clean_evaluate(cfg) -> None:
    for f in P.reports.glob("eval_*.json"):
        f.unlink(missing_ok=True)


def run_plots(cfg, man, reg, h) -> dict:
    from . import plots
    run = reg.active("run")
    files = plots.make(run["id"] if run else "")
    return {"plots": len(files), "dir": str(P.plots)}


def clean_plots(cfg) -> None:
    rmtree(P.plots)


def run_export(cfg, man, reg, h) -> dict:
    from .infer import WaterSheep, export
    run = reg.active("run")
    if not run:
        raise RuntimeError("nothing to export")
    cal = read_json(P.reports / ("calibration_%s.json" % run["id"]), {}) or {}
    temps = cal.get("temperatures", {})
    ev = read_json(P.reports / ("eval_%s.json" % run["id"]), {}) or {}
    metrics = {k: {"acc": (ev.get(k) or {}).get("acc"), "ece": (ev.get(k) or {}).get("ece")}
               for k in ("test_calibrated", "zeroshot_calibrated")}
    d = export(cfg, Path(run["path"]).parent, load_tok(cfg), encoder_spec(cfg), temps, metrics,
               run["id"], multi_threshold=cal.get("multi_threshold", 0.5))
    ws = WaterSheep.load(d, device="cpu")
    probe = ws.decide("The package arrived with a cracked screen and the customer wants a refund.",
                      "Which team should handle this?", ["billing", "shipping", "technical support"])
    LOG.info("export OK: %s  (probe answer: %s, %.2f)", d, probe.get("answer"), probe.get("confidence", 0))
    reg.add(d.name, "export", d, parent=run["id"], metrics=metrics)
    return {"dir": str(d)}


def clean_export(cfg) -> None:
    rmtree(P.export)


STAGES: List[Stage] = [
    Stage("env", "packages, CUDA torch, GPU check", run_env, clean_env),
    Stage("teacher", "Ollama + Qwen weights + llama-server (all GPU slots)", run_teacher, clean_teacher),
    Stage("sources", "download + map public datasets", run_sources, clean_sources),
    Stage("audit", "teacher spot-checks every public source", run_audit, clean_audit),
    Stage("synth", "generate + verify synthetic decisions", run_synth, clean_synth),
    Stage("describe", "teacher writes label descriptions, scale levels, yes/no criteria",
          run_describe, clean_describe),
    Stage("build", "merge, dedup, split, add label descriptions", run_build, clean_build),
    Stage("train", "train the ModernBERT decision model", run_train, clean_train),
    Stage("calibrate", "fit per-type temperatures", run_calibrate, clean_calibrate),
    Stage("evaluate", "accuracy + calibration, zero-shot sources", run_evaluate, clean_evaluate),
    Stage("plots", "render graphs", run_plots, clean_plots),
    Stage("export", "safetensors + tokenizer + config", run_export, clean_export),
]
BY_NAME = {s.name: s for s in STAGES}
ORDER = [s.name for s in STAGES]
