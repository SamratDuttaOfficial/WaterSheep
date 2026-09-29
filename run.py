#!/usr/bin/env python
"""WaterSheep: build, train, calibrate and export the decision model.

Every stage resumes where it stopped. Ctrl+C pauses safely.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import traceback

from watersheep import interrupt
from watersheep.config import Config, stage_hashes
from watersheep.paths import P, ensure_dirs, pin_caches
from watersheep.state import Manifest, Registry
from watersheep.util import (LOG, dir_size, human_bytes, human_time, now_ts, pid_alive, read_json,
                             rmtree, setup_logging)
from watersheep import stages as S

BAR = "-" * 78


def take_lock() -> bool:
    f = P.state / "run.lock"
    try:
        pid = int(f.read_text().strip())
    except Exception:
        pid = 0
    if pid and pid != os.getpid() and pid_alive(pid):
        return False
    f.write_text(str(os.getpid()))
    interrupt.on_exit(lambda: f.unlink(missing_ok=True))
    return True


def show_status(cfg, man, h) -> None:
    print(BAR)
    print("%-4s%-10s %-9s %-20s %s" % ("", "STAGE", "STATUS", "WHEN", "DETAIL"))
    print(BAR)
    for name in S.ORDER:
        e = man.entry(name)
        status = e.get("status", "pending")
        if status == "done" and e.get("hash") != h[name]:
            status = "stale"
        when = (e.get("finished") or e.get("stopped_at") or e.get("started") or "")[:19]
        meta = e.get("meta", {}) or {}
        bits = []
        for k in ("backend", "slots", "tok_per_s", "sources", "records", "accepted", "acceptance",
                  "splits", "steps", "run_id", "test", "zeroshot", "plots", "dir"):
            if k in meta:
                v = meta[k]
                if k == "splits":
                    v = "/".join(str(x) for x in v.values())
                bits.append("%s=%s" % (k, v))
        detail = " ".join(bits)[:40] or S.BY_NAME[name].desc[:40]
        mark = {"done": "[x]", "running": "[~]", "failed": "[!]", "stopped": "[>]",
                "stale": "[s]"}.get(status, "[ ]")
        print("%-4s%-10s %-9s %-20s %s" % (mark, name, status, when, detail))
    print(BAR)
    syn = read_json(P.reports / "synth.json", {}) or {}
    if syn:
        print("synthetic: %s accepted of %s tried (target %d)" % (syn.get("accepted"),
                                                                  syn.get("tried"), cfg.synth_target))
    sizes = [("data", P.data), ("hf_cache", P.hf), ("kaggle", P.kaggle), ("checkpoints", P.ckpt),
             ("out", P.out)]
    print("disk: " + "  ".join("%s=%s" % (n, human_bytes(dir_size(p))) for n, p in sizes))
    print(BAR)


def show_summary(reg) -> None:
    run = reg.active("run")
    ev = read_json(P.reports / ("eval_%s.json" % run["id"]), {}) if run else {}
    print("\n" + BAR + "\n  WATERSHEEP - RESULTS\n" + BAR)
    if not ev:
        print("  no evaluation yet")
    for split, label in (("test", "test (seen sources)"), ("zeroshot", "zero-shot (unseen sources)")):
        c, r = ev.get(split + "_calibrated"), ev.get(split + "_raw")
        if not c:
            continue
        print("  %-28s n=%-6d accuracy %.3f   ECE %.3f (raw %.3f)" % (label, c["n"], c["acc"],
                                                                     c["ece"], r["ece"]))
        for t, d in c["by_type"].items():
            print("      %-8s accuracy %.3f  ECE %.3f" % (t, d["acc"], d["ece"]))
    ex = reg.active("export")
    if ex:
        print(BAR + "\n  model: %s" % ex["path"])
        print("  use it:  python predict.py --question \"...\" --state \"...\" --options a,b,c")
    print("  graphs:  %s" % (P.plots / "dashboard.png"))
    print(BAR + "\n")


def run_stage(name, cfg, man, reg, h, force=False) -> None:
    st = S.BY_NAME[name]
    if not force:
        reason = man.stale_reason(name, h[name])
        if reason is None:
            LOG.info("[skip] %-10s already done", name)
            return
        LOG.info("[run ] %-10s (%s)", name, reason)
    else:
        LOG.info("[run ] %-10s (forced)", name)
    man.mark_running(name, h[name])
    t0 = time.time()
    try:
        meta = st.run(cfg, man, reg, h) or {}
    except KeyboardInterrupt:
        man.mark_stopped(name)
        raise
    except Exception as e:
        man.mark_failed(name, repr(e))
        LOG.error("stage %s failed: %s", name, e)
        LOG.debug("%s", traceback.format_exc())
        raise
    if interrupt.stopping():
        man.mark_stopped(name)
        raise interrupt.Interrupted(name)
    man.mark_done(name, h[name], meta if isinstance(meta, dict) else {})
    LOG.info("[done] %-10s in %s", name, human_time(time.time() - t0))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run.py", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__, epilog="""examples:
  python run.py                        run / resume everything
  python run.py --forever              keep going: more synthetic data, retrain, repeat
  python run.py --status               what is done, what is left
  python run.py --ui                   only the web UI (http://127.0.0.1:8765)
  python run.py --stage synth          run one stage only
  python run.py --from train           this stage onward
  python run.py --set synth_target=40000     change a knob (saved in config.json)
  python run.py --list-sources         every public dataset and its state
  python run.py --clean synth          delete one stage's output
  python run.py --reset                wipe everything and start over
""")
    g = p.add_argument_group("what to run")
    g.add_argument("--stage", action="append", metavar="NAME", help="run only this stage (repeatable)")
    g.add_argument("--from", dest="from_", metavar="NAME", help="start at this stage")
    g.add_argument("--until", metavar="NAME", help="stop after this stage")
    g.add_argument("--force", action="store_true", help="re-run even if marked done")
    g.add_argument("--forever", action="store_true",
                   help="after finishing, raise synth_target by round_increment and go again")
    g.add_argument("--skip-install", action="store_true", help="never pip-install anything")
    g.add_argument("--ui", action="store_true",
                   help="only open the web UI (status, datasets, corpus editor); runs nothing")
    g.add_argument("--no-ui", action="store_true", help="run without starting the web UI")
    g = p.add_argument_group("inspect")
    g.add_argument("--status", action="store_true")
    g.add_argument("--summary", action="store_true")
    g.add_argument("--list", action="store_true", help="all trained runs and exports")
    g.add_argument("--list-sources", action="store_true")
    g.add_argument("--show-config", action="store_true")
    g = p.add_argument_group("manage")
    g.add_argument("--set", action="append", metavar="KEY=VAL", help="change a config value (saved)")
    g.add_argument("--promote", metavar="RUN_ID", help="make this trained run the active one")
    g.add_argument("--clean", action="append", metavar="STAGE", help="delete a stage's outputs")
    g.add_argument("--reset", action="store_true", help="delete ALL data, checkpoints and state")
    g.add_argument("--keep-cache", action="store_true", help="with --reset, keep downloads")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    interrupt.install()
    ensure_dirs()
    pin_caches()
    setup_logging(P.logs / ("run_%s.log" % now_ts()), args.verbose)
    cfg = Config.load()
    if args.set:
        LOG.info("config updated: %s", ", ".join(cfg.apply(args.set)))
    if args.skip_install:
        cfg.auto_install = False
    cfg.save()
    man, reg = Manifest(), Registry()
    reg.prune_missing()
    h = stage_hashes(cfg, S.ORDER)

    if args.show_config:
        import json
        print(json.dumps(cfg.as_dict(), indent=2))
        return 0
    if args.status:
        show_status(cfg, man, h)
        return 0
    if args.summary:
        show_summary(reg)
        return 0
    if args.list:
        for kind in ("run", "export"):
            print("\n%s:" % kind.upper())
            for i in reg.by_kind(kind):
                print(" %s %-32s %s  %s" % ("*" if i.get("active") else " ", i["id"],
                                            i["created"][:19], i.get("metrics", {})))
        return 0
    if args.list_sources:
        from watersheep import sources as SRC
        from watersheep.data import raw_file
        rep = read_json(P.reports / "sources.json", {}) or {}
        on = {s.name for s in SRC.enabled(cfg)}
        for s in list(SRC.REGISTRY.values()):
            m = read_json(raw_file(s.name).with_suffix(".done"), {}) or {}
            state = ("%d rows" % m["n"]) if m else ("FAILED: " + rep.get("failed", {})[s.name][:50]
                                                   if s.name in rep.get("failed", {}) else "-")
            print("%s %-26s %-7s %-44s %s" % ("+" if s.name in on else " ", s.name, s.type,
                                              (s.hf or s.kaggle)[:44], state))
        return 0
    if args.set and not any([args.stage, args.from_, args.until, args.force, args.forever]):
        show_status(cfg, man, h)
        LOG.info("config saved. run `python run.py` to continue.")
        return 0
    if args.promote:
        it = reg.resolve(args.promote)
        if not it:
            LOG.error("no such run: %s (see --list)", args.promote)
            return 2
        reg.promote(it["id"])
        return 0
    if args.reset:
        LOG.warning("resetting project")
        for d in (P.data, P.ckpt, P.state, P.out, P.metrics):
            rmtree(d)
        if not args.keep_cache:
            rmtree(P.hf)
            rmtree(P.kaggle)
        ensure_dirs()
        LOG.info("reset complete")
        return 0
    if args.clean:
        targets = S.ORDER if "all" in args.clean else args.clean
        for name in targets:
            if name not in S.BY_NAME:
                LOG.error("unknown stage %r (choices: %s)", name, ", ".join(S.ORDER))
                return 2
            S.BY_NAME[name].clean(cfg)
            man.clear(name)
        LOG.info("cleaned: %s", ", ".join(targets))
        return 0

    if args.ui:
        from watersheep import webui
        if not webui.start(cfg.webui_port, cfg.webui_open_browser):
            return 1
        LOG.info("web UI only - the pipeline is not started. Ctrl+C closes it.")
        while not interrupt.stopping():
            time.sleep(0.5)
        return 0

    if not take_lock():
        LOG.error("another WaterSheep run is already active in this folder - stop it first "
                  "(to just look at it, use --ui)")
        return 3
    ui = None
    if cfg.webui and not args.no_ui:
        from watersheep import webui
        ui = webui.start(cfg.webui_port, cfg.webui_open_browser)

    todo = list(S.ORDER)
    if args.stage:
        bad = [n for n in args.stage if n not in S.BY_NAME]
        if bad:
            LOG.error("unknown stage %s (choices: %s)", bad, ", ".join(S.ORDER))
            return 2
        todo = [n for n in S.ORDER if n in args.stage]
    else:
        if args.from_:
            todo = todo[S.ORDER.index(args.from_):]
        if args.until:
            todo = [n for n in todo if S.ORDER.index(n) <= S.ORDER.index(args.until)]

    LOG.info("WaterSheep - Ctrl+C pauses safely (progress is saved); run again to resume")
    t0 = time.time()
    rounds = 0
    try:
        while True:
            h = stage_hashes(cfg, S.ORDER)
            pending = [n for n in todo if args.force or man.stale_reason(n, h[n]) is not None]
            if pending:
                LOG.info("plan: %s", " -> ".join(pending))
            for name in todo:
                interrupt.check()
                run_stage(name, cfg, man, reg, h, force=args.force and rounds == 0)
            if not args.forever:
                break
            rounds += 1
            cfg.synth_target += cfg.round_increment
            cfg.save()
            LOG.info("=== round %d finished - raising synth_target to %d and continuing ===",
                     rounds, cfg.synth_target)
            todo = S.ORDER[S.ORDER.index("synth"):]
    except KeyboardInterrupt:
        LOG.info("paused after %s. everything is saved - run again to continue",
                 human_time(time.time() - t0))
        show_status(cfg, man, stage_hashes(cfg, S.ORDER))
        return 130
    except Exception:
        LOG.error("halted after %s - fix the error above and re-run; finished work is kept "
                  "(log: %s)", human_time(time.time() - t0), P.logs)
        return 1
    finally:
        try:
            from watersheep import llm
            llm.shutdown()
        except Exception:
            pass
    LOG.info("pipeline complete in %s", human_time(time.time() - t0))
    show_status(cfg, man, stage_hashes(cfg, S.ORDER))
    show_summary(reg)
    if ui:
        LOG.info("the web UI closes with this window; reopen it any time with: run.bat --ui")
    return 0


if __name__ == "__main__":
    sys.exit(main())
