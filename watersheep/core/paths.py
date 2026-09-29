"""Project paths. WATERSHEEP_HOME overrides the root."""
from __future__ import annotations
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("WATERSHEEP_HOME") or CODE_ROOT).resolve()


class P:
    root      = ROOT
    data      = ROOT / "data"
    raw       = ROOT / "data" / "raw"
    synth     = ROOT / "data" / "synth"
    build     = ROOT / "data" / "build"
    tok       = ROOT / "data" / "tok"
    hf        = ROOT / "hf_cache"
    kaggle    = ROOT / "kaggle_cache"
    downloads = ROOT / "downloads"
    ollama_bin = ROOT / "ollama_bin"
    ckpt      = ROOT / "checkpoints"
    state     = ROOT / "state"
    logs      = ROOT / "logs"
    metrics   = ROOT / "logs" / "metrics"
    out       = ROOT / "out"
    plots     = ROOT / "out" / "plots"
    reports   = ROOT / "out" / "reports"
    export    = ROOT / "out" / "export"

    config    = ROOT / "config.json"
    manifest  = ROOT / "state" / "manifest.json"
    registry  = ROOT / "state" / "registry.json"

    ALL = (data, raw, synth, build, tok, hf, kaggle, downloads, ckpt, state,
           logs, metrics, out, plots, reports, export)


def ensure_dirs() -> None:
    for d in P.ALL:
        d.mkdir(parents=True, exist_ok=True)


def pin_caches() -> None:
    """Keep third-party caches inside the project."""
    P.hf.mkdir(parents=True, exist_ok=True)
    for k in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
              "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
        os.environ[k] = str(P.hf)
    os.environ["KAGGLEHUB_CACHE"] = str(P.kaggle)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
