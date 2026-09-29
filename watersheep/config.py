"""All settings, stored in config.json."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List

from .paths import P
from .util import read_json, sha1, write_json


@dataclass
class Config:
    llm_model: str = "qwen3.5:4b"
    llm_backend: str = "auto"
    llm_ctx_per_slot: int = 2048
    llm_slots: int = 0  # 0 = auto
    llm_max_slots: int = 32
    llm_vram_margin_mb: int = 450
    llm_port: int = 11511
    llm_timeout_s: int = 900
    free_ollama_vram: bool = True

    webui: bool = True
    webui_port: int = 8765
    webui_open_browser: bool = True

    sources: List[str] = field(default_factory=list)  # empty = all
    exclude_sources: List[str] = field(default_factory=list)
    use_kaggle: bool = True
    per_source_max: int = 5000
    heldout_sources: List[str] = field(default_factory=lambda: [
        "kg:sarcasm", "kg:news_category", "hf:qasc", "kg:phone_reviews"])
    max_state_chars: int = 3000
    max_option_chars: int = 160
    max_options: int = 10

    audit_per_source: int = 40
    audit_drop_below_chance: bool = True

    synth_target: int = 20000
    synth_shard: int = 200
    synth_max_tokens: int = 900
    synth_temperature: float = 0.95
    synth_top_p: float = 0.95
    synth_max_shards: int = 2000
    verify_orders: int = 2
    verify_min_p: float = 0.5
    verify_min_quality: float = 3.0
    verify_dedup_bits: int = 3
    soft_label_mix: float = 0.6
    synth_multi_share: float = 0.4
    verify_relabel_conf: float = 0.85

    aug_descriptions: float = 0.4
    aug_levels: float = 0.5
    aug_criteria: float = 0.3
    aug_to_multi: float = 0.05

    synth_repeat: int = 2
    val_fraction: float = 0.03
    test_fraction: float = 0.03
    label_smoothing: float = 0.03
    score_neighbor_mass: float = 0.15

    encoder: str = "answerdotai/ModernBERT-base"
    max_len: int = 512
    max_question_tokens: int = 96
    max_option_tokens: int = 32
    head_layers: int = 1
    head_dropout: float = 0.1

    epochs: float = 3.0
    lr: float = 5e-5
    head_lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    min_lr_ratio: float = 0.05
    batch_tokens: int = 0  # 0 = auto
    target_batch_examples: int = 64
    grad_clip: float = 1.0
    grad_checkpointing: bool = False
    amp_dtype: str = "bf16"
    eval_every: int = 400
    eval_examples: int = 3000
    ckpt_every: int = 200
    early_stop_patience: int = 8
    seed: int = 1234
    device: str = "cuda"

    round_increment: int = 10000

    auto_install: bool = True

    @classmethod
    def load(cls) -> "Config":
        raw = read_json(P.config, {}) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        write_json(P.config, asdict(self))

    def as_dict(self) -> dict:
        return asdict(self)

    def apply(self, overrides: List[str]) -> List[str]:
        """Apply key=value strings; returns what changed."""
        applied = []
        names = {f.name for f in fields(self)}
        for ov in overrides or []:
            if "=" not in ov:
                raise SystemExit("--set expects key=value, got: " + repr(ov))
            k, v = ov.split("=", 1)
            k = k.strip()
            if k not in names:
                raise SystemExit("unknown config key %r (see --show-config)" % k)
            cur = getattr(self, k)
            try:
                if isinstance(cur, bool):
                    nv: Any = str(v).lower() in ("1", "true", "yes", "y", "on")
                elif isinstance(cur, int):
                    nv = int(float(v))
                elif isinstance(cur, float):
                    nv = float(v)
                elif isinstance(cur, list):
                    nv = [s for s in (x.strip() for x in v.split(",")) if s]
                else:
                    nv = v
            except ValueError:
                raise SystemExit("bad value for %s: %r" % (k, v))
            setattr(self, k, nv)
            applied.append("%s=%s" % (k, nv))
        return applied


STAGE_KEYS: Dict[str, List[str]] = {
    "env":       [],
    "teacher":   ["llm_model"],
    "sources":   ["sources", "exclude_sources", "use_kaggle", "per_source_max",
                  "max_state_chars", "max_option_chars", "max_options", "seed"],
    "audit":     ["audit_per_source", "audit_drop_below_chance"],
    "synth":     ["synth_target", "verify_orders", "verify_min_p",
                  "verify_min_quality", "soft_label_mix", "synth_multi_share", "verify_relabel_conf"],
    "describe":  ["aug_levels"],
    "build":     ["heldout_sources", "synth_repeat", "val_fraction", "test_fraction",
                  "label_smoothing", "score_neighbor_mass", "seed", "aug_descriptions",
                  "aug_levels", "aug_criteria", "aug_to_multi"],
    "train":     ["encoder", "max_len", "max_question_tokens", "max_option_tokens",
                  "head_layers", "head_dropout", "epochs", "lr", "head_lr",
                  "weight_decay", "warmup_ratio", "min_lr_ratio",
                  "target_batch_examples", "grad_clip", "amp_dtype", "seed"],
    "calibrate": [],
    "evaluate":  [],
    "plots":     [],
    "export":    [],
}

STAGE_DEPS: Dict[str, List[str]] = {
    "env":       [],
    "teacher":   ["env"],
    "sources":   ["env"],
    "audit":     ["teacher", "sources"],
    "synth":     ["teacher"],
    "describe":  ["teacher", "sources", "synth"],
    "build":     ["sources", "audit", "synth", "describe"],
    "train":     ["build"],
    "calibrate": ["train"],
    "evaluate":  ["calibrate"],
    "plots":     ["evaluate"],
    "export":    ["evaluate"],
}


def stage_hashes(cfg: Config, order: List[str]) -> Dict[str, str]:
    d = cfg.as_dict()
    out: Dict[str, str] = {}
    for name in order:
        own = {k: d.get(k) for k in STAGE_KEYS.get(name, [])}
        if name == "sources":
            from .sources import REGISTRY
            own["_catalog"] = sorted(n if s.version == 1 else "%s@%d" % (n, s.version)
                                     for n, s in REGISTRY.items())
        elif name == "build":
            from .corpus import digest
            own["_edits"] = digest()
        out[name] = sha1({"own": own, "deps": [out.get(x) for x in STAGE_DEPS.get(name, [])]})
    return out
