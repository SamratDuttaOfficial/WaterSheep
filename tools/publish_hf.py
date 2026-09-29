#!/usr/bin/env python
"""Publish an export on the Hugging Face Hub.

Uploads the export (with its ONNX build, if any), the transformers code in hf/, a model card
(tools/model_card.md), LICENSE and NOTICE in one commit. Log in first with `hf auth login`.

  python tools/publish_hf.py --repo OWNER/NAME --dry-run     write the card to out/hf/, upload nothing
  python tools/publish_hf.py --repo OWNER/NAME               the newest export
  python tools/publish_hf.py --repo OWNER/NAME --model out/export/NAME --tag v0.1.0
"""
from __future__ import annotations
import argparse
import fnmatch
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from watersheep import __version__                                       # noqa: E402
from watersheep.infer import HUB_FILES, HUB_ID, find_export              # noqa: E402
from watersheep.paths import P                                           # noqa: E402
from watersheep.util import read_json                                    # noqa: E402

TEMPLATE = Path(__file__).resolve().parent / "model_card.md"
HF_CODE = ROOT / "hf"
IN_TRAINING = {"no": "no", "YES - same split": "same split", "other split only": "other split"}
ONNX = "onnx/model_quantized.onnx"
PUBLISH_FILES = HUB_FILES + ["onnx/*"]


def git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception:
        return ""


def code_url() -> str:
    """The 'origin' remote as a public https URL, without credentials."""
    u = git("remote", "get-url", "origin")
    m = re.match(r"^[\w.-]+@([^:/]+):(.+)$", u) or \
        re.match(r"^ssh://(?:[^@/]+@)?([^/:]+)(?::\d+)?/(.+)$", u)
    if m:
        u = "https://%s/%s" % m.groups()
    if not u.startswith(("https://", "http://")):
        return ""
    u = re.sub(r"^(https?://)[^/@]*@", r"\1", u)
    return re.sub(r"\.git$", "", u.rstrip("/"))


def author() -> str:
    """The copyright holder in NOTICE, else the git user."""
    try:
        m = re.search(r"^Copyright\s+\S+\s+(.+?)\s*(<[^>]*>)?\s*$",
                      (ROOT / "NOTICE").read_text(encoding="utf-8"), re.M)
        if m:
            return m.group(1)
    except OSError:
        pass
    return git("config", "user.name")


def export_files(d: Path) -> List[Tuple[str, Path]]:
    """(path in the repo, local file) for the files a model needs, except watersheep.json."""
    out = []
    for f in sorted(d.rglob("*")):
        rel = f.relative_to(d).as_posix()
        if f.is_file() and rel != "watersheep.json" and any(fnmatch.fnmatch(rel, p) for p in PUBLISH_FILES):
            out.append((rel, f))
    return out


def online(repo: str, url: str) -> Tuple[str, str, bool]:
    """The project site and the Space for this repo, if they exist, and whether it has the ONNX build."""
    space, has_onnx = "", False
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        if api.repo_exists(repo, repo_type="space"):
            space = "https://huggingface.co/spaces/%s" % repo
        has_onnx = api.file_exists(repo, ONNX)
    except Exception:
        pass
    m = re.match(r"^https://github\.com/([^/]+)/([^/]+)$", url or "")
    site = "https://%s.github.io/%s/" % (m.group(1).lower(), m.group(2)) if m else ""
    try:
        with urllib.request.urlopen(site, timeout=10) as r:
            site = site if r.status == 200 else ""
    except Exception:
        site = ""
    return site, space, has_onnx


def header(title: str, site: str, space: str, url: str) -> str:
    links = " · ".join("[%s](%s)" % (k, v) for k, v in (("Website", site), ("Demo", space), ("Code", url)) if v)
    if not site:
        return "# %s\n\n%s" % (title, links)
    return "![%s](%sbanner.svg)\n\n# ![](%slogo.svg) %s\n\n%s" % (title, site, site, title, links)


def js_block(site: str, title: str, onnx: bool) -> str:
    if not site:
        return ""
    lines = ["## JavaScript", "", "No install; runs in the browser:", "",
             "```html", '<script type="module">',
             '  import { decide } from "%swatersheep.js";' % site,
             '  console.log(await decide("I was charged twice.", "Which team should handle this?", '
             '["billing", "shipping", "support"]));', "</script>", "```", "",
             'With a downloaded copy on your web server, call `load({ base: "%s/" })` first.' % title]
    if onnx:
        lines += ["", "Other languages: run `%s` with ONNX Runtime; `watersheep.js` shows the input format." % ONNX]
    return "\n".join(lines)


def hf_config(d: Path, meta: dict) -> dict:
    """config.json for loading the export with transformers (trust_remote_code)."""
    enc = read_json(d / "encoder" / "config.json", {}) or {}
    return {
        "architectures": ["WaterSheepModel"],
        "model_type": "watersheep",
        "auto_map": {"AutoConfig": "modeling_watersheep.WaterSheepConfig",
                     "AutoModel": "modeling_watersheep.WaterSheepModel"},
        "custom_pipelines": {"zero-shot-classification": {
            "impl": "pipeline_watersheep.WaterSheepPipeline", "pt": ["AutoModel"]}},
        "encoder_config": enc,
        "head_layers": meta.get("head_layers", 1),
        "max_len": meta["max_len"],
        "max_question_tokens": meta["max_question_tokens"],
        "max_option_tokens": meta["max_option_tokens"],
        "max_options": meta.get("max_options", 10),
        "temperatures": meta.get("temperatures") or {},
        "multi_threshold": meta.get("multi_threshold", 0.5),
        "dtype": enc.get("dtype", "float32"),
        "transformers_version": enc.get("transformers_version"),
    }


def train_datasets() -> List[str]:
    """Hugging Face ids of the training sources (held-out sources excluded)."""
    try:
        from watersheep.config import Config
        from watersheep.sources import enabled
        cfg = Config.load()
        held = set(cfg.heldout_sources)
        return sorted({s.hf for s in enabled(cfg) if getattr(s, "hf", None) and s.name not in held})
    except Exception as e:
        print("note: the card metadata lists no datasets (%s)" % e)
        return []


def metrics_table(meta: dict) -> str:
    m = meta.get("metrics") or {}
    rows = [("In-distribution test split", m.get("test_calibrated")),
            ("Held-out datasets, not seen in training", m.get("zeroshot_calibrated"))]
    lines = ["| Evaluation | Accuracy | ECE |", "|---|---|---|"]
    lines += ["| %s | %.1f%% | %.3f |" % (k, 100 * r["acc"], r["ece"]) for k, r in rows
              if isinstance(r, dict) and r.get("acc") is not None and r.get("ece") is not None]
    return "\n".join(lines) if len(lines) > 2 else "No evaluation results were saved with this export."


def bench_table(name: str) -> str:
    """Saved results of run-benchmarks for this export, if any."""
    try:
        from benchmark import BENCHMARKS
    except Exception:
        return ""
    known = {b.name: b for b in BENCHMARKS}
    rows = []
    for f in sorted((P.out / "benchmarks" / name.replace("watersheep-", "watersheep_")).glob("*.json")):
        r = read_json(f, {}) or {}
        b = known.get(r.get("benchmark")) if isinstance(r, dict) else None
        if b and r.get("acc") is not None:
            label = "[%s](https://huggingface.co/datasets/%s)" % (b.name, b.hf) if b.hf else b.name
            rows.append("| %s | %s | %d | %.1f%% | %s | %s |" % (
                label, b.suite, r.get("n", 0), 100 * r["acc"],
                "%.3f" % r["ece"] if r.get("ece") is not None else "-",
                IN_TRAINING.get(r.get("trained_on"), "?")))
    if not rows:
        return ""
    return "\n".join(["### Benchmarks", "",
                      "| Benchmark | Suite | Questions | Accuracy | ECE | In training data |",
                      "|---|---|---|---|---|---|"] + rows)


def front_matter(meta: dict) -> str:
    base = str(meta.get("encoder") or "")
    lines = ["---", "license: apache-2.0", "language:", "- en", "library_name: transformers",
             "pipeline_tag: zero-shot-classification"]
    if HUB_ID.match(base):
        lines.append("base_model: %s" % base)
    lines += ["tags:", "- decision-model", "- calibration", "- multi-label"]
    ds = train_datasets()
    if ds:
        lines += ["datasets:"] + ["- %s" % x for x in ds]
    return "\n".join(lines + ["---", ""])


def render_card(repo: str, meta: dict, d: Path, a) -> str:
    url = code_url() if a.code_url is None else a.code_url
    site, space, remote_onnx = online(repo, url)
    title = repo.split("/")[-1]
    fill = {
        "title": title, "repo_id": repo, "version": __version__, "name": meta["name"],
        "header": header(title, site, space, url),
        "javascript": js_block(site, title, (d / ONNX).exists() or remote_onnx),
        "metrics": metrics_table(meta), "benchmarks": bench_table(meta["name"]),
        "author": author(), "year": str(meta.get("created") or time.strftime("%Y"))[:4],
    }
    text = TEMPLATE.read_text(encoding="utf-8")
    for k, v in fill.items():
        text = text.replace("{{%s}}" % k, str(v))
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", text))
    return front_matter(meta) + "\n" + text


def upload(a, uploads, card: str, meta_bytes: bytes, config: bytes, name: str) -> int:
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        print("not logged in to Hugging Face - run `hf auth login` (or set HF_TOKEN) and try again")
        return 1
    print("logged in as %s, uploading..." % user)
    api.create_repo(a.repo, private=a.private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(f)) for rel, f in uploads]
    ops += [CommitOperationAdd(path_in_repo="watersheep.json", path_or_fileobj=meta_bytes),
            CommitOperationAdd(path_in_repo="config.json", path_or_fileobj=config),
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode("utf-8"))]
    info = api.create_commit(a.repo, operations=ops, commit_message="Upload %s %s (%s)" % (
        a.repo.split("/")[-1], __version__, name))
    if a.tag:
        try:
            api.create_tag(a.repo, tag=a.tag, revision=info.oid)
        except Exception as e:
            print("tag %s not created: %s" % (a.tag, str(e).strip().splitlines()[0]))
    print("published: https://huggingface.co/%s" % a.repo)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="Hugging Face repo id, owner/name")
    ap.add_argument("--model", help="export folder or export name (default: the newest export)")
    ap.add_argument("--tag", help="also tag the upload, e.g. v%s" % __version__)
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    ap.add_argument("--code-url", help="link to the code (default: the git remote 'origin', '' for none)")
    ap.add_argument("--dry-run", action="store_true", help="write the card to out/hf/ and upload nothing")
    a = ap.parse_args()
    if not HUB_ID.match(a.repo) or "@" in a.repo:
        ap.error("--repo must look like owner/name")

    d = find_export(a.model)
    if not d:
        print("no export found%s - run the pipeline first" % (" at %r" % a.model if a.model else ""))
        return 1
    meta = read_json(d / "watersheep.json", {}) or {}
    meta.setdefault("name", d.name.replace("watersheep_", "watersheep-"))
    files = export_files(d)
    rels = {rel for rel, _ in files}
    missing = [x for x in ("model.safetensors", "encoder/config.json") if x not in rels]
    if meta.get("tokenizer", "hf") == "hf" and not any(r.startswith("tokenizer/") for r in rels):
        missing.append("tokenizer/")
    if missing:
        print("%s is incomplete, missing: %s" % (d, ", ".join(missing)))
        return 1
    extra = [(n, ROOT / n) for n in ("LICENSE", "NOTICE") if (ROOT / n).exists()]
    extra += [(f.name, f) for f in sorted(HF_CODE.glob("*")) if f.suffix in (".py", ".txt")]
    if len(extra) < 2:
        print("warning: LICENSE or NOTICE is missing in %s" % ROOT)
    if not (d / ONNX).exists():
        print("note: no %s - run tools/export_onnx.py first, or an ONNX build already in the repo "
              "stays and no longer matches" % ONNX)
    elif (d / ONNX).stat().st_mtime < (d / "model.safetensors").stat().st_mtime:
        print("warning: %s is older than model.safetensors - run tools/export_onnx.py again" % ONNX)

    card = render_card(a.repo, meta, d, a)
    meta_bytes = (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    config = (json.dumps(hf_config(d, meta), indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    uploads = files + extra
    print("%s -> https://huggingface.co/%s" % (d, a.repo))
    for rel, f in uploads:
        print("  %-30s %9.1f MB" % (rel, f.stat().st_size / 2 ** 20))
    print("  %-30s (with name %s)\n  %-30s\n  %-30s (model card)" % (
        "watersheep.json", meta["name"], "config.json", "README.md"))
    if a.dry_run:
        out = P.out / "hf" / a.repo.replace("/", "--")
        out.mkdir(parents=True, exist_ok=True)
        (out / "README.md").write_text(card, encoding="utf-8")
        (out / "watersheep.json").write_bytes(meta_bytes)
        (out / "config.json").write_bytes(config)
        print("dry run - nothing uploaded; model card: %s" % (out / "README.md"))
        return 0
    return upload(a, uploads, card, meta_bytes, config, meta["name"])


if __name__ == "__main__":
    sys.exit(main())
