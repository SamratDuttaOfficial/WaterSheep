"""Show configs, splits and a sample row of Hugging Face datasets.

    python tools/check_hf.py org/name[:config] ...
"""
import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://datasets-server.huggingface.co/"


def get(path, **q):
    url = API + path + "?" + urllib.parse.urlencode(q)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except Exception as e:
        try:
            return {"error": json.load(e).get("error", str(e))}
        except Exception:
            return {"error": str(e)}


def feat(f):
    if not isinstance(f, dict):
        return "?"
    t = f.get("_type")
    if t == "ClassLabel":
        n = f.get("names", [])
        return "CL%d%s" % (len(n), n[:12])
    if t == "Value":
        return f.get("dtype")
    if t in ("Sequence", "List"):
        return "[%s]" % feat(f.get("feature"))
    if t is None:
        return "{%s}" % ",".join("%s:%s" % (k, feat(v)) for k, v in f.items())
    return t


def short(v, n=160):
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "..."


def check(spec):
    ds, _, cfg = spec.partition(":")
    info = get("info", dataset=ds, **({"config": cfg} if cfg else {}))
    if "error" in info:
        return "%s  !! %s" % (spec, str(info["error"])[:150])
    di = info.get("dataset_info", {})
    if cfg:
        di = {cfg: di}
    out = ["%s  configs=%d %s" % (spec, len(di), list(di)[:15])]
    c0 = cfg or next(iter(di))
    d = di[c0]
    out.append("  [%s] splits: %s" % (c0, {k: v.get("num_examples") for k, v in d.get("splits", {}).items()}))
    out.append("  cols: " + ", ".join("%s=%s" % (k, feat(v)) for k, v in d.get("features", {}).items()))
    split = "train" if "train" in d.get("splits", {}) else next(iter(d.get("splits", {})), "train")
    fr = get("first-rows", dataset=ds, config=c0, split=split)
    rows = fr.get("rows") or []
    if rows:
        out.append("  row: " + " | ".join("%s=%s" % (k, short(v)) for k, v in rows[0]["row"].items()))
    elif "error" in fr:
        out.append("  (no rows: %s)" % str(fr["error"])[:100])
    return "\n".join(out)


if __name__ == "__main__":
    specs = sys.argv[1:] or [l.strip() for l in sys.stdin if l.strip()]
    with ThreadPoolExecutor(16) as ex:
        for s in ex.map(check, specs):
            print(s, flush=True)
