"""Local web UI: status, datasets and corpus browser."""
from __future__ import annotations
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .data import corpus
from .core.paths import P
from .core.util import LOG, dir_size, human_bytes, iter_jsonl, pid_alive, read_json

_CACHE: dict = {}
_LOCK = threading.Lock()


def _cached(key, stamp, make):
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
    val = make()
    with _LOCK:
        _CACHE[key] = (stamp, val)
    return val


def _stamp(f: Path):
    try:
        st = f.stat()
        return st.st_mtime_ns, st.st_size
    except OSError:
        return None


def _tail(f: Path, n: int = 40) -> list:
    try:
        with open(f, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 16000))
            return fh.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []


def _synth_rows() -> list:
    out = []
    for f in sorted(list(P.synth.glob("shard_*.jsonl")) + list(P.synth.glob("shard_*.part"))):
        out.extend(_cached(("synth", f.name), _stamp(f), lambda f=f: [
            _synth_view(r) for r in iter_jsonl(f)]))
    return out


def _synth_view(r: dict) -> dict:
    rec = r.get("record") or {}
    return {"id": rec.get("id") or "syn:" + str(r.get("jid")), "source": rec.get("source", "synth"),
            "kind": "synth", "status": r.get("status"), "reason": r.get("reason"),
            "type": rec.get("type") or r.get("type"), "state": rec.get("state", ""),
            "question": rec.get("question", ""), "options": rec.get("options", []),
            "answer": rec.get("answer"), "p": r.get("p_answer"), "quality": r.get("quality"),
            "meta": {k: (rec.get("meta") or {}).get(k) for k in ("family", "domain", "difficulty", "why")}}


def _raw_rows(name: str) -> list:
    from .data.dataset import raw_file
    f = raw_file(name)
    return _cached(("raw", name), _stamp(f), lambda: list(iter_jsonl(f)))


def _filter(q: dict) -> list:
    src, text = q.get("src", ""), q.get("q", "").strip().lower()
    status, show = q.get("status", "accepted"), q.get("deleted", "hide")
    rows = _synth_rows() if src == "synth" else _raw_rows(src)
    deleted = corpus.load()["deleted"]
    out = []
    for r in rows:
        if src == "synth" and status != "all" and r.get("status") != status:
            continue
        gone = r["id"] in deleted
        if (show == "hide" and gone) or (show == "only" and not gone):
            continue
        if text and text not in "\n".join([r.get("state", ""), r.get("question", "")] +
                                          list(r.get("options") or [])).lower():
            continue
        out.append(dict(r, deleted=gone))
    return out


def records(q: dict) -> dict:
    rows = _filter(q)
    offset, limit = int(q.get("offset", 0)), min(200, int(q.get("limit", 50)))
    return {"total": len(rows), "rows": rows[offset:offset + limit]}


def sources() -> list:
    from .core.config import Config
    from .data.dataset import raw_file
    from .data import sources as S
    cfg = Config.load()
    rep = read_json(P.reports / "sources.json", {}) or {}
    audit = (read_json(P.reports / "audit.json", {}) or {}).get("per_source", {})
    dropped = set((read_json(P.reports / "audit.json", {}) or {}).get("dropped", []))
    edits = corpus.load()
    per_del: dict = {}
    for src in edits["deleted"].values():
        per_del[src] = per_del.get(src, 0) + 1
    on = {s.name for s in S.enabled(cfg)}
    out = []
    for s in S.REGISTRY.values():
        m = read_json(raw_file(s.name).with_suffix(".done"), {}) or {}
        failed = (rep.get("failed") or {}).get(s.name)
        state = ("loaded" if m else "failed" if failed else "pending") if s.name in on else "excluded"
        link = ("https://huggingface.co/datasets/" + s.hf if s.hf else
                "https://www.kaggle.com/datasets/" + s.kaggle if s.kaggle else s.url)
        a = audit.get(s.name) or {}
        out.append({"name": s.name, "type": s.type, "origin": s.origin, "ref": s.ref, "link": link,
                    "rows": m.get("n", 0), "state": state, "error": failed or "",
                    "audit": a.get("agree"), "chance": a.get("chance"),
                    "audit_dropped": s.name in dropped, "heldout": s.name in cfg.heldout_sources,
                    "disabled": s.name in edits["disabled"], "deleted": per_del.get(s.name, 0),
                    "note": s.note})
    syn = read_json(P.reports / "synth.json", {}) or {}
    n_syn = sum(v for k, v in per_del.items() if k.startswith("synth"))
    out.append({"name": "synth", "type": "mixed", "origin": "generated", "ref": "Qwen3.5-4B teacher",
                "link": "", "rows": syn.get("accepted", 0), "state": "loaded" if syn else "pending",
                "error": "", "audit": syn.get("acceptance"), "chance": None, "audit_dropped": False,
                "heldout": False, "disabled": "synth" in edits["disabled"], "deleted": n_syn,
                "note": "accepted synthetic examples (acceptance rate shown as audit)"})
    return out


def _disk() -> dict:
    return _cached("disk", int(time.time() // 300), lambda: {
        n: human_bytes(dir_size(p)) for n, p in (("data", P.data), ("hf cache", P.hf),
                                                 ("kaggle", P.kaggle), ("downloads", P.downloads),
                                                 ("checkpoints", P.ckpt), ("out", P.out))})


def overview() -> dict:
    from .core.config import Config, stage_hashes
    from .training.stages import ORDER
    from .core.state import Manifest, Registry
    from .core import gpu
    cfg = Config.load()
    man, h = Manifest(), stage_hashes(cfg, ORDER)
    stages = []
    for name in ORDER:
        e = man.entry(name)
        st = e.get("status", "pending")
        if st == "done" and e.get("hash") != h[name]:
            st = "stale"
        stages.append({"name": name, "status": st, "meta": e.get("meta") or {},
                       "when": (e.get("finished") or e.get("stopped_at") or e.get("started") or "")[:19],
                       "progress": e.get("progress")})
    try:
        pid = int((P.state / "run.lock").read_text().strip())
    except Exception:
        pid = 0
    synth_log = _tail(P.logs / "synth.jsonl", 1)
    live = json.loads(synth_log[-1]) if synth_log else {}
    syn = read_json(P.reports / "synth.json", {}) or {}
    reg = Registry()
    run = reg.active("run")
    ev = read_json(P.reports / ("eval_%s.json" % run["id"]), {}) if run else {}
    logs = sorted(P.logs.glob("run_*.log"))
    src_rep = read_json(P.reports / "sources.json", {}) or {}
    edits = corpus.load()
    return {
        "running": bool(pid and pid_alive(pid)), "stages": stages,
        "synth": {"accepted": live.get("accepted", syn.get("accepted", 0)),
                  "tried": live.get("tried", syn.get("tried", 0)), "target": cfg.synth_target,
                  "tok_s": live.get("tok_s"), "updated": live.get("t"),
                  "reasons": syn.get("reasons", {})},
        "gpu": gpu.status(max_age=5.0),
        "corpus": {"sources": len(src_rep.get("counts") or {}),
                   "records": sum((src_rep.get("counts") or {}).values()),
                   "failed": len(src_rep.get("failed") or {}),
                   "deleted": len(edits["deleted"]), "disabled": edits["disabled"],
                   "build": read_json(P.build / "stats.json", {}) or {}},
        "eval": {k: {x: (ev.get(k) or {}).get(x) for x in ("n", "acc", "ece")}
                 for k in ("test_calibrated", "zeroshot_calibrated") if ev.get(k)},
        "disk": _disk(), "log": _tail(logs[-1], 40) if logs else [], "log_file": logs[-1].name if logs else "",
    }


def act(path: str, body: dict) -> dict:
    src = body.get("src", "")
    if path == "/api/delete":
        if body.get("all"):
            items = {r["id"]: r["source"] for r in _filter(dict(body, deleted="hide"))}
        elif src == "synth":
            src_of = {r["id"]: r["source"] for r in _synth_rows()}
            items = {i: src_of.get(i, "synth") for i in body.get("ids", [])}
        else:
            items = {i: src for i in body.get("ids", [])}
        return {"deleted": corpus.delete(items)}
    if path == "/api/restore":
        if not body.get("all"):
            return {"restored": corpus.restore(body.get("ids", []))}
        if src == "synth":
            return {"restored": corpus.restore([k for k, v in corpus.load()["deleted"].items()
                                                if v.startswith("synth")])}
        return {"restored": corpus.restore(source=src)}
    if path == "/api/source":
        corpus.set_disabled(body["name"], bool(body.get("disabled")))
        return {"ok": True}
    raise KeyError(path)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                return self._send(200, PAGE.encode("utf-8"), "text/html")
            fn = {"/api/overview": overview, "/api/sources": sources,
                  "/api/records": lambda: records(q)}.get(u.path)
            if not fn:
                return self._send(404, {"error": "not found"})
            self._send(200, fn())
        except Exception as e:
            self._send(500, {"error": repr(e)})

    def do_POST(self):
        if self.headers.get("X-WaterSheep") != "1":  # blocks cross-site form posts
            return self._send(403, {"error": "forbidden"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, act(urlparse(self.path).path, body))
        except KeyError:
            self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": repr(e)})


def start(port: int = 8765, open_browser: bool = True):
    """Serve in a background thread; returns the URL or None."""
    for p in range(port, port + 10):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), Handler)
        except OSError:
            continue
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, name="webui", daemon=True).start()
        url = "http://127.0.0.1:%d/" % p
        LOG.info("web UI: %s", url)
        if open_browser:
            threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
        return url
    LOG.warning("web UI: ports %d-%d are busy, not started", port, port + 9)
    return None


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WaterSheep</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1c2330;--mut:#667085;--line:#e3e6eb;--acc:#2f6fdf;--ok:#1f8f4e;--warn:#b7791f;--bad:#c53030;--chip:#eef2f7}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--fg:#e6e9ef;--mut:#98a2b3;--line:#2b3240;--acc:#6ea0ff;--ok:#4cc17f;--warn:#e0a84a;--bad:#f07373;--chip:#252c38}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:12px 20px;background:var(--card);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:2;flex-wrap:wrap}
h1{font-size:17px;margin:0}nav button{background:none;border:0;color:var(--mut);font:inherit;padding:6px 10px;border-radius:6px;cursor:pointer}
nav button.on{background:var(--chip);color:var(--fg);font-weight:600}
main{padding:16px 20px;max-width:1400px;margin:0 auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);margin:0 0 10px}
.big{font-size:26px;font-weight:650}.mut{color:var(--mut)}.pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:12px;background:var(--chip)}
.done,.loaded{color:var(--ok)}.running,.stale,.pending{color:var(--warn)}.failed{color:var(--bad)}.stopped{color:var(--acc)}
table{width:100%;border-collapse:collapse}td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--mut);font-weight:500;font-size:12px;cursor:pointer;user-select:none}
.bar{height:8px;background:var(--chip);border-radius:99px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
pre{white-space:pre-wrap;font:12px/1.4 ui-monospace,Consolas,monospace;margin:0;max-height:340px;overflow:auto}
input,select,button.b{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
button.b{cursor:pointer}button.b:hover{border-color:var(--acc)}button.red{color:var(--bad)}
.tools{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
.rec{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:10px}.rec.gone{opacity:.45}
.rec .st{white-space:pre-wrap;max-height:180px;overflow:auto;background:var(--bg);padding:8px;border-radius:6px;margin:6px 0;font-size:13px}
.opt{display:inline-block;margin:2px 4px 2px 0;padding:2px 8px;border-radius:6px;background:var(--chip)}.opt.ans{outline:2px solid var(--ok)}
.hide{display:none}a{color:var(--acc)}.scroll{overflow-x:auto}
</style></head><body>
<header><h1>🐑 WaterSheep</h1><span id="live" class="pill">…</span>
<nav><button data-t="ov" class="on">Overview</button><button data-t="ds">Datasets</button><button data-t="br">Browse & edit corpus</button></nav>
<span class="mut" id="clock"></span></header>
<main>
<section id="ov"><div class="grid" id="cards"></div>
<div class="grid" style="margin-top:14px"><div class="card"><h2>Pipeline</h2><div class="scroll"><table id="stages"></table></div></div>
<div class="card"><h2>Latest log <span class="mut" id="logf"></span></h2><pre id="log"></pre></div></div></section>

<section id="ds" class="hide"><div class="tools"><input id="dsq" placeholder="filter by name or origin" size="28">
<select id="dso"><option value="">all origins</option><option>huggingface</option><option>kaggle</option><option>web</option><option>generated</option></select>
<select id="dss"><option value="">any state</option><option>loaded</option><option>pending</option><option>failed</option><option>excluded</option></select>
<span class="mut" id="dsn"></span></div>
<div class="card scroll"><table id="dst"></table></div>
<p class="mut">"Off" leaves a whole source out of the next build. Records deleted in "Browse" are listed here too. Changes take effect at the next <b>build</b>, which also means retraining.</p></section>

<section id="br" class="hide"><div class="tools">
<select id="bsrc"></select><input id="bq" placeholder="search text" size="26">
<select id="bst"><option value="accepted">accepted</option><option value="rejected">rejected</option><option value="all">all</option></select>
<select id="bdel"><option value="hide">hide deleted</option><option value="show">show deleted</option><option value="only">only deleted</option></select>
<button class="b" id="bgo">Search</button><span class="mut" id="bn"></span></div>
<div class="tools"><button class="b red" id="bdelsel">Delete selected</button><button class="b red" id="bdelall">Delete all matching</button>
<button class="b" id="bres">Restore everything deleted in this source</button>
<button class="b" id="bprev">← Prev</button><button class="b" id="bnext">Next →</button></div>
<div id="recs"></div></section>
</main>
<script>
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const get=u=>fetch(u).then(r=>r.json()), post=(u,b)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json','X-WaterSheep':'1'},body:JSON.stringify(b)}).then(r=>r.json());
let tab='ov', DS=[], sortK='name', sortD=1, off=0, total=0;
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{tab=b.dataset.t;document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('on',x===b));
 ['ov','ds','br'].forEach(t=>$('#'+t).classList.toggle('hide',t!==tab)); if(tab==='ds')loadDS(); if(tab==='br'&&!$('#bsrc').options.length)loadDS().then(()=>fillSrc());});
const pct=(a,b)=>b?Math.min(100,100*a/b):0, fmt=n=>n==null?'–':Number(n).toLocaleString();
function card(t,big,sub,bar){return `<div class="card"><h2>${t}</h2><div class="big">${big}</div>${bar!=null?`<div class="bar"><i style="width:${bar}%"></i></div>`:''}<div class="mut">${sub||''}</div></div>`}
async function loadOV(){const o=await get('/api/overview'); if(o.error)return;
 $('#live').textContent=o.running?'● pipeline running':'○ pipeline not running'; $('#live').style.color=o.running?'var(--ok)':'var(--mut)';
 const s=o.synth, g=o.gpu, c=o.corpus, b=c.build.splits;
 const reasons=Object.entries(s.reasons||{}).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([k,v])=>`${k} ${fmt(v)}`).join(' · ');
 let ev=Object.entries(o.eval).map(([k,v])=>`${k.split('_')[0]}: acc ${(v.acc*100).toFixed(1)}% · ECE ${v.ece.toFixed(3)}`).join('<br>');
 $('#cards').innerHTML=card('Synthetic data',`${fmt(s.accepted)} / ${fmt(s.target)}`,`accepted of ${fmt(s.tried)} tried (${pct(s.accepted,s.tried).toFixed(0)}%)${s.tok_s?' · '+s.tok_s.toFixed(0)+' tok/s':''}<br>${reasons}`,pct(s.accepted,s.target))
 +card('Public corpus',`${fmt(c.records)} records`,`${c.sources} sources loaded · ${c.failed} failed · ${fmt(c.deleted)} deleted by you${c.disabled.length?' · off: '+c.disabled.join(', '):''}${b?'<br>last build: '+Object.entries(b).map(([k,v])=>k+' '+fmt(v)).join(' · '):''}`)
 +card('GPU',g?`${g.util??'–'}%`:'–',g?`${esc(g.name)}<br>${fmt(g.used)} / ${fmt(g.total)} MB`:'no NVIDIA GPU found',g?g.util:null)
 +card('Model',ev?'':'not trained yet',ev||'')
 +card('Disk',Object.values(o.disk).length?'':'',Object.entries(o.disk).map(([k,v])=>`${k}: ${v}`).join('<br>'));
 $('#stages').innerHTML='<tr><th>stage</th><th>status</th><th>when</th><th>detail</th></tr>'+o.stages.map(x=>`<tr><td>${x.name}</td><td class="${x.status}">${x.status}</td><td class="mut">${esc(x.when)}</td><td class="mut">${esc(Object.entries(x.meta).filter(([k,v])=>typeof v!=='object').map(([k,v])=>k+'='+v).join(' ').slice(0,90))}</td></tr>`).join('');
 $('#log').textContent=o.log.join('\n'); $('#logf').textContent=o.log_file; $('#log').scrollTop=1e9; $('#clock').textContent='updated '+new Date().toLocaleTimeString();}
async function loadDS(){DS=await get('/api/sources'); drawDS();}
function drawDS(){const q=$('#dsq').value.toLowerCase(), o=$('#dso').value, st=$('#dss').value;
 let rows=DS.filter(d=>(!q||(d.name+' '+d.ref+' '+d.origin).toLowerCase().includes(q))&&(!o||d.origin===o)&&(!st||d.state===st));
 rows.sort((a,b)=>(a[sortK]>b[sortK]?1:a[sortK]<b[sortK]?-1:0)*sortD);
 const tot=rows.reduce((a,d)=>a+(d.rows||0),0); $('#dsn').textContent=`${rows.length} sources · ${fmt(tot)} records`;
 const H=[['name','source'],['origin','origin'],['type','type'],['rows','rows'],['state','state'],['audit','teacher agrees'],['deleted','deleted'],['disabled','use']];
 $('#dst').innerHTML='<tr>'+H.map(([k,t])=>`<th data-k="${k}">${t}</th>`).join('')+'</tr>'+rows.map(d=>`<tr>
 <td><a href="#" data-b="${esc(d.name)}">${esc(d.name)}</a>${d.heldout?' <span class="pill">zero-shot test</span>':''}<br><span class="mut">${d.link?`<a href="${esc(d.link)}" target="_blank" rel="noopener">${esc(d.ref)}</a>`:esc(d.ref)}</span></td>
 <td>${d.origin}</td><td>${d.type}</td><td>${fmt(d.rows)}</td><td class="${d.state}" title="${esc(d.error)}">${d.state}${d.error?'<br><span class="mut">'+esc(d.error.slice(0,70))+'</span>':''}</td>
 <td>${d.audit==null?'–':(d.audit*100).toFixed(0)+'%'}${d.chance!=null?' <span class="mut">(chance '+(d.chance*100).toFixed(0)+'%)</span>':''}${d.audit_dropped?' <span class="failed">dropped</span>':''}</td>
 <td>${d.deleted||''}</td><td><button class="b" data-t="${esc(d.name)}" data-off="${d.disabled?0:1}">${d.disabled?'off':'on'}</button></td></tr>`).join('');
 $('#dst').querySelectorAll('th').forEach(th=>th.onclick=()=>{sortD=sortK===th.dataset.k?-sortD:1;sortK=th.dataset.k;drawDS();});
 $('#dst').querySelectorAll('[data-t]').forEach(b=>b.onclick=async()=>{await post('/api/source',{name:b.dataset.t,disabled:b.dataset.off==='1'});loadDS();});
 $('#dst').querySelectorAll('[data-b]').forEach(a=>a.onclick=e=>{e.preventDefault();fillSrc(a.dataset.b);document.querySelector('nav button[data-t=br]').click();});}
['#dsq','#dso','#dss'].forEach(s=>$(s).oninput=drawDS);
function fillSrc(pick){const sel=$('#bsrc'); if(!sel.options.length) sel.innerHTML=DS.filter(d=>d.rows>0||d.name==='synth').map(d=>`<option>${esc(d.name)}</option>`).join('');
 if(pick)sel.value=pick; off=0; loadRecs();}
function query(){return {src:$('#bsrc').value,q:$('#bq').value,status:$('#bst').value,deleted:$('#bdel').value}}
async function loadRecs(){const q=query(); $('#bst').disabled=q.src!=='synth';
 const r=await get('/api/records?'+new URLSearchParams({...q,offset:off,limit:50})); if(r.error){$('#recs').textContent=r.error;return;}
 total=r.total; $('#bn').textContent=`${fmt(total)} records · showing ${total?off+1:0}-${Math.min(off+50,total)}`;
 $('#recs').innerHTML=r.rows.map(x=>`<div class="rec ${x.deleted?'gone':''}"><label><input type="checkbox" data-id="${esc(x.id)}"> <b>${esc(x.id)}</b></label>
 <span class="pill">${x.type}</span> ${x.status?`<span class="pill ${x.status==='accepted'?'done':'failed'}">${x.status}${x.reason&&x.reason!=='ok'?': '+esc(x.reason):''}</span>`:''}
 ${x.p!=null?`<span class="mut">teacher p=${x.p} · quality ${x.quality??'–'}</span>`:''} ${x.meta&&x.meta.family?`<span class="mut">${esc(x.meta.family)} · ${esc(x.meta.domain)} · ${esc(x.meta.difficulty)}</span>`:''}
 <button class="b ${x.deleted?'':'red'}" style="float:right" data-one="${esc(x.id)}" data-restore="${x.deleted?1:0}">${x.deleted?'restore':'delete'}</button>
 ${x.state?`<div class="st">${esc(x.state)}</div>`:''}<div><b>${esc(x.question)}</b></div>
 <div>${(x.options||[]).map((o,i)=>`<span class="opt ${(Array.isArray(x.answer)?x.answer.includes(i):i===x.answer)?'ans':''}">${esc(o)}</span>`).join('')}</div>${x.meta&&x.meta.why?`<div class="mut">why: ${esc(x.meta.why)}</div>`:''}</div>`).join('')||'<p class="mut">nothing matches</p>';
 $('#recs').querySelectorAll('[data-one]').forEach(b=>b.onclick=async()=>{const id=b.dataset.one; await post(b.dataset.restore==='1'?'/api/restore':'/api/delete',{ids:[id],src:query().src}); loadRecs();});}
$('#bgo').onclick=()=>{off=0;loadRecs()}; $('#bq').onkeydown=e=>{if(e.key==='Enter'){off=0;loadRecs()}}; ['#bsrc','#bst','#bdel'].forEach(s=>$(s).onchange=()=>{off=0;loadRecs()});
$('#bprev').onclick=()=>{off=Math.max(0,off-50);loadRecs()}; $('#bnext').onclick=()=>{if(off+50<total){off+=50;loadRecs()}};
$('#bdelsel').onclick=async()=>{const ids=[...document.querySelectorAll('#recs input:checked')].map(i=>i.dataset.id); if(!ids.length)return; await post('/api/delete',{ids,src:query().src}); loadRecs();};
$('#bdelall').onclick=async()=>{if(!confirm(`Delete all ${total} matching records from training? (can be restored)`))return; await post('/api/delete',{...query(),all:true}); loadRecs();};
$('#bres').onclick=async()=>{if(!confirm('Restore every deleted record of this source?'))return; await post('/api/restore',{src:query().src,all:true}); loadRecs();};
loadOV(); setInterval(()=>{if(tab==='ov')loadOV()},10000);
</script></body></html>"""
