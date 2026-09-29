// WaterSheep in the browser (onnxruntime-web).
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
const TOKENIZERS = "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.min.mjs";
const ONNX_FILE = "onnx/model_quantized.onnx";
const CACHE = "watersheep-models";
export const MODEL = "samratduttaofficial/WaterSheep";

const YESNO = ["yes", "no"];
const TYPE_TAG = { binary: "[yes/no]", choice: "[choose]", score: "[rate]", multi: "[select all]" };
const TYPES_IN = { noul: "binary", binary: "binary", "yes/no": "binary", boolean: "binary", choice: "choice",
  score: "score", multi: "multi", multi_choice: "multi", multilabel: "multi", "multi-label": "multi" };
const MAX_CHOICES = 255, MAX_LEVELS = 10;

let ort, tok, meta, session, special, loading = null, queue = Promise.resolve();

async function locate(model, base) {
  if (base) return { url: new URL(base.endsWith("/") ? base : base + "/", location.href).href, key: null };
  try {
    const r = await fetch(`https://huggingface.co/api/models/${model}/revision/main?blobs=true`);
    if (r.ok) {
      const info = await r.json();
      const f = (info.siblings || []).find((x) => x.rfilename === ONNX_FILE);
      return { url: `https://huggingface.co/${model}/resolve/${info.sha}/`, key: (f && f.lfs && f.lfs.sha256) || info.sha };
    }
  } catch (e) { /* fall back to main */ }
  return { url: `https://huggingface.co/${model}/resolve/main/`, key: null };
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`could not download ${url.split("/").pop()} (${r.status})`);
  return r.json();
}

async function fetchModel(model, url, key, progress) {
  let cache = null;
  const id = `https://huggingface.co/${model}/${ONNX_FILE}?v=${key}`;
  if (key) {
    try {
      cache = await caches.open(CACHE);
      const hit = await cache.match(id);
      if (hit) return new Uint8Array(await hit.arrayBuffer());
    } catch (e) { cache = null; }
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`could not download the model (${r.status})`);
  const total = Number(r.headers.get("Content-Length")) || 0;
  const reader = r.body.getReader(), parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    progress("download", got, total);
  }
  const bytes = new Uint8Array(got);
  let at = 0;
  for (const p of parts) { bytes.set(p, at); at += p.length; }
  if (cache) {
    try {
      for (const k of await cache.keys()) if (k.url !== id && k.url.includes(`${model}/${ONNX_FILE}`)) await cache.delete(k);
      await cache.put(id, new Response(bytes));
    } catch (e) { /* not cached */ }
  }
  return bytes;
}

/** Loads the model once. Options: model (Hugging Face repo id), base (URL of a local copy), progress. */
export function load({ model = MODEL, base = null, progress = () => {} } = {}) {
  if (!loading) {
    loading = (async () => {
      const [b, ortModule, tokModule] = await Promise.all([locate(model, base), import(ORT + "ort.wasm.min.mjs"), import(TOKENIZERS)]);
      ort = ortModule;
      ort.env.wasm.wasmPaths = ORT;
      ort.env.wasm.proxy = true;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      const [m, tj, tc] = await Promise.all([fetchJSON(b.url + "watersheep.json"),
        fetchJSON(b.url + "tokenizer/tokenizer.json"), fetchJSON(b.url + "tokenizer/tokenizer_config.json")]);
      meta = m;
      tok = new tokModule.Tokenizer(tj, tc);
      special = {};
      for (const k of ["cls", "sep", "mask", "pad"]) {
        const t = tc[k + "_token"];
        special[k] = tok.token_to_id(t && typeof t === "object" ? t.content : t);
      }
      if (special.cls === undefined || special.sep === undefined || special.mask === undefined) {
        throw new Error("the tokenizer has no CLS/SEP/MASK tokens");
      }
      const bytes = await fetchModel(model, b.url + ONNX_FILE, b.key, progress);
      progress("start");
      session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      progress("ready");
      return meta;
    })().catch((e) => { loading = null; throw e; });
  }
  return loading;
}

const encode = (text) => tok.encode(text || "", { add_special_tokens: false }).ids;

function encodeRecord(r) {
  const m = meta, { cls, sep, mask } = special;
  const q = encode(TYPE_TAG[r.type] + " " + r.question).slice(0, m.max_question_tokens);
  const s = encode(r.state || "");
  let ids = [cls, ...q, sep], pos = [];
  for (const o of r.options) {
    pos.push(ids.length);
    ids.push(mask, ...encode(o).slice(0, m.max_option_tokens));
  }
  ids.push(sep);
  const room = m.max_len - ids.length - 1;
  if (room > 0 && s.length) ids.push(...s.slice(0, room));
  ids.push(sep);
  if (ids.length > m.max_len) {
    ids = [...ids.slice(0, m.max_len - 1), sep];
    pos = pos.filter((p) => p < m.max_len - 1);
  }
  return [ids, pos];
}

function run(ids, pos) {
  const i64 = (a) => BigInt64Array.from(a, (x) => BigInt(x));
  const feeds = {
    input_ids: new ort.Tensor("int64", i64(ids), [1, ids.length]),
    attention_mask: new ort.Tensor("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
    option_positions: new ort.Tensor("int64", i64(pos), [1, pos.length]),
    option_mask: new ort.Tensor("int64", new BigInt64Array(pos.length).fill(1n), [1, pos.length]),
  };
  const job = queue.then(() => session.run(feeds));
  queue = job.catch(() => {});
  return job.then((out) => Array.from(out.logits.data));
}

async function logits(r) {
  const [ids, pos] = encodeRecord(r);
  if (pos.length !== r.options.length) return { z: null, tokens: ids.length };
  return { z: await run(ids, pos), tokens: ids.length };
}

const softmax = (z, T) => {
  const x = z.map((v) => v / Math.max(1e-6, T)), mx = Math.max(...x), e = x.map((v) => Math.exp(v - mx));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};
const sigmoid = (z, T) => z.map((v) => 1 / (1 + Math.exp(-Math.min(60, Math.max(-60, v / Math.max(1e-6, T))))));
const argmax = (p) => p.reduce((best, v, i) => (v > p[best] ? i : best), 0);
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const concentration = (p) => {
  if (p.length < 2) return 1;
  const h = p.reduce((a, v) => { const c = Math.min(1, Math.max(1e-12, v)); return a + c * Math.log(c); }, 0);
  return Math.max(0, 1 + h / Math.log(p.length));
};
const isDigitScale = (opts) => opts.every((o) => o.length === 1 && o >= "0" && o <= "9");

async function probs(t, state, question, opts, size = 0) {
  const T = (meta.temperatures || {})[t] ?? 1.0;
  const cap = Math.max(2, Math.min(10, Number(meta.max_options) || 10));
  const n = opts.length;
  size = Math.min(size || cap, n);
  let tokens = 0, groups, zs;
  for (;;) {
    if (n <= size) {
      const r = await logits({ type: t, state, question, options: opts });
      tokens += r.tokens;
      if (r.z) return { p: t === "multi" ? sigmoid(r.z, T) : softmax(r.z, T), tokens };
    } else {
      groups = [];
      for (let k = 0; k < n; k += size) groups.push(Array.from({ length: Math.min(n, k + size) - k }, (_, j) => k + j));
      zs = [];
      for (const g of groups) {
        const r = await logits({ type: t, state, question, options: g.map((i) => opts[i]) });
        tokens += r.tokens;
        zs.push(r.z);
      }
      if (zs.every((z) => z)) break;
    }
    if (size <= 2) throw new Error("options are too long");
    size = Math.max(2, size > 8 ? Math.min(size - 1, 8) : Math.floor(size / 2));
  }
  if (t === "multi") {
    const p = new Array(n).fill(0);
    groups.forEach((g, gi) => sigmoid(zs[gi], T).forEach((v, k) => { p[g[k]] = v; }));
    return { p, tokens };
  }
  const keep = Math.max(1, Math.floor(size / groups.length)), local = {}, finalists = [];
  groups.forEach((g, gi) => {
    const pg = softmax(zs[gi], T);
    g.forEach((i, k) => { local[i] = pg[k]; });
    finalists.push(...pg.map((v, k) => [v, k]).sort((a, b) => b[0] - a[0]).slice(0, keep).map(([, k]) => g[k]));
  });
  const f = await probs(t, state, question, finalists.map((i) => opts[i]), size);
  const p = new Array(n).fill(0);
  for (const g of groups) {
    const top = g.filter((i) => finalists.includes(i)).reduce((a, b) => (local[b] > local[a] ? b : a));
    const scale = f.p[finalists.indexOf(top)] / Math.max(1e-12, local[top]);
    for (const i of g) p[i] = local[i] * scale;
  }
  finalists.forEach((i, k) => { p[i] = f.p[k]; });
  const sum = p.reduce((a, b) => a + b, 0);
  return { p: p.map((v) => v / sum), tokens: tokens + f.tokens };
}

function inferType(options) {
  if (!options || !options.length) return "binary";
  const low = options.map((o) => String(o).trim().toLowerCase());
  if (low.length === 2 && low[0] === "yes" && low[1] === "no") return "binary";
  if (isDigitScale(low)) {
    const v = low.map(Number);
    if (v.every((x, i) => x === v[0] + i)) return "score";
  }
  return "choice";
}

/** One question: {type, answer, confidence, probs, ...}. */
export async function decide(state, question, options = null, type = null) {
  await load();
  let opts = (options && options.length ? options : YESNO).map(String);
  const t = TYPES_IN[String(type || "").toLowerCase()] || inferType(opts);
  if (t === "binary") opts = [...YESNO];
  const { p } = await probs(t, state || "", question, opts);
  const res = { type: t, probs: Object.fromEntries(opts.map((o, i) => [o, p[i]])) };
  if (t === "multi") {
    res.answer = opts.filter((o, i) => p[i] >= (meta.multi_threshold ?? 0.5));
    res.confidence = mean(p.map((v) => Math.max(v, 1 - v)));
  } else {
    const i = argmax(p);
    Object.assign(res, { answer: opts[i], index: i, confidence: p[i] });
  }
  if (t === "binary") res.p_yes = p[0];
  else if (t === "score") res.expected = p.reduce((a, v, k) => a + k * v, 0) + (isDigitScale(opts) ? Number(opts[0]) : 0);
  return res;
}

function pyRepr(v) {
  return v === null || v === undefined ? "None" : v === true ? "True" : v === false ? "False" : String(v);
}

function pyDumps(v) {
  if (Array.isArray(v)) return "[" + v.map(pyDumps).join(", ") + "]";
  if (v && typeof v === "object") return "{" + Object.entries(v).map(([k, x]) => JSON.stringify(k) + ": " + pyDumps(x)).join(", ") + "}";
  return JSON.stringify(v ?? null);
}

function renderState(state) {
  if (state === null || state === undefined) return "";
  if (typeof state === "string") return state;
  if (Array.isArray(state)) return state.map(renderState).join("\n");
  if (typeof state === "object") {
    return Object.entries(state).map(([k, v]) =>
      `${k}: ${v === null || ["string", "number", "boolean"].includes(typeof v) ? pyRepr(v) : pyDumps(v)}`).join("\n");
  }
  return String(state);
}

const r4 = (x) => Math.round(x * 1e4) / 1e4;

function labelsOf(criteria) {
  if (criteria && typeof criteria === "object" && !Array.isArray(criteria)) {
    return Object.entries(criteria).map(([k, v]) => [k, String(v || "").trim() ? `${k}: ${v}` : k]);
  }
  return Array.from(criteria || [], (k) => [String(k), String(k)]);
}

async function answer(state, q) {
  const t = TYPES_IN[String(q.type ?? "").toLowerCase()];
  if (!t) {
    const shown = typeof q.type === "string" ? `'${q.type}'` : pyRepr(q.type);
    throw new Error(`unknown question type ${shown} (noul, choice, score or multi)`);
  }
  const instr = String(q.instructions || q.question || "").trim();
  if (!instr) throw new Error("instructions are required");
  const crit = q.criteria;
  if (t === "binary") {
    const c = crit && typeof crit === "object" && !Array.isArray(crit) ? crit : {};
    const pick = (a, b) => String(((a in c) ? c[a] : c[b] ?? "") || "");
    const extra = [["yes", pick("true", "yes")], ["no", pick("false", "no")]]
      .filter(([, v]) => v.trim()).map(([k, v]) => `\n${k} = ${v.trim()}`).join("");
    const { p, tokens } = await probs("binary", state, instr.replace(/\s+$/, "") + extra, [...YESNO]);
    return [{ type: "noul", noul: r4(p[0]) }, tokens];
  }
  if (t === "score") {
    const levels = (Array.isArray(crit) || typeof crit === "string" ? Array.from(crit)
      : crit && typeof crit === "object" ? Object.values(crit) : []).map(String);
    if (levels.length < 2 || levels.length > MAX_LEVELS) throw new Error(`score needs 2-${MAX_LEVELS} levels in criteria`);
    const { p, tokens } = await probs("score", state, instr, levels);
    return [{ type: "score", score: r4(p.reduce((a, v, i) => a + i * v, 0)), confidence: r4(concentration(p)),
      legend: Object.fromEntries(levels.map((l, i) => [String(i), l])),
      probabilities: Object.fromEntries(p.map((v, i) => [String(i), r4(v)])) }, tokens];
  }
  const labels = labelsOf(crit);
  if (labels.length < 2 || labels.length > MAX_CHOICES) throw new Error(`${t} needs 2-${MAX_CHOICES} labels in criteria`);
  if (new Set(labels.map(([k]) => k)).size !== labels.length) throw new Error("labels must be unique");
  const { p, tokens } = await probs(t, state, instr, labels.map(([, text]) => text));
  const probsOut = Object.fromEntries(labels.map(([k], i) => [k, r4(p[i])]));
  if (t === "multi") {
    const thr = Number(q.threshold ?? meta.multi_threshold ?? 0.5);
    return [{ type: "multi", choices: Object.keys(probsOut).filter((k) => probsOut[k] >= thr),
      confidence: r4(mean(p.map((v) => Math.max(v, 1 - v)))), probabilities: probsOut }, tokens];
  }
  const best = Object.keys(probsOut).reduce((a, b) => (probsOut[b] > probsOut[a] ? b : a));
  return [{ type: "choice", choice: best, confidence: r4(concentration(p)), probabilities: probsOut }, tokens];
}

/** Several questions about one text: {state, questions} -> {model, answers, usage}. */
export async function ask(request) {
  await load();
  const state = renderState(request.state);
  const answers = {};
  let tokens = 0;
  for (const [name, q] of Object.entries(request.questions || {})) {
    try {
      const [a, n] = await answer(state, q && typeof q === "object" ? q : {});
      answers[name] = a;
      tokens += n;
    } catch (e) {
      answers[name] = { type: q && typeof q === "object" ? String(q.type ?? "?") : "?", error: e.message };
    }
  }
  return { model: meta.name || "watersheep", answers, usage: { input_tokens: tokens, output_tokens: 0 } };
}
