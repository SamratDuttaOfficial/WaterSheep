// WaterSheep demo page.
import { load, decide, ask } from "./watersheep.js";

const YESNO = ["yes", "no"];
const isDigitScale = (opts) => opts.every((o) => o.length === 1 && o >= "0" && o <= "9");
let meta = {};

const EXAMPLES = [
  ["Routing", "My card was billed twice this month. Please refund the duplicate payment.",
    "Which team should handle this?", "billing, shipping, technical support, sales", "auto"],
  ["Urgency", "The production database is down and customers cannot check out.",
    "Does this need immediate attention?", "", "noul"],
  ["Sentiment", "This is the third time I'm writing. Nobody answers and I'm about to cancel my subscription.",
    "How frustrated is the customer?", "calm, annoyed, frustrated, furious", "score"],
  ["Tagging", "The box arrived crushed, one of the mugs was broken and the invoice has the wrong address.",
    "Which problems are reported?", "damaged item, wrong address, late delivery, missing item, billing error", "multi"],
  ["Tool selection", "User: what will the weather be like in Paris tomorrow?",
    "Which tool should the assistant call?", "web_search, weather_forecast, calculator, calendar", "auto"],
];

const EXAMPLE_REQUEST = {
  state: { customer: "Priya Nair (premium plan)",
    message: "Third message this week. I was charged twice for order #4411 and the package is 12 days late. Refund me today or I will dispute the charge." },
  questions: {
    escalate: { type: "noul", instructions: "Should this be escalated to a human agent now?" },
    department: { type: "choice", instructions: "Which team should handle this?",
      criteria: { billing: "payments, refunds", shipping: "delivery, lost or damaged packages",
        technical: "bugs, login, app errors", sales: "pricing, upgrades" } },
    frustration: { type: "score", instructions: "How frustrated is the customer?",
      criteria: ["calm", "annoyed", "frustrated", "furious"] },
    issues: { type: "multi", instructions: "Which issues are reported?",
      criteria: ["double charge", "late delivery", "damaged item", "wrong item", "login problem"] },
  },
};

const TYPE_CHOICES = [["auto", "Auto"], ["noul", "Yes / No"], ["choice", "Single choice"],
  ["score", "Rating"], ["multi", "Multi-label"]];

function html(strings, ...values) {
  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  return strings.reduce((out, s, i) => out + s + (i < values.length ? esc(values[i]) : ""), "");
}

function mount(root) {
  root.classList.add("ws-demo");
  root.innerHTML = `
    <div class="ws-tabs" role="tablist">
      <button type="button" role="tab" aria-selected="true" data-tab="one">Single question</button>
      <button type="button" role="tab" aria-selected="false" data-tab="json">Batch (JSON)</button>
    </div>
    <div class="ws-status" role="status" aria-live="polite">
      <span class="ws-status-text">Runs in your browser. Data stays on your device.</span>
      <button type="button" class="ws-load">Load model</button>
      <div class="ws-progress" hidden><div class="ws-progress-bar"></div></div>
    </div>
    <div class="ws-panel" data-panel="one">
      <form class="ws-form">
        <label>Context<textarea name="state" rows="5" placeholder="Text to evaluate"></textarea></label>
        <label>Question<input name="question" placeholder="Which team should handle this?" autocomplete="off"></label>
        <label>Options<textarea name="options" rows="2" placeholder="Comma-separated. Leave empty for yes/no."></textarea></label>
        <fieldset class="ws-types"><legend>Type</legend>
          ${TYPE_CHOICES.map(([v, l], i) => html`<label class="ws-radio"><input type="radio" name="type" value="${v}"${i ? "" : " checked"}> ${l}</label>`).join("")}
        </fieldset>
        <button type="submit" class="ws-go">Run</button>
        <div class="ws-examples"><span>Examples:</span>
          ${EXAMPLES.map(([label], i) => html`<button type="button" class="ws-chip" data-example="${i}">${label}</button>`).join("")}
        </div>
      </form>
      <div class="ws-result" aria-live="polite">
        <p class="ws-empty">Results appear here.</p>
      </div>
    </div>
    <div class="ws-panel" data-panel="json" hidden>
      <p class="ws-hint">Several questions on one context. Types: <code>noul</code>, <code>choice</code>,
        <code>score</code>, <code>multi</code>.</p>
      <div class="ws-json">
        <label>Request<textarea class="ws-request" rows="20" spellcheck="false"></textarea></label>
        <div><button type="button" class="ws-go ws-ask">Run</button><pre class="ws-response" aria-live="polite"></pre></div>
      </div>
    </div>`;

  const $ = (s) => root.querySelector(s);
  const form = $(".ws-form"), f = form.elements, result = $(".ws-result"), statusText = $(".ws-status-text");
  const loadBtn = $(".ws-load"), bar = $(".ws-progress"), barFill = $(".ws-progress-bar");
  $(".ws-request").value = JSON.stringify(EXAMPLE_REQUEST, null, 2);

  root.querySelectorAll("[role=tab]").forEach((tab) => tab.addEventListener("click", () => {
    root.querySelectorAll("[role=tab]").forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
    root.querySelectorAll(".ws-panel").forEach((p) => { p.hidden = p.dataset.panel !== tab.dataset.tab; });
  }));

  function ensure() {
    loadBtn.hidden = true;
    return load({ progress: (stage, got, total) => {
      if (stage === "download") {
        bar.hidden = false;
        barFill.style.width = total ? `${(100 * got) / total}%` : "30%";
        statusText.textContent = total ? `Loading model… ${Math.floor((100 * got) / total)}%` : "Loading model…";
      } else if (stage === "start") {
        barFill.style.width = "100%";
        statusText.textContent = "Initializing…";
      } else if (stage === "ready") {
        bar.hidden = true;
        statusText.textContent = "Ready. Running locally.";
      }
    } }).then((m) => { meta = m; }).catch((e) => {
      bar.hidden = true;
      loadBtn.hidden = false;
      loadBtn.textContent = "Retry";
      statusText.textContent = `Unable to load the model: ${e.message}`;
      throw e;
    });
  }
  loadBtn.addEventListener("click", () => ensure().catch(() => {}));

  root.querySelectorAll("[data-example]").forEach((b) => b.addEventListener("click", () => {
    const [, state, question, options, type] = EXAMPLES[Number(b.dataset.example)];
    f.state.value = state;
    f.question.value = question;
    f.options.value = options;
    form.querySelector(`input[name=type][value="${type}"]`).checked = true;
    form.requestSubmit();
  }));

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const question = f.question.value.trim();
    const kind = form.querySelector("input[name=type]:checked").value;
    let opts = f.options.value.split(/[\n,]/).map((o) => o.trim()).filter(Boolean);
    if (!question) return showError("Enter a question.");
    if (kind === "noul") opts = [];
    else if (opts.length === 1 || (kind !== "auto" && opts.length < 2)) {
      return showError("Provide at least two options, or none for yes/no.");
    }
    const go = form.querySelector(".ws-go");
    go.disabled = true;
    result.innerHTML = '<p class="ws-empty">Running…</p>';
    try {
      await ensure();
      const t0 = performance.now();
      const r = await decide(f.state.value, question, opts.length ? opts : null, kind === "auto" ? null : kind);
      showResult(r, opts, performance.now() - t0);
    } catch (e) {
      showError(e.message);
    } finally {
      go.disabled = false;
    }
  });

  function showError(msg) {
    result.innerHTML = html`<p class="ws-error">${msg}</p>`;
  }

  function showResult(r, opts, ms) {
    const levels = r.type === "binary" ? YESNO : opts;
    let head;
    if (r.type === "binary") head = html`<strong>${r.answer}</strong> · P(yes) ${r.p_yes.toFixed(2)}`;
    else if (r.type === "choice") head = html`<strong>${r.answer}</strong> · confidence ${r.confidence.toFixed(2)}`;
    else if (r.type === "score") {
      head = isDigitScale(levels)
        ? html`<strong>${r.answer}</strong> · expected ${r.expected.toFixed(2)}`
        : html`<strong>${r.answer}</strong> · expected level ${r.expected.toFixed(2)} (0–${levels.length - 1})`;
    } else {
      head = r.answer.length ? html`<strong>${r.answer.join(", ")}</strong>` : html`No option above ${meta.multi_threshold ?? 0.5}`;
    }
    const kinds = { binary: "Yes / No", choice: "Single choice", score: "Rating", multi: "Multi-label" };
    const rows = Object.entries(r.probs).sort((a, b) => b[1] - a[1]).map(([o, p]) => {
      const on = r.type === "multi" ? r.answer.includes(o) : o === r.answer;
      return html`<div class="ws-row${on ? " ws-on" : ""}"><span class="ws-label">${o}</span>` +
        html`<span class="ws-track"><span class="ws-fill" style="width:${(100 * p).toFixed(1)}%"></span></span>` +
        html`<span class="ws-pct">${(100 * p).toFixed(p < 0.1 ? 1 : 0)}%</span></div>`;
    }).join("");
    result.innerHTML = `<p class="ws-answer">${head}</p>` +
      html`<p class="ws-meta">${kinds[r.type]} · ${Math.round(ms)} ms</p>` +
      `<div class="ws-bars">${rows}</div>` +
      html`<details><summary>JSON</summary><pre>${JSON.stringify(r, null, 2)}</pre></details>`;
  }

  $(".ws-ask").addEventListener("click", async () => {
    const out = $(".ws-response"), btn = $(".ws-ask");
    let req;
    try {
      req = JSON.parse($(".ws-request").value);
      if (!req || typeof req !== "object" || !req.questions || typeof req.questions !== "object") throw new Error('expected {"state": ..., "questions": {...}}');
    } catch (e) {
      out.textContent = `Error: ${e.message}`;
      return;
    }
    btn.disabled = true;
    out.textContent = "Running…";
    try {
      await ensure();
      const t0 = performance.now();
      const res = await ask(req);
      res.usage.ms = Math.round(performance.now() - t0);
      out.textContent = JSON.stringify(res, null, 2);
    } catch (e) {
      out.textContent = `Error: ${e.message}`;
    } finally {
      btn.disabled = false;
    }
  });
}

const root = document.getElementById("watersheep-demo");
if (root) mount(root);
