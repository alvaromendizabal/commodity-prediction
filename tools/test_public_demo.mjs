import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url),
  root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const E = require("../public-demo/engine.js"),
  App = require("../public-demo/app.js");
let count = 0;
function test(name, fn) {
  fn();
  count++;
  console.log(`PASS ${name}`);
}
const data = E.generateData(),
  config = { horizon: 5, delay: 3, window: 80, alpha: 5, start: 80, end: 219 };
const result = E.evaluate(data, config);
test("authored panel is deterministic, finite, six-series, and scenario-specific", () => {
  assert.deepEqual(E.generateData(), data);
  assert.equal(data.scope, "SYNTHETIC_ONLY");
  assert.equal(data.series.length, 6);
  E.validateData(data);
  assert.notDeepEqual(E.generateData({ scenario: "shift" }), data);
  assert.notDeepEqual(E.generateData({ seed: 2027 }), data);
  assert.deepEqual(
    E.generateData({ scenario: "shift" }).series[0].values.slice(0, 150),
    data.series[0].values.slice(0, 150),
  );
});
test("release guard rejects before and accepts exactly at publication", () => {
  const origin = 40,
    release = E.labelRelease(origin, 5, 3);
  assert.equal(release, 48);
  assert.throws(() => E.readLabel(data, "A", origin, 47, config), /Blocked/);
  assert.equal(
    E.readLabel(data, "A", origin, 48, config),
    E.targetAt(data.series[0], origin, 5),
  );
  assert.throws(() => E.targetAt(data.series[0], 1, 0), /horizon/);
});
test("future observations cannot change the current fitted model or preprocessing", () => {
  const origin = 120,
    changed = structuredClone(data);
  for (const s of changed.series)
    for (let t = origin; t < data.days; t++) s.values[t] += 10000 + t;
  const before = E.fitAt(data, origin, config),
    after = E.fitAt(changed, origin, config);
  assert.deepEqual(after, before);
  assert.notEqual(
    E.targetAt(changed.series[0], origin, 5),
    E.targetAt(data.series[0], origin, 5),
  );
});
test("withheld-but-observed labels never enter scaling, coefficients, or training means", () => {
  const fit = E.fitAt(data, 120, config),
    receipt = fit.training;
  assert.equal(receipt.last_origin, 112);
  assert.equal(receipt.max_release, 120);
  assert.equal(receipt.first_origin, 40);
  assert.equal(receipt.rows, 73 * 6);
  const xs = [];
  for (let t = 40; t <= 112; t++)
    for (let j = 0; j < 6; j++) xs.push(E.featureAt(data.series[j], j, t));
  for (let j = 0; j < E.FEATURES.length; j++) {
    const average = xs.reduce((s, x) => s + x[j], 0) / xs.length;
    assert(Math.abs(average - receipt.feature_mean[j]) < 1e-12);
  }
  const ledger = E.eligibility(data, 120, config, "A");
  assert(ledger.filter((r) => !r.eligible).every((r) => r.label === null));
  assert.equal(ledger.filter((r) => r.eligible).length, 73);
  assert.equal(ledger.filter((r) => !r.eligible).length, 7);
});
test("all forward origins use matched evaluation scope and valid timing receipts", () => {
  assert.equal(result.fit_count, 140);
  assert.equal(result.predictions.length, 840);
  assert(
    result.receipts.every(
      (r) => r.max_release <= r.origin && r.last_origin < r.origin,
    ),
  );
  for (const metrics of Object.values(result.metrics)) {
    assert.equal(metrics.count, 840);
    assert(Number.isFinite(metrics.rmse));
    assert(Number.isFinite(metrics.mae));
  }
  assert.equal(result.metrics.zero.correlation, null);
  for (const row of result.predictions)
    assert.equal(
      row.actual,
      E.targetAt(
        data.series.find((s) => s.id === row.series_id),
        row.origin,
        5,
      ),
    );
});
test("later data mutations leave earlier evaluation predictions intact", () => {
  const changed = structuredClone(data);
  for (const s of changed.series)
    for (let t = 150; t < 240; t++) s.values[t] += 5;
  const later = E.evaluate(changed, { ...config, end: 160 });
  for (const row of later.predictions.filter((r) => r.origin <= 150)) {
    const original = result.predictions.find(
      (p) => p.origin === row.origin && p.series_id === row.series_id,
    );
    assert.equal(row.ridge, original.ridge);
    assert.equal(row.mean, original.mean);
  }
  assert.notEqual(
    later.predictions.find((r) => r.origin === 150).actual,
    result.predictions.find((r) => r.origin === 150).actual,
  );
});
test("ridge learns an authored linear signal and constant target without numerical failure", () => {
  const linear = structuredClone(data);
  for (let j = 0; j < 6; j++)
    linear.series[j].values = Array.from(
      { length: 240 },
      (_, t) => t * 0.01 + j * 0.1,
    );
  const fit = E.fitAt(linear, 120, { ...config, alpha: 0.01 });
  fit.predictions.forEach((row, j) =>
    assert(Math.abs(row.ridge - E.targetAt(linear.series[j], 120, 5)) < 0.002),
  );
  const constant = structuredClone(data);
  for (const s of constant.series) s.values.fill(2);
  const fixed = E.fitAt(constant, 120, config);
  assert(
    fixed.predictions.every(
      (p) => Math.abs(p.ridge - 2) < 1e-12 && p.mean === 2,
    ),
  );
});
test("score is calculated from forecasts, not a hard-coded claim", () => {
  const rows = [
    { actual: 1, p: 2 },
    { actual: 3, p: 1 },
  ];
  const metrics = E.score(rows, "p");
  assert.equal(metrics.rmse, Math.sqrt(2.5));
  assert.equal(metrics.mae, 1.5);
  assert.equal(metrics.correlation, -1);
  const fresh = E.evaluate(data, config);
  assert.deepEqual(fresh, result);
});
test("invalid, missing, and insufficient training inputs fail explicitly", () => {
  assert.throws(() => E.evaluate(data, { ...config, horizon: 2 }), /supported/);
  assert.throws(() => E.evaluate(data, { ...config, delay: -1 }), /supported/);
  assert.throws(
    () => E.evaluate(data, { ...config, start: 90, end: 80 }),
    /chronological/,
  );
  assert.throws(
    () => E.evaluate(data, { ...config, window: 40, horizon: 20, delay: 20 }),
    /Too few released/,
  );
  const broken = structuredClone(data);
  broken.series[0].values[3] = NaN;
  assert.throws(() => E.evaluate(broken, config), /finite/);
  const empty = structuredClone(data);
  empty.series = [];
  assert.throws(() => E.evaluate(empty, config), /six-series/);
});
class Element {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.style = {};
    this.attributes = {};
    this.listeners = new Map();
    this.value = "";
    this.hidden = false;
    this.disabled = false;
    this.className = "";
    this._text = "";
  }
  set textContent(v) {
    this._text = String(v);
    this.children = [];
  }
  get textContent() {
    return this._text + this.children.map((e) => e.textContent).join("");
  }
  appendChild(e) {
    e.parentNode = this;
    this.children.push(e);
    return e;
  }
  replaceChildren(...children) {
    this._text = "";
    this.children = [];
    children.forEach((e) => this.appendChild(e));
  }
  setAttribute(k, v) {
    this.attributes[k] = String(v);
  }
  getAttribute(k) {
    return this.attributes[k];
  }
  addEventListener(t, fn) {
    if (!this.listeners.has(t)) this.listeners.set(t, new Set());
    this.listeners.get(t).add(fn);
  }
  removeEventListener(t, fn) {
    this.listeners.get(t)?.delete(fn);
  }
  fire(t, extra = {}) {
    const e = {
      target: this,
      preventDefault() {
        this.defaultPrevented = true;
      },
      ...extra,
    };
    for (const fn of this.listeners.get(t) || []) fn(e);
    return e;
  }
  click() {
    this.fire("click");
  }
  remove() {
    if (this.parentNode)
      this.parentNode.children = this.parentNode.children.filter(
        (e) => e !== this,
      );
  }
  getBoundingClientRect() {
    return { width: 780, height: 330, left: 0, top: 0 };
  }
}
function fakeDOM() {
  const html = fs.readFileSync(
      path.join(root, "public-demo/index.html"),
      "utf8",
    ),
    nodes = new Map();
  for (const m of html.matchAll(/<([\w-]+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const el = new Element(m[1]);
    el.hidden = /\bhidden\b/.test(m[2]);
    el.disabled = /\bdisabled\b/.test(m[2]);
    const value = m[2].match(/\bvalue="([^"]*)"/);
    if (value) el.value = value[1];
    nodes.set(m[3], el);
  }
  const calls = [],
    context = new Proxy(
      {},
      {
        get(target, k) {
          if (k in target) return target[k];
          return (...args) => calls.push([k, ...args]);
        },
        set(target, k, v) {
          target[k] = v;
          return true;
        },
      },
    );
  nodes.get("forecast-canvas").getContext = () => context;
  for (const [id, value] of Object.entries({
    scenario: "steady",
    horizon: "5",
    delay: "3",
    window: "80",
    alpha: "5",
    origin: "120",
  }))
    nodes.get(id).value = value;
  const doc = new Element("document");
  doc.body = new Element("body");
  doc.getElementById = (id) => nodes.get(id);
  doc.createElement = (tag) => new Element(tag);
  const win = new Element("window"),
    downloads = [],
    revoked = [];
  win.devicePixelRatio = 1;
  win.Blob = Blob;
  win.URL = {
    createObjectURL(blob) {
      downloads.push(blob);
      return "blob:test";
    },
    revokeObjectURL(url) {
      revoked.push(url);
    },
  };
  win.setTimeout = (fn) => fn();
  win.ResizeObserver = class {
    observe() {}
    disconnect() {}
  };
  return { html, nodes, doc, win, calls, downloads, revoked };
}
const env = fakeDOM(),
  ui = App.init(env.doc, env.win, E),
  el = (id) => env.nodes.get(id);
test("production UI initializes real fits, chart, ledger, and candidate metrics", () => {
  assert.equal(ui.getResult().fit_count, 140);
  assert.equal(el("results").hidden, false);
  assert.equal(el("score-section").hidden, false);
  assert.equal(el("metric-cards").children.length, 4);
  assert.equal(el("comparison-body").children.length, 3);
  assert.equal(el("series").children.length, 6);
  assert.match(el("boundary-equation").textContent, /s \+ 5 \+ 3 ≤ 120/);
  assert.match(el("ledger-body").textContent, /Not exposed/);
  assert(env.calls.some((c) => c[0] === "lineTo"));
  assert.match(
    el("forecast-canvas").getAttribute("aria-label"),
    /Latest eligible label origin 112/,
  );
});
test("actual series, origin, keyboard, and candidate handlers update inspected result", () => {
  el("series").value = "F";
  el("series").fire("change");
  assert.equal(ui.getState().seriesId, "F");
  assert.equal(el("plot-title").textContent, "Crop F");
  el("origin").value = "150";
  el("origin").fire("input");
  assert.equal(ui.getState().origin, 150);
  const event = el("forecast-canvas").fire("keydown", { key: "ArrowLeft" });
  assert.equal(event.defaultPrevented, true);
  assert.equal(ui.getState().origin, 149);
  el("comparison-body").children[1].children[0].children[0].click();
  assert.equal(ui.getState().model, "mean");
  assert.equal(el("plot-model").textContent, "Released-label mean");
  ui.setOrigin(999);
  assert.equal(ui.getState().origin, 219);
  ui.setOrigin(-99);
  assert.equal(ui.getState().origin, 80);
});
test("future-label button runs the engine guard and announces rejection", () => {
  el("guard-button").click();
  assert.match(el("guard-result").textContent, /Blocked: origin/);
  assert.match(el("guard-result").textContent, /after snapshot day 80/);
});
test("changed settings immediately remove stale scores and disable export", () => {
  el("horizon").value = "20";
  el("horizon").fire("change");
  assert.equal(ui.getResult(), null);
  assert.equal(el("results").hidden, true);
  assert.equal(el("score-section").hidden, true);
  assert.equal(el("export-button").disabled, true);
  el("export-button").click();
  assert.equal(env.downloads.length, 0);
  assert.match(el("run-status").textContent, /Settings changed/);
});
test("invalid combinations show a visible error with no stale metrics or export", () => {
  el("window").value = "40";
  el("window").fire("change");
  el("delay").value = "20";
  el("delay").fire("input");
  assert.equal(el("delay-value").textContent, "20 days");
  const event = el("experiment-form").fire("submit");
  assert.equal(event.defaultPrevented, true);
  assert.equal(ui.getResult(), null);
  assert.equal(el("error-message").hidden, false);
  assert.match(el("error-message").textContent, /Too few released/);
  assert.equal(el("score-section").hidden, true);
  assert.equal(el("export-button").disabled, true);
});
test("valid rerun recovers and all candidates use the new matched horizon", () => {
  el("window").value = "120";
  el("window").fire("change");
  el("scenario").value = "shift";
  el("scenario").fire("change");
  el("experiment-form").fire("submit");
  assert(ui.getResult());
  assert.equal(ui.getResult().config.horizon, 20);
  assert.equal(ui.getResult().config.delay, 20);
  assert.equal(ui.getData().scenario, "shift");
  assert.equal(el("error-message").hidden, true);
  assert.equal(el("export-button").disabled, false);
  assert.equal(ui.getResult().metrics.ridge.count, 840);
  assert.match(el("evaluation-scope").textContent, /20-observation mean/);
});
test("export includes exact authored observations, receipts and current numerical result", () => {
  const exported = App.exportResult(
    ui.getData(),
    ui.getResult(),
    ui.getState(),
  );
  assert.equal(exported.scope, "SYNTHETIC_ONLY");
  assert.deepEqual(exported.result, ui.getResult());
  assert.equal(exported.data.series.length, 6);
  el("export-button").click();
  assert.equal(env.downloads.length, 1);
  assert.equal(env.revoked.length, 1);
  assert.throws(() => App.exportResult(data, null, {}), /valid/);
});
test("offline scripts, accessible controls, and minimum text/hit styles are explicit", () => {
  assert(!/<script[^>]+src="https?:/i.test(env.html));
  assert(
    env.html.indexOf('src="engine.js"') < env.html.indexOf('src="app.js"'),
  );
  assert.match(env.html, /SYNTHETIC_ONLY/);
  assert.match(env.html, /role="alert"/);
  assert.match(env.html, /aria-live="polite"/);
  const css = fs.readFileSync(
    path.join(root, "public-demo/styles.css"),
    "utf8",
  );
  assert(!/font-size:\s*(?:[0-9]|1[01])px\b/.test(css));
  assert.match(css, /min-height:\s*24px/);
  const app = fs.readFileSync(path.join(root, "public-demo/app.js"), "utf8");
  assert(!app.includes(".innerHTML"));
  assert(!app.includes("fetch("));
});
test("dispose removes persistent input handlers", () => {
  const before = ui.getState().origin;
  ui.destroy();
  el("forecast-canvas").fire("keydown", { key: "ArrowRight" });
  assert.equal(ui.getState().origin, before);
});
const payload = JSON.parse(await env.downloads[0].text());
assert.equal(payload.result.config.horizon, 20);
assert.equal(payload.result.predictions.length, 840);
assert.equal(payload.data.scenario, "shift");
console.log(
  `PASS exported JSON bytes match the final run\n${count + 1} public demo tests passed.`,
);
