/* SYNTHETIC_ONLY: UI for a browser-local point-in-time forecasting demonstration. */
(function (root) {
  "use strict";
  function exportResult(data, result, state) {
    if (!result || result.scope !== "SYNTHETIC_ONLY")
      throw new Error("Run a valid synthetic evaluation before exporting.");
    return {
      schema: "commodity-forecasting-lab-export-v1",
      scope: "SYNTHETIC_ONLY",
      authored_seed: data.seed,
      data,
      result,
      review: {
        series_id: state.seriesId,
        model: state.model,
        forecast_origin: state.origin,
      },
      reproduction:
        "Use CommodityEngine.evaluate(data, result.config) from public-demo/engine.js. All data are authored and included here.",
      exclusions: [
        "No market observations",
        "No private fitted models",
        "No financial performance claim",
      ],
      numerical_scope:
        "JavaScript Number arithmetic; deterministic in a fixed runtime. Tiny floating-point differences across engines are possible.",
    };
  }
  function init(doc, win, Engine) {
    if (!Engine)
      throw new Error("The offline forecasting engine is unavailable.");
    const $ = (id) => {
      const node = doc.getElementById(id);
      if (!node) throw new Error(`Missing lab control: ${id}`);
      return node;
    };
    const el = (tag, text, cls) => {
      const node = doc.createElement(tag);
      if (text !== undefined) node.textContent = text;
      if (cls) node.className = cls;
      return node;
    };
    const listeners = [];
    const on = (node, type, handler) => {
      node.addEventListener(type, handler);
      listeners.push(() => node.removeEventListener(type, handler));
    };
    const canvas = $("forecast-canvas"),
      ctx = canvas.getContext("2d");
    if (!ctx)
      throw new Error("Canvas 2D support is needed for the forecast plot.");
    const state = {
      origin: 120,
      seriesId: "A",
      model: "ridge",
      pending: false,
    };
    let data = null,
      result = null,
      disposed = false;
    const format = (value) =>
      value === null ? "Not defined" : Number(value).toFixed(3);
    function config() {
      return {
        horizon: Number($("horizon").value),
        delay: Number($("delay").value),
        window: Number($("window").value),
        alpha: Number($("alpha").value),
        start: 80,
        end: 219,
      };
    }
    function invalidate() {
      state.pending = true;
      result = null;
      $("results").hidden = true;
      $("score-section").hidden = true;
      $("export-button").disabled = true;
      $("error-message").hidden = true;
      $("run-status").className = "run-status pending";
      $("run-status").textContent =
        "Settings changed. Fit and evaluate to produce a new matched comparison.";
    }
    function currentReceipt() {
      return result.receipts.find((r) => r.origin === state.origin);
    }
    function currentPrediction() {
      return result.predictions.find(
        (r) => r.origin === state.origin && r.series_id === state.seriesId,
      );
    }
    function draw() {
      if (!result || disposed) return;
      const rect = canvas.getBoundingClientRect(),
        width = Math.max(1, rect.width),
        height = Math.max(1, rect.height),
        dpr = Math.min(2, win.devicePixelRatio || 1);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = "#112a34";
      ctx.fillRect(0, 0, width, height);
      const series = data.series.find((s) => s.id === state.seriesId),
        rows = result.predictions.filter((r) => r.series_id === state.seriesId),
        receipt = currentReceipt();
      const left = 49,
        right = 19,
        top = 28,
        bottom = 35,
        w = width - left - right,
        h = height - top - bottom;
      const values = [
          ...series.values,
          ...rows.map((r) => r[state.model]),
          ...rows.map((r) => r.actual),
        ],
        lo = Math.min(...values) - 0.18,
        hi = Math.max(...values) + 0.18;
      const x = (day) => left + ((day - 20) / (data.days - 21)) * w,
        y = (value) => top + ((hi - value) / (hi - lo)) * h;
      ctx.fillStyle = "#28594f55";
      ctx.fillRect(
        x(receipt.first_origin),
        top,
        Math.max(0, x(receipt.last_origin + 1) - x(receipt.first_origin)),
        h,
      );
      ctx.fillStyle = "#8b713733";
      ctx.fillRect(
        x(receipt.last_origin + 1),
        top,
        x(state.origin) - x(receipt.last_origin + 1),
        h,
      );
      ctx.font = "12px ui-monospace,monospace";
      ctx.lineWidth = 0.5;
      ctx.textAlign = "right";
      for (let i = 0; i <= 4; i++) {
        const v = lo + ((hi - lo) * i) / 4,
          py = y(v);
        ctx.strokeStyle = "#35515a";
        ctx.beginPath();
        ctx.moveTo(left, py);
        ctx.lineTo(width - right, py);
        ctx.stroke();
        ctx.fillStyle = "#b7cccf";
        ctx.fillText(v.toFixed(1), left - 9, py + 4);
      }
      ctx.textAlign = "center";
      for (const day of [20, 60, 100, 140, 180, 220]) {
        const px = x(day);
        ctx.fillStyle = "#b7cccf";
        ctx.fillText(String(day), px, height - 12);
      }
      function line(points, color, lineWidth, dashed) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash(dashed ? [5, 4] : []);
        ctx.beginPath();
        points.forEach((p, i) => {
          if (i === 0) ctx.moveTo(x(p[0]), y(p[1]));
          else ctx.lineTo(x(p[0]), y(p[1]));
        });
        ctx.stroke();
        ctx.restore();
      }
      line(
        series.values.slice(20).map((v, i) => [i + 20, v]),
        "#95aaa89c",
        1,
        false,
      );
      line(
        rows.map((r) => [r.origin, r.actual]),
        "#e4be73",
        1.5,
        true,
      );
      line(
        rows.map((r) => [r.origin, r[state.model]]),
        "#73d1be",
        2.3,
        false,
      );
      ctx.strokeStyle = "#e4eade";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(x(state.origin), top);
      ctx.lineTo(x(state.origin), top + h);
      ctx.stroke();
      ctx.setLineDash([]);
      const point = currentPrediction();
      ctx.fillStyle = "#73d1be";
      ctx.strokeStyle = "#112a34";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x(state.origin), y(point[state.model]), 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      const label = `DAY ${state.origin}`;
      ctx.font = "12px ui-monospace,monospace";
      ctx.textAlign = "center";
      const px = Math.max(
        left + 37,
        Math.min(width - right - 37, x(state.origin)),
      );
      ctx.fillStyle = "#e7eadc";
      ctx.fillRect(px - 37, 5, 74, 18);
      ctx.fillStyle = "#142e36";
      ctx.fillText(label, px, 18);
      canvas.setAttribute(
        "aria-label",
        `${series.name}, ${result.models[state.model]}. At day ${state.origin}, forecast ${format(point[state.model])}, realized target ${format(point.actual)}. Latest eligible label origin ${receipt.last_origin}. Training region shown in green; withheld origins in amber.`,
      );
    }
    function renderMetrics() {
      const metric = result.metrics[state.model],
        cards = $("metric-cards");
      cards.replaceChildren();
      const rows = [
        ["Forecast RMSE", format(metric.rmse), result.models[state.model]],
        [
          "Mean absolute error",
          format(metric.mae),
          "Same matched evaluation panel",
        ],
        [
          "Forecasts evaluated",
          String(metric.count),
          `${result.origins} origins × ${data.series.length} series`,
        ],
        [
          "Release checks",
          `${result.receipts.filter((r) => r.max_release <= r.origin).length} / ${result.fit_count}`,
          "One verified boundary per fit",
        ],
      ];
      for (const [label, value, note] of rows) {
        const card = el("div", undefined, "metric");
        card.appendChild(el("span", label, "metric-label"));
        card.appendChild(el("strong", value));
        card.appendChild(el("span", note, "metric-note"));
        cards.appendChild(card);
      }
      $("evaluation-scope").textContent =
        `Origins ${result.config.start}–${result.config.end} · ${result.config.horizon}-observation mean · ${result.config.delay}-day release delay · seed ${data.seed}`;
      const body = $("comparison-body");
      body.replaceChildren();
      for (const [key, label] of Object.entries(result.models)) {
        const row = el("tr");
        row.className = key === state.model ? "active" : "";
        const name = el("td"),
          button = el("button", label);
        button.setAttribute("aria-pressed", String(key === state.model));
        button.addEventListener("click", () => setModel(key));
        name.appendChild(button);
        row.appendChild(name);
        for (const metric of ["rmse", "mae", "correlation"])
          row.appendChild(el("td", format(result.metrics[key][metric])));
        row.appendChild(el("td", String(result.metrics[key].count)));
        body.appendChild(row);
      }
    }
    function renderOrigin() {
      if (!result) return;
      const receipt = currentReceipt(),
        point = currentPrediction(),
        c = result.config,
        series = data.series.find((s) => s.id === state.seriesId),
        ledger = Engine.eligibility(data, state.origin, c, state.seriesId);
      $("origin").value = String(state.origin);
      $("origin-value").textContent = `Day ${state.origin}`;
      $("origin").setAttribute(
        "aria-valuetext",
        `Forecast at the start of day ${state.origin}`,
      );
      $("plot-title").textContent = series.name;
      $("plot-model").textContent = result.models[state.model];
      $("boundary-equation").textContent =
        `s + ${c.horizon} + ${c.delay} ≤ ${state.origin}`;
      const facts = $("boundary-facts");
      facts.replaceChildren();
      const withheld = ledger.filter((r) => !r.eligible),
        eligible = ledger.filter((r) => r.eligible);
      for (const [name, value] of [
        ["Training origins", `${receipt.first_origin}–${receipt.last_origin}`],
        ["Latest label release", `Day ${receipt.max_release}`],
        ["Eligible panel rows", String(receipt.rows)],
        ["Withheld panel rows", String(withheld.length * data.series.length)],
      ]) {
        facts.appendChild(el("dt", name));
        facts.appendChild(el("dd", value));
      }
      $("eligible-strip").style.width =
        `${(eligible.length / ledger.length) * 100}%`;
      $("withheld-strip").style.width =
        `${(withheld.length / ledger.length) * 100}%`;
      $("snapshot-prediction").textContent = format(point[state.model]);
      $("snapshot-actual").textContent = format(point.actual);
      const body = $("ledger-body");
      body.replaceChildren();
      const visible = [...eligible.slice(-3), ...withheld.slice(0, 3)];
      for (const row of visible) {
        const tr = el("tr");
        if (!row.eligible) tr.className = "locked";
        for (const text of [
          `Day ${row.origin}`,
          `Day ${row.target_end}`,
          `Day ${row.released_day}`,
        ])
          tr.appendChild(el("td", text));
        const access = el("td");
        access.appendChild(
          el("span", row.eligible ? "Available" : "Withheld", "availability"),
        );
        tr.appendChild(access);
        tr.appendChild(
          el("td", row.label === null ? "Not exposed" : format(row.label)),
        );
        body.appendChild(tr);
      }
      $("ledger-scope").textContent =
        `${series.name} at the start of day ${state.origin}. Showing the boundary rows, not the entire training panel.`;
      $("ledger-note").textContent =
        `${eligible.length} eligible origins and ${withheld.length} withheld origins in this window. ${withheld.length ? "Unreleased target values are not exposed by the snapshot accessor." : "All completed label origins in the window have been released."}`;
      $("guard-result").textContent =
        "The same release guard protects every training snapshot.";
      $("receipt-json").textContent = JSON.stringify(
        {
          scope: "SYNTHETIC_ONLY",
          model: result.models[state.model],
          config: c,
          target: result.target,
          timing_rule: result.timing_rule,
          ridge_objective:
            "Sum of squared errors + alpha × squared standardized coefficients; intercept unpenalized.",
          features: result.features,
          receipt,
          selected_forecast: point,
          note: "Coefficients and scaling describe the ridge candidate, regardless of the plotted comparison model.",
        },
        null,
        2,
      );
      draw();
    }
    function setOrigin(origin) {
      if (!result) return;
      state.origin = Math.max(
        result.config.start,
        Math.min(
          result.config.end,
          Math.round(Number(origin) || result.config.start),
        ),
      );
      renderOrigin();
    }
    function setModel(model) {
      if (!result || !Object.hasOwn(result.models, model))
        throw new Error("Unknown forecast candidate.");
      state.model = model;
      renderMetrics();
      renderOrigin();
    }
    function setSeries(id) {
      if (!data || !data.series.some((s) => s.id === id))
        throw new Error("Unknown synthetic series.");
      state.seriesId = id;
      $("series").value = id;
      renderOrigin();
    }
    function run() {
      $("run-button").disabled = true;
      $("error-message").hidden = true;
      $("run-status").className = "run-status";
      $("run-status").textContent = "Fitting chronological snapshots…";
      result = null;
      $("results").hidden = true;
      $("score-section").hidden = true;
      $("export-button").disabled = true;
      try {
        const candidate = Engine.generateData({
            seed: 2026,
            scenario: $("scenario").value,
          }),
          evaluated = Engine.evaluate(candidate, config());
        data = candidate;
        result = evaluated;
        state.pending = false;
        state.origin = Math.max(
          result.config.start,
          Math.min(result.config.end, state.origin),
        );
        const select = $("series");
        select.replaceChildren();
        for (const item of data.series) {
          const option = el("option", item.name);
          option.value = item.id;
          select.appendChild(option);
        }
        select.value = state.seriesId;
        $("origin").min = String(result.config.start);
        $("origin").max = String(result.config.end);
        $("origin-end").textContent = `Day ${result.config.end}`;
        $("results").hidden = false;
        $("score-section").hidden = false;
        $("export-button").disabled = false;
        $("run-status").textContent =
          `Complete: ${result.fit_count} real ridge fits, ${result.predictions.length} forecasts per candidate. Every fit uses only labels released by its origin.`;
        renderMetrics();
        renderOrigin();
        return result;
      } catch (error) {
        result = null;
        state.pending = true;
        $("results").hidden = true;
        $("score-section").hidden = true;
        $("export-button").disabled = true;
        $("error-message").hidden = false;
        $("error-message").textContent = error.message;
        $("run-status").textContent =
          "No results for these settings. Adjust the controls and run again.";
        return null;
      } finally {
        $("run-button").disabled = false;
      }
    }
    on($("experiment-form"), "submit", (event) => {
      event.preventDefault();
      run();
    });
    for (const id of ["scenario", "horizon", "window", "alpha"])
      on($(id), "change", invalidate);
    on($("delay"), "input", () => {
      $("delay-value").textContent = `${$("delay").value} days`;
      invalidate();
    });
    on($("series"), "change", () => setSeries($("series").value));
    on($("origin"), "input", () => setOrigin($("origin").value));
    on(canvas, "keydown", (event) => {
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault();
        setOrigin(state.origin + (event.key === "ArrowRight" ? 1 : -1));
      }
    });
    on($("guard-button"), "click", () => {
      if (!result) return;
      const next = currentReceipt().last_origin + 1;
      try {
        Engine.readLabel(
          data,
          state.seriesId,
          next,
          state.origin,
          result.config,
        );
        $("guard-result").textContent =
          "Unexpected: the future-label guard allowed this request.";
      } catch (error) {
        $("guard-result").textContent = error.message;
      }
    });
    on($("export-button"), "click", () => {
      if (!result || state.pending) return;
      const raw = JSON.stringify(exportResult(data, result, state), null, 2),
        blob = new win.Blob([raw], { type: "application/json" }),
        url = win.URL.createObjectURL(blob),
        anchor = el("a");
      anchor.href = url;
      anchor.download = "synthetic-forecast-review.json";
      doc.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      win.setTimeout(() => win.URL.revokeObjectURL(url), 1000);
    });
    on(win, "resize", draw);
    let observer = null;
    if (win.ResizeObserver) {
      observer = new win.ResizeObserver(draw);
      observer.observe(canvas);
    }
    run();
    return {
      run,
      setOrigin,
      setSeries,
      setModel,
      draw,
      getState: () => ({ ...state }),
      getResult: () => result,
      getData: () => data,
      destroy() {
        disposed = true;
        listeners.forEach((fn) => fn());
        if (observer) observer.disconnect();
      },
    };
  }
  const API = { init, exportResult };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  root.CommodityLab = API;
  if (root.document) {
    try {
      root.commodityLab = init(root.document, root, root.CommodityEngine);
    } catch (error) {
      const node = root.document.getElementById("error-message");
      if (node) {
        node.hidden = false;
        node.textContent = `The lab could not start: ${error.message}`;
      }
      if (root.console) root.console.error(error);
    }
  }
})(typeof window !== "undefined" ? window : globalThis);
