/* Authored synthetic forecasting example. No market data or financial evidence. */
(function (root) {
  "use strict";
  const FEATURES = [
    "Last observation",
    "Mean of last 5",
    "Mean of last 20",
    "Calendar sine",
    "Calendar cosine",
    "Series B",
    "Series C",
    "Series D",
    "Series E",
    "Series F",
  ];
  const MODELS = {
    ridge: "Regularized lag model",
    mean: "Released-label mean",
    zero: "Zero signal",
  };
  const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;
  function generateData({ seed = 2026, scenario = "steady", days = 240 } = {}) {
    if (
      !Number.isInteger(seed) ||
      seed < 0 ||
      !["steady", "shift", "noisy"].includes(scenario) ||
      !Number.isInteger(days) ||
      days < 100 ||
      days > 1000
    )
      throw new Error("Invalid synthetic generator settings.");
    let state = seed >>> 0;
    const random = () => {
      state = (Math.imul(1664525, state) + 1013904223) >>> 0;
      return state / 4294967296;
    };
    const names = [
      "Alloy A",
      "Energy B",
      "Grain C",
      "Metal D",
      "Fuel E",
      "Crop F",
    ];
    const colors = [
      "#68c8bd",
      "#e4be73",
      "#b2b5ec",
      "#eea78f",
      "#a9cc84",
      "#dfaccf",
    ];
    const series = names.map((name, j) => {
      const values = [];
      let previous = (random() - 0.5) * 0.5;
      for (let t = 0; t < days; t++) {
        const noise =
          (random() + random() + random() - 1.5) *
          (scenario === "noisy" ? 0.7 : 0.22);
        const cycle =
          0.15 * Math.sin(t / 8 + j * 0.47) +
          0.08 * Math.cos(t / 19 - j * 0.33);
        const shift =
          scenario === "shift" && t >= 150 ? (j % 2 === 0 ? 0.22 : -0.18) : 0;
        previous = 0.78 * previous + cycle + noise + shift;
        values.push(previous);
      }
      return {
        id: String.fromCharCode(65 + j),
        name,
        color: colors[j],
        values,
      };
    });
    return {
      schema: "commodity-authored-panel-v1",
      scope: "SYNTHETIC_ONLY",
      seed,
      scenario,
      days,
      series,
      description:
        "Six fictional signal series in arbitrary units. Authored cycles, persistence and noise; no market observations.",
    };
  }
  function validateData(data) {
    if (
      !data ||
      data.scope !== "SYNTHETIC_ONLY" ||
      !Number.isInteger(data.days) ||
      !Array.isArray(data.series) ||
      data.series.length !== 6
    )
      throw new Error("Expected the six-series authored synthetic panel.");
    if (new Set(data.series.map((s) => s.id)).size !== 6)
      throw new Error("Series identifiers must be unique.");
    for (const series of data.series)
      if (
        !Array.isArray(series.values) ||
        series.values.length !== data.days ||
        series.values.some((x) => typeof x !== "number" || !Number.isFinite(x))
      )
        throw new Error(
          "Every synthetic observation must be finite and aligned.",
        );
  }
  function validateConfig(config, data) {
    const c = {
      horizon: 5,
      delay: 3,
      window: 80,
      alpha: 5,
      start: 80,
      end: 219,
      ...config,
    };
    if (
      ![1, 5, 10, 20].includes(c.horizon) ||
      !Number.isInteger(c.delay) ||
      c.delay < 0 ||
      c.delay > 20 ||
      ![40, 80, 120].includes(c.window) ||
      !Number.isFinite(c.alpha) ||
      c.alpha < 0.01 ||
      c.alpha > 100
    )
      throw new Error(
        "Use a supported horizon, release delay, window, and ridge penalty.",
      );
    if (
      !Number.isInteger(c.start) ||
      !Number.isInteger(c.end) ||
      c.start < 20 ||
      c.end >= data.days ||
      c.start > c.end ||
      c.end + c.horizon > data.days
    )
      throw new Error(
        "Evaluation origins must be chronological and have complete synthetic targets.",
      );
    return c;
  }
  function featureAt(series, index, origin) {
    if (
      !Number.isInteger(origin) ||
      origin < 20 ||
      origin > series.values.length
    )
      throw new Error("A feature requires 20 completed observations.");
    const past = series.values;
    return [
      past[origin - 1],
      mean(past.slice(origin - 5, origin)),
      mean(past.slice(origin - 20, origin)),
      Math.sin(origin / 8),
      Math.cos(origin / 8),
      ...Array.from({ length: 5 }, (_, j) => Number(index === j + 1)),
    ];
  }
  function targetAt(series, origin, horizon) {
    if (
      !Number.isInteger(origin) ||
      !Number.isInteger(horizon) ||
      horizon < 1 ||
      origin < 0 ||
      origin + horizon > series.values.length
    )
      throw new Error("Target horizon exceeds the authored panel.");
    return mean(series.values.slice(origin, origin + horizon));
  }
  const labelRelease = (origin, horizon, delay) => origin + horizon + delay;
  function releasedTarget(series, labelOrigin, asOf, c) {
    const release = labelRelease(labelOrigin, c.horizon, c.delay);
    if (release > asOf)
      throw new Error(
        `Blocked: origin ${labelOrigin} is released on day ${release}, after snapshot day ${asOf}.`,
      );
    return targetAt(series, labelOrigin, c.horizon);
  }
  function readLabel(data, seriesId, labelOrigin, asOf, config = {}) {
    const c = validateConfig(config, data),
      series = data.series.find((s) => s.id === seriesId);
    if (!series || !Number.isInteger(asOf) || asOf < 0)
      throw new Error("Invalid label request.");
    return releasedTarget(series, labelOrigin, asOf, c);
  }
  function solveLinear(matrix, vector) {
    const n = vector.length,
      a = matrix.map((row, i) => [...row, vector[i]]);
    for (let k = 0; k < n; k++) {
      let pivot = k;
      for (let i = k + 1; i < n; i++)
        if (Math.abs(a[i][k]) > Math.abs(a[pivot][k])) pivot = i;
      if (Math.abs(a[pivot][k]) < 1e-12)
        throw new Error("The small ridge system is singular.");
      [a[k], a[pivot]] = [a[pivot], a[k]];
      const v = a[k][k];
      for (let j = k; j <= n; j++) a[k][j] /= v;
      for (let i = 0; i < n; i++)
        if (i !== k) {
          const factor = a[i][k];
          for (let j = k; j <= n; j++) a[i][j] -= factor * a[k][j];
        }
    }
    return a.map((row) => row[n]);
  }
  function fitAt(data, origin, config = {}) {
    const c = validateConfig(config, data);
    if (
      !Number.isInteger(origin) ||
      origin < 20 ||
      origin + c.horizon > data.days
    )
      throw new Error("Invalid forecast origin.");
    const first = Math.max(20, origin - c.window),
      last = origin - c.horizon - c.delay;
    const rows = [];
    for (let t = first; t <= last; t++)
      for (let j = 0; j < data.series.length; j++)
        rows.push({
          origin: t,
          index: j,
          x: featureAt(data.series[j], j, t),
          y: releasedTarget(data.series[j], t, origin, c),
          release: labelRelease(t, c.horizon, c.delay),
        });
    if (rows.length < 18)
      throw new Error(
        "Too few released labels in this training window. Reduce the horizon or publication delay, or use a longer window.",
      );
    const p = FEATURES.length,
      mu = Array(p).fill(0),
      scale = Array(p).fill(0),
      yMean = mean(rows.map((r) => r.y));
    for (const row of rows)
      for (let j = 0; j < p; j++) mu[j] += row.x[j] / rows.length;
    for (const row of rows)
      for (let j = 0; j < p; j++)
        scale[j] += (row.x[j] - mu[j]) ** 2 / rows.length;
    for (let j = 0; j < p; j++) scale[j] = Math.sqrt(scale[j]) || 1;
    const gram = Array.from({ length: p }, () => Array(p).fill(0)),
      rhs = Array(p).fill(0);
    for (const row of rows) {
      const x = row.x.map((v, j) => (v - mu[j]) / scale[j]);
      for (let i = 0; i < p; i++) {
        rhs[i] += x[i] * (row.y - yMean);
        for (let j = 0; j <= i; j++) gram[i][j] += x[i] * x[j];
      }
    }
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < i; j++) gram[j][i] = gram[i][j];
      gram[i][i] += c.alpha;
    }
    const beta = solveLinear(gram, rhs);
    const predictions = data.series.map((series, index) => {
      const x = featureAt(series, index, origin),
        past = rows.filter((row) => row.index === index);
      return {
        series_id: series.id,
        ridge:
          yMean +
          x.reduce((sum, v, j) => sum + (beta[j] * (v - mu[j])) / scale[j], 0),
        mean: mean(past.map((row) => row.y)),
        zero: 0,
      };
    });
    return {
      origin,
      training: {
        rows: rows.length,
        first_origin: first,
        last_origin: last,
        max_release: labelRelease(last, c.horizon, c.delay),
        cutoff: origin,
        eligible_origins: last - first + 1,
        feature_mean: mu,
        feature_scale: scale,
      },
      predictions,
      coefficients: beta,
      intercept: yMean,
    };
  }
  function score(rows, key) {
    const actual = rows.map((r) => r.actual),
      predicted = rows.map((r) => r[key]),
      ma = mean(actual),
      mp = mean(predicted);
    let sq = 0,
      abs = 0,
      cov = 0,
      va = 0,
      vp = 0;
    for (let i = 0; i < rows.length; i++) {
      const error = predicted[i] - actual[i];
      sq += error * error;
      abs += Math.abs(error);
      cov += (actual[i] - ma) * (predicted[i] - mp);
      va += (actual[i] - ma) ** 2;
      vp += (predicted[i] - mp) ** 2;
    }
    return {
      rmse: Math.sqrt(sq / rows.length),
      mae: abs / rows.length,
      correlation: va > 0 && vp > 0 ? cov / Math.sqrt(va * vp) : null,
      count: rows.length,
    };
  }
  function evaluate(data, config = {}) {
    validateData(data);
    const c = validateConfig(config, data),
      predictions = [],
      receipts = [];
    for (let origin = c.start; origin <= c.end; origin++) {
      const fit = fitAt(data, origin, c);
      receipts.push({
        origin,
        ...fit.training,
        coefficients: fit.coefficients,
        intercept: fit.intercept,
      });
      for (let j = 0; j < data.series.length; j++)
        predictions.push({
          origin,
          ...fit.predictions[j],
          actual: targetAt(data.series[j], origin, c.horizon),
          target_end: origin + c.horizon - 1,
          released_day: labelRelease(origin, c.horizon, c.delay),
        });
    }
    const metrics = {};
    for (const key of Object.keys(MODELS))
      metrics[key] = score(predictions, key);
    return {
      schema: "commodity-browser-forecast-v1",
      scope: "SYNTHETIC_ONLY",
      config: c,
      origins: c.end - c.start + 1,
      fit_count: c.end - c.start + 1,
      predictions,
      receipts,
      metrics,
      features: FEATURES,
      models: MODELS,
      timing_rule:
        "Forecast at the start of origin t. Features use observations before t. Training labels require origin + horizon + publication delay <= t.",
      target:
        "Mean of the next horizon observations, in arbitrary synthetic signal units.",
      caveat:
        "Overlapping target horizons are not independent samples. These descriptive errors are synthetic demonstration evidence, not market or financial performance.",
    };
  }
  function eligibility(data, origin, config, seriesId) {
    const c = validateConfig(config, data),
      series = data.series.find((s) => s.id === seriesId);
    if (!series) throw new Error("Unknown series.");
    const first = Math.max(20, origin - c.window),
      rows = [];
    for (let t = first; t < origin; t++) {
      const release = labelRelease(t, c.horizon, c.delay),
        eligible = release <= origin;
      rows.push({
        origin: t,
        target_end: t + c.horizon - 1,
        released_day: release,
        eligible,
        label: eligible ? targetAt(series, t, c.horizon) : null,
        status: eligible ? "Available for training" : "Not released",
      });
    }
    return rows;
  }
  const API = {
    FEATURES,
    MODELS,
    generateData,
    validateData,
    validateConfig,
    featureAt,
    targetAt,
    labelRelease,
    readLabel,
    fitAt,
    evaluate,
    eligibility,
    score,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  root.CommodityEngine = API;
})(typeof window !== "undefined" ? window : globalThis);
