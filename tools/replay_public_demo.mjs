import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { match, requireThat, runBundle } from "./replay_bundle.mjs";
const require = createRequire(import.meta.url);
const E = require("../public-demo/engine.js"),
  App = require("../public-demo/app.js");
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export function generate() {
  const data = E.generateData(),
    result = E.evaluate(data);
  return App.exportResult(data, result, {
    seriesId: "A",
    model: "ridge",
    origin: 120,
  });
}
export function replay(exported) {
  requireThat(
    exported?.schema === "commodity-forecasting-lab-export-v1" &&
      exported.scope === "SYNTHETIC_ONLY",
    "Unsupported synthetic forecast export.",
  );
  const supplied = exported.data;
  requireThat(
    supplied &&
      Number.isSafeInteger(supplied.seed) &&
      supplied.seed <= 4294967295,
    "Invalid authored seed.",
  );
  const data = E.generateData({
    seed: supplied.seed,
    scenario: supplied.scenario,
    days: supplied.days,
  });
  match(data, supplied, "data", 1e-10);
  const view = exported.review;
  requireThat(
    view &&
      data.series.some((s) => s.id === view.series_id) &&
      Object.hasOwn(E.MODELS, view.model),
    "Invalid review selection.",
  );
  requireThat(
    exported.result?.config && typeof exported.result.config === "object",
    "Missing forecast configuration.",
  );
  match(
    ["horizon", "delay", "window", "alpha", "start", "end"].sort(),
    Object.keys(exported.result.config).sort(),
    "config.fields",
  );
  const result = E.evaluate(data, exported.result.config);
  requireThat(
    Number.isInteger(view.forecast_origin) &&
      view.forecast_origin >= result.config.start &&
      view.forecast_origin <= result.config.end,
    "Invalid review origin.",
  );
  const expected = App.exportResult(data, result, {
    seriesId: view.series_id,
    model: view.model,
    origin: view.forecast_origin,
  });
  match(expected, exported, "export", 1e-10);
  return {
    scope: "SYNTHETIC_ONLY",
    fits: result.fit_count,
    forecast_rows: result.predictions.length,
    candidates: Object.keys(result.models).length,
    series: data.series.length,
    horizon: result.config.horizon,
    delay: result.config.delay,
    tolerance:
      "Floating values: 1e-10 * (1 + abs(expected)); integer fields exact.",
    checks: [
      "authored panel regenerated",
      "all chronological fits recomputed",
      "training boundaries and preprocessing receipts recomputed",
      "all targets, predictions, and matched metrics recomputed",
      "review selection validated",
    ],
  };
}
export const options = {
  root,
  project: "commodity-prediction",
  generate,
  replay,
};
if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  try {
    runBundle(options);
  } catch (error) {
    console.error(`REPLAY_FAILED: ${error.message}`);
    process.exitCode = 1;
  }
}
