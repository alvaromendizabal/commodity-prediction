import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { generate, replay, options } from "./replay_public_demo.mjs";
import { runBundle, readJson, digest } from "./replay_bundle.mjs";

const canonical = generate();
test("default public artifact replays every fit, target and timing receipt", () => {
  const verified = replay(canonical);
  assert.equal(verified.fits, 140);
  assert.equal(verified.forecast_rows, 840);
});
for (const [name, mutate] of [
  [
    "modified observations",
    (x) => {
      x.data.series[0].values[50] += 1;
    },
  ],
  [
    "invented prediction",
    (x) => {
      x.result.predictions[0].ridge += 0.01;
    },
  ],
  [
    "altered aggregate metric",
    (x) => {
      x.result.metrics.ridge.rmse = 0;
    },
  ],
  [
    "changed training boundary",
    (x) => {
      x.result.receipts[0].max_release += 1;
    },
  ],
  [
    "future-fitted scaling",
    (x) => {
      x.result.receipts[0].feature_mean[0] += 1;
    },
  ],
  [
    "missing prediction",
    (x) => {
      x.result.predictions.pop();
    },
  ],
  [
    "extra unverified field",
    (x) => {
      x.result.private_score = 99;
    },
  ],
  [
    "invalid inspected origin",
    (x) => {
      x.review.forecast_origin = 500;
    },
  ],
  [
    "mismatched generator seed",
    (x) => {
      x.authored_seed += 1;
    },
  ],
  [
    "unsafe generator seed",
    (x) => {
      x.data.seed = Number.MAX_SAFE_INTEGER;
    },
  ],
])
  test(`replay rejects ${name}`, () => {
    const changed = structuredClone(canonical);
    mutate(changed);
    assert.throws(() => replay(changed));
  });
test("floating tolerance accepts only the documented tiny numerical variation", () => {
  const changed = structuredClone(canonical);
  changed.result.predictions[0].ridge += 1e-12;
  assert.equal(replay(changed).fits, 140);
});
test("reordered object fields are accepted without weakening content equality", () => {
  assert.deepEqual(
    replay(Object.fromEntries(Object.entries(canonical).reverse())),
    replay(canonical),
  );
});

function workspace(fn) {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "forecast-replay-test-"));
  try {
    fn(base);
  } finally {
    fs.rmSync(base, { recursive: true, force: true });
  }
}
test("create, re-execute and check preserve byte-identical saved artifacts", () =>
  workspace((base) => {
    const output = path.join(base, "result");
    runBundle(options, ["--output", output]);
    const before = fs
      .readdirSync(output)
      .map((f) => digest(fs.readFileSync(path.join(output, f))));
    runBundle(options, ["--output", output, "--check"]);
    assert.deepEqual(
      fs
        .readdirSync(output)
        .map((f) => digest(fs.readFileSync(path.join(output, f)))),
      before,
    );
  }));
test("actual exported input is replayed and differs from changed input", () =>
  workspace((base) => {
    const input = path.join(base, "browser.json"),
      output = path.join(base, "result");
    fs.writeFileSync(input, JSON.stringify(canonical));
    runBundle(options, ["--input", input, "--output", output]);
    runBundle(options, ["--input", input, "--output", output, "--check"]);
    const changed = structuredClone(canonical);
    changed.review.model = "mean";
    fs.writeFileSync(input, JSON.stringify(changed));
    assert.throws(
      () => runBundle(options, ["--input", input, "--output", output]),
      /mismatch/,
    );
  }));
test("malformed input cannot publish a successful output directory", () =>
  workspace((base) => {
    const input = path.join(base, "bad.json"),
      output = path.join(base, "result");
    fs.writeFileSync(input, '{"schema":"wrong"}');
    assert.throws(() =>
      runBundle(options, ["--input", input, "--output", output]),
    );
    assert.equal(fs.existsSync(output), false);
  }));
test("tampered saved artifact is rejected even when the manifest is recomputed", () =>
  workspace((base) => {
    const output = path.join(base, "result");
    runBundle(options, ["--output", output]);
    const file = path.join(output, "export.json"),
      changed = readJson(file);
    changed.result.predictions[0].ridge += 5;
    fs.writeFileSync(file, JSON.stringify(changed));
    const manifestPath = path.join(output, "manifest.json"),
      manifest = readJson(manifestPath);
    manifest.files["export.json"] = digest(fs.readFileSync(file));
    fs.writeFileSync(manifestPath, JSON.stringify(manifest));
    assert.throws(
      () => runBundle(options, ["--output", output, "--check"]),
      /mismatch/,
    );
  }));
test("changed computation source invalidates a saved receipt", () =>
  workspace((base) => {
    const root = path.join(base, "source"),
      output = path.join(base, "result");
    for (const name of [
      "public-demo/engine.js",
      "public-demo/app.js",
      "tools/replay_bundle.mjs",
      "tools/replay_public_demo.mjs",
    ]) {
      fs.mkdirSync(path.dirname(path.join(root, name)), { recursive: true });
      fs.copyFileSync(path.join(options.root, name), path.join(root, name));
    }
    runBundle({ ...options, root }, ["--output", output]);
    fs.appendFileSync(
      path.join(root, "public-demo/engine.js"),
      "\n// changed\n",
    );
    assert.throws(
      () => runBundle({ ...options, root }, ["--output", output, "--check"]),
      /source_sha256/,
    );
  }));
test("existing unrelated files, missing checks and checkout outputs fail closed", () =>
  workspace((base) => {
    fs.writeFileSync(path.join(base, "keep.txt"), "preserve me");
    assert.throws(() => runBundle(options, ["--output", base]));
    assert.equal(
      fs.readFileSync(path.join(base, "keep.txt"), "utf8"),
      "preserve me",
    );
    assert.throws(() =>
      runBundle(options, ["--output", path.join(base, "absent"), "--check"]),
    );
    assert.throws(
      () =>
        runBundle(options, [
          "--output",
          path.join(options.root, "replay-output"),
        ]),
      /outside/,
    );
  }));
test("symlink inputs and oversized files are rejected before parsing", () =>
  workspace((base) => {
    const input = path.join(base, "input.json"),
      link = path.join(base, "link.json");
    fs.writeFileSync(input, JSON.stringify(canonical));
    fs.symlinkSync(input, link);
    assert.throws(() => readJson(link), /Symlink/);
    const fd = fs.openSync(input, "w");
    fs.ftruncateSync(fd, 8 * 1024 * 1024 + 1);
    fs.closeSync(fd);
    assert.throws(() => readJson(input), /8 MiB/);
  }));

test("receipt scope, authentication, shape, runtime and timestamps are validated beyond hashes", () =>
  workspace((base) => {
    const output = path.join(base, "result");
    runBundle(options, ["--output", output]);
    const receiptPath = path.join(output, "receipt.json"),
      original = readJson(receiptPath),
      manifestPath = path.join(output, "manifest.json");
    for (const mutate of [
      (x) => {
        x.scope = "PRIVATE_PRODUCTION_EVIDENCE";
      },
      (x) => {
        x.authentication = "Independently signed evidence";
      },
      (x) => {
        x.elapsed_seconds = -1;
      },
      (x) => {
        x.injected_claim = "world best";
      },
      (x) => {
        x.created_utc = "yesterday";
      },
      (x) => {
        x.runtime.cpus = 0;
      },
      (x) => {
        x.runtime.node = "invented";
      },
    ]) {
      const receipt = structuredClone(original);
      mutate(receipt);
      fs.writeFileSync(receiptPath, JSON.stringify(receipt));
      const manifest = readJson(manifestPath);
      manifest.files["receipt.json"] = digest(fs.readFileSync(receiptPath));
      fs.writeFileSync(manifestPath, JSON.stringify(manifest));
      assert.throws(() => runBundle(options, ["--output", output, "--check"]));
    }
  }));
test("oversized saved bundle member is rejected before hashing", () =>
  workspace((base) => {
    const output = path.join(base, "result");
    runBundle(options, ["--output", output]);
    const fd = fs.openSync(path.join(output, "export.json"), "w");
    fs.ftruncateSync(fd, 8 * 1024 * 1024 + 1);
    fs.closeSync(fd);
    assert.throws(
      () => runBundle(options, ["--output", output, "--check"]),
      /8 MiB/,
    );
  }));

test("unrecognized configuration fields cannot become unverified evidence", () => {
  const changed = structuredClone(canonical);
  changed.result.config.invented_validation =
    "private market performance confirmed";
  assert.throws(() => replay(changed), /config.fields/);
});
