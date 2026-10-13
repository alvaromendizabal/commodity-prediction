# From a browser result to verifiable forecasts

The Forecast Lab export contains enough public information to recompute its entire
synthetic experiment. The replay command regenerates the authored panel, runs the
same forecasting engine as the browser, and compares every prediction, target,
training receipt and metric. It does not accept a score just because its checksum matches.

## Run a reference experiment

Use Node 24.19.0 for the recorded CI environment; Node 22 is also tested. No packages,
Python environment, credentials, market data or cloud infrastructure are required.
From a checkout, choose a new output directory outside the repository:

```bash
node tools/replay_public_demo.mjs --output /tmp/forecast-replay
node tools/replay_public_demo.mjs --output /tmp/forecast-replay --check
```

The reference run contains six fictional series, 140 chronological ridge fits,
840 forecast rows and three matched candidates. The second command recomputes the
experiment and checks the existing bundle without changing its files.

## Verify a result you exported

In the [live lab](https://alvaro-forecasting-lab.tartmacaw2.chatgpt.site), choose
settings, select **Fit & evaluate**, then **Export data + result**. Point the verifier
at that downloaded JSON file:

```bash
node tools/replay_public_demo.mjs --input synthetic-forecast-review.json --output /tmp/my-forecast-replay
node tools/replay_public_demo.mjs --output /tmp/my-forecast-replay --check
```

A `PASS` requires all of these conditions:

1. The generator seed, scenario and bounded panel shape reproduce the supplied data.
2. Every fit uses eligible released labels; preprocessing and coefficients reproduce
   the supplied per-origin receipt.
3. All forward targets, baseline predictions, ridge predictions and aggregate metrics
   match the recomputation.
4. The inspected series, candidate and origin are valid and the export has no missing
   or unrecognized fields.

JavaScript implementations can differ slightly in floating-point arithmetic. Floating
values use a tolerance of `1e-10 * (1 + abs(expected))`; integer fields are exact.
The tolerance is recorded in the receipt. Altered observations, predictions, metrics,
training cutoffs and scaling are covered by rejection tests.

## What the bundle records

| File | Purpose |
|---|---|
| `export.json` | Complete synthetic input, fitted outputs and inspection selection |
| `receipt.json` | Replay checks, Node/platform details, UTC time, elapsed replay time and SHA-256 source identities |
| `manifest.json` | SHA-256 hashes for the exact export and receipt bytes |

The receipt binds the public engine, browser export implementation, replay validator
and transport utility. Changed source invalidates an existing bundle: use a new output
directory after an intentional code update. The verifier rejects missing/extra files,
symlinks, malformed or oversized JSON, altered manifests and changed results even when
their manifest has been recomputed. It publishes a directory only after replay succeeds;
existing unrelated output is preserved.

## Inspect the implementation

- [Domain replay](../tools/replay_public_demo.mjs): regenerates the panel and recomputes the actual public engine.
- [Artifact transport](../tools/replay_bundle.mjs): validates paths, binds source identity and publishes the bundle.
- [Adversarial tests](../tools/test_replay_public_demo.mjs): tampering, stale source, input rejection and repeat verification.
- [UI tests](../tools/test_public_demo.mjs): feed the real browser export handler's output into the replay validator.
- [Public-demo CI](../.github/workflows/public-demo.yml): both Node versions run export, import and recheck; downloadable artifacts retain the evidence.

```bash
node --test tools/test_replay_public_demo.mjs
node tools/test_public_demo.mjs
```

This verifies computational consistency of an authored demonstration. A local hash is
not an independent signature, a proof of historical execution, or evidence of private
forecasting quality. It does not reproduce the research score, claim profitability, or
release private market observations, fitted research models or feature recipes.
