# A five-minute delayed-label example

The forecasting problem has a timing constraint: an observation can exist before
its label is available for use. Reading an unreleased label gives a model
information that would not have existed when a forecast was made.

From the repository root, run:

```bash
python3 examples/delayed_label_demo.py
```

This uses only Python's standard library. It downloads nothing, writes no files,
and needs no credentials, GPU, or competition data.

The four authored observations have different publication delays:

| Fictional instrument | Observation day | Release day | Available on day 5? |
|---|---:|---:|---|
| `metal_alpha` | 1 | 3 | Yes |
| `metal_beta` | 1 | 5 | Yes, at the boundary |
| `metal_alpha` | 2 | 6 | No |
| `metal_beta` | 3 | 7 | No |

The output shows two available labels, their mean of `0.05`, and a deliberately
rejected request for the day-6 release. The mean illustrates aggregation over
available information; it is not a fitted forecast or a performance score.

The example defines publication as the **start of the release day**, so equality
is allowed. Labels become readable only when `released_day <= as_of_day`.
Negative or zero delays, duplicate observation keys, and nonfinite values are
rejected. Input order does not change the snapshot.

This is a small educational implementation with fictional dates and values. It
does not reproduce the official competition's release schedule, actual research
features, private data, model checkpoints, or submission recipe. The guard
illustrates a data contract, not protection against a caller inspecting memory.

For the actual public research workflow and its limits, see
[manual reproduction](../docs/MANUAL_REPRODUCTION.md) and
[publication scope](../docs/PUBLICATION_SCOPE.md). The example's tests cover the
exact release boundary and verify that changing future label values cannot alter
the current snapshot.
