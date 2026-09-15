"""Prepared USER-RUN tests for rounds18/19. No private model fitting or network."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
m = importlib.import_module("commodity_next_research")


def fixture(T=210, J=8):
    rng = np.random.default_rng(162719)
    pairs = pd.DataFrame(
        {"target": [f"target_{i}" for i in range(J)], "lag": np.resize([1, 2, 3, 4], J)}
    )
    y = rng.normal(0, 0.03, (T, J))
    risk = np.full((T, J), 0.05)
    mean = np.full((T, J), 0.002)
    return y, risk, mean, pairs


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.y, self.risk, self.mean, self.pairs = fixture()

    def test_delay_exact_each_horizon(self):
        a = m.release_stream(self.y, self.pairs)
        for j, h in enumerate(self.pairs.lag):
            np.testing.assert_array_equal(a[int(h) + 1 :, j], self.y[: -int(h) - 1, j])
            self.assertTrue(np.isnan(a[: int(h) + 1, j]).all())

    def test_no_original_mutation(self):
        a = self.y.copy()
        m.release_stream(self.y, self.pairs)
        np.testing.assert_array_equal(a, self.y)

    def test_sentinal_missing(self):
        a = self.y.copy()
        a[10, 0] = -999999
        self.assertTrue(np.isnan(m.clean_labels(a, self.pairs)[10, 0]))

    def test_infinite_is_missing(self):
        a = self.y.copy()
        a[10, 0] = np.inf
        self.assertTrue(np.isnan(m.clean_labels(a, self.pairs)[10, 0]))

    def test_zero_remains_valid(self):
        a = self.y.copy()
        a[10, 0] = 0
        self.assertEqual(m.clean_labels(a, self.pairs)[10, 0], 0)

    def test_final_rows_rejected(self):
        with self.assertRaises(m.p.Stop):
            m.clean_labels(np.zeros((1705, 8)), self.pairs)

    def test_bad_horizon(self):
        self.pairs.loc[0, "lag"] = 0
        with self.assertRaises(m.p.Stop):
            m.clean_labels(self.y, self.pairs)

    def test_duplicate_targets(self):
        self.pairs.loc[1, "target"] = "target_0"
        with self.assertRaises(m.p.Stop):
            m.clean_labels(self.y, self.pairs)

    def test_reference_axes(self):
        with self.assertRaises(m.p.Stop):
            m.clean_labels(self.y, self.pairs, self.mean[:-1])

    def test_read_only_labels(self):
        self.y.setflags(write=False)
        a = m.clean_labels(self.y, self.pairs)
        self.assertTrue(a.flags.writeable)


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.y, self.risk, self.mean, self.pairs = fixture()

    def build(self, y=None, risk=None):
        return m.sequence_block(
            self.y if y is None else y, self.risk if risk is None else risk, self.pairs
        )

    def test_shape_and_names(self):
        c, n, meta = self.build()
        self.assertEqual(c.shape, (210, 8, 24))
        self.assertEqual(len(set(n)), 24)

    def test_group_sizes(self):
        _, _, z = self.build()
        self.assertEqual([len(x) for x in m.groups(18, z).values()], [6, 15, 15, 24])

    def test_exact_group_union(self):
        _, _, z = self.build()
        g = m.groups(18, z)
        a, b, c, d = m.INFO[18]["variants"]
        self.assertEqual(set(g[d]), set(g[b]) | set(g[c]))
        self.assertEqual(set(g[a]), set(g[b]) & set(g[c]))

    def test_prefix(self):
        full, _, _ = self.build()
        for end in (80, 150, 200):
            np.testing.assert_array_equal(
                full[:end], m.sequence_block(self.y[:end], self.risk[:end], self.pairs)[0]
            )

    def test_future_perturbation(self):
        a = self.y.copy()
        a[160:] *= 7
        np.testing.assert_array_equal(self.build()[0][:160], self.build(y=a)[0][:160])

    def test_each_release_boundary(self):
        original = self.build()[0]
        for j, h in enumerate(self.pairs.lag):
            y = self.y.copy()
            y[155, j] += 1
            changed = self.build(y=y)[0]
            np.testing.assert_array_equal(
                original[: 155 + int(h) + 1, j], changed[: 155 + int(h) + 1, j]
            )

    def test_pairwise_corr_independent_reference(self):
        cube, n, _ = self.build()
        w = 63
        t = 190
        j = 3
        h = int(self.pairs.iloc[j]["lag"])
        delay = h + 1
        x = self.y[t - w + 1 - delay : t + 1 - delay, j]
        q = self.y[t - w + 1 - delay - h : t + 1 - delay - h, j]
        expected = np.corrcoef(x, q)[0, 1]
        self.assertAlmostEqual(
            float(cube[t, j, n.index("released_sequence18__w63__lag_correlation")]),
            expected,
            places=6,
        )

    def test_mismatched_missing_pairs_use_same_rows(self):
        self.y[120:150:4, 2] = np.nan
        c, n, _ = self.build()
        t = 190
        h = 3
        d = 4
        a = self.y[t - 62 - d : t + 1 - d, 2]
        b = self.y[t - 62 - d - h : t + 1 - d - h, 2]
        mask = np.isfinite(a) & np.isfinite(b)
        expected = np.corrcoef(a[mask], b[mask])[0, 1]
        self.assertAlmostEqual(
            float(c[t, 2, n.index("released_sequence18__w63__lag_correlation")]), expected, places=6
        )

    def test_clocks_ignore_magnitudes(self):
        a, n, z = self.build()
        b, _, _ = self.build(y=self.y * 3)
        ix = [i for i, q in enumerate(z) if q["group"] == "clock"]
        np.testing.assert_array_equal(a[:, :, ix], b[:, :, ix])

    def test_positive_units_invariance(self):
        a, _, _ = self.build()
        b, _, _ = self.build(y=self.y * 7, risk=self.risk * 7)
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5, equal_nan=True)

    def test_missing_outcomes_not_zero_filled(self):
        self.y[:, 0] = np.nan
        c, n, z = self.build()
        ix = [i for i, x in enumerate(z) if x["group"] != "clock"]
        self.assertTrue(np.isnan(c[:, 0, ix]).all())

    def test_no_infinite_features(self):
        self.y[150, 0] = np.inf
        c, _, _ = self.build()
        self.assertFalse(np.isinf(c).any())

    def test_readonly_copy_on_write(self):
        self.y.setflags(write=False)
        self.risk.setflags(write=False)
        # pandas >=3 always uses Copy-on-Write; read-only NumPy inputs still exercise mutation safety.
        self.build()

    def test_input_unchanged(self):
        y = self.y.copy()
        r = self.risk.copy()
        self.build()
        np.testing.assert_array_equal(y, self.y)
        np.testing.assert_array_equal(r, self.risk)


class ErrorMemoryTests(unittest.TestCase):
    def setUp(self):
        self.y, self.risk, self.mean, self.pairs = fixture()

    def build(self, y=None, mean=None, risk=None):
        return m.error_block(
            self.y if y is None else y,
            self.mean if mean is None else mean,
            self.risk if risk is None else risk,
            self.pairs,
        )

    def test_shape_names(self):
        c, n, z = self.build()
        self.assertEqual(c.shape, (210, 8, 24))
        self.assertEqual(len(set(n)), 24)

    def test_group_sizes(self):
        _, _, z = self.build()
        self.assertEqual([len(x) for x in m.groups(19, z).values()], [6, 15, 15, 24])

    def test_exact_union(self):
        _, _, z = self.build()
        g = m.groups(19, z)
        a, b, c, d = m.INFO[19]["variants"]
        self.assertEqual(set(g[d]), set(g[b]) | set(g[c]))
        self.assertEqual(set(g[a]), set(g[b]) & set(g[c]))

    def test_prefix(self):
        full, _, _ = self.build()
        for end in (80, 150, 200):
            np.testing.assert_array_equal(
                full[:end],
                m.error_block(self.y[:end], self.mean[:end], self.risk[:end], self.pairs)[0],
            )

    def test_reference_at_original_origin(self):
        self.mean = np.arange(210)[:, None] * np.ones((1, 8)) * 0.0002
        c, n, _ = self.build()
        t = 190
        j = 3
        d = 5
        e = (self.y[t - 20 - d : t + 1 - d, j] - self.mean[t - 20 - d : t + 1 - d, j]) / self.risk[
            t - 20 - d : t + 1 - d, j
        ]
        self.assertAlmostEqual(
            float(c[t, j, n.index("prior_error19__w21__bias_mean")]),
            float(np.clip(e, -12, 12).mean()),
            places=6,
        )

    def test_no_current_origin_reference(self):
        a, _, _ = self.build()
        changed = self.mean.copy()
        changed[150, 3] += 10
        b, _, _ = self.build(mean=changed)
        np.testing.assert_array_equal(a[:155, 3], b[:155, 3])

    def test_delay_each_horizon(self):
        original = self.build()[0]
        for j, h in enumerate(self.pairs.lag):
            y = self.y.copy()
            y[155, j] += 1
            b = self.build(y=y)[0]
            np.testing.assert_array_equal(original[: 155 + int(h) + 1, j], b[: 155 + int(h) + 1, j])

    def test_median_reference(self):
        c, n, _ = self.build()
        e = (self.y[166:187, 1] - self.mean[166:187, 1]) / self.risk[166:187, 1]
        self.assertAlmostEqual(
            float(c[189, 1, n.index("prior_error19__w21__bias_median")]),
            float(np.median(np.clip(e, -12, 12))),
            places=6,
        )

    def test_clocks_ignore_errors(self):
        a, n, z = self.build()
        b, _, _ = self.build(y=self.y * 3)
        ix = [i for i, q in enumerate(z) if q["group"] == "clock"]
        np.testing.assert_array_equal(a[:, :, ix], b[:, :, ix])

    def test_units_invariance(self):
        a, _, _ = self.build()
        b, _, _ = self.build(y=self.y * 2, mean=self.mean * 2, risk=self.risk * 2)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_zero_error(self):
        c, n, _ = self.build(y=self.mean.copy())
        self.assertEqual(c[180, 0, n.index("prior_error19__w63__root_squared_error")], 0)

    def test_invalid_reference_risk_not_replaced(self):
        self.risk[:] = 0
        c, n, z = self.build()
        ix = [i for i, x in enumerate(z) if x["group"] != "clock"]
        self.assertTrue(np.isnan(c[:, :, ix]).all())

    def test_fixed_large_error_threshold(self):
        self.y[:] = 0.152
        self.mean[:] = 0.002
        self.risk[:] = 0.05
        c, n, _ = self.build()
        self.assertEqual(c[190, 0, n.index("prior_error19__w63__large_error_rate")], 1)

    def test_raw_outcomes_missing(self):
        self.y[:, 0] = -999999
        c, n, z = self.build()
        ix = [i for i, x in enumerate(z) if x["group"] != "clock"]
        self.assertTrue(np.isnan(c[:, 0, ix]).all())

    def test_readonly_copy_on_write(self):
        self.y.setflags(write=False)
        self.risk.setflags(write=False)
        self.mean.setflags(write=False)
        # pandas >=3 always uses Copy-on-Write; read-only NumPy inputs still exercise mutation safety.
        self.build()

    def test_input_unchanged(self):
        y = self.y.copy()
        mean = self.mean.copy()
        self.build()
        np.testing.assert_array_equal(y, self.y)
        np.testing.assert_array_equal(mean, self.mean)

    def test_no_infinite_features(self):
        self.y[150, 0] = np.inf
        c, _, _ = self.build()
        self.assertFalse(np.isinf(c).any())


class ExecutionDeclarationTests(unittest.TestCase):
    def test_packaged_declarations_match_code(self):
        for n in (18, 19):
            saved = json.loads(
                (ROOT / "configs" / ("manual_" + m.INFO[n]["slug"] + ".json")).read_text()
            )
            self.assertEqual(saved, m.plan(n))

    def test_only_last_development_fold(self):
        for n in (18, 19):
            self.assertEqual(m.plan(n)["validation_stop"], 1704)
            self.assertEqual(m.plan(n)["validation_start"], 1529)

    def test_four_fits_each(self):
        for n in (18, 19):
            self.assertEqual(m.plan(n)["new_fit_limit"], 4)
            self.assertEqual(m.plan(n)["candidate_templates"], 24)

    def test_no_other_round_selection(self):
        for n in (18, 19):
            self.assertFalse(m.plan(n)["other_round_results_used_for_selection"])

    def test_numeric_panels_compare_to_clocks(self):
        for n in (18, 19):
            v = m.INFO[n]["variants"]
            for x in v[1:]:
                self.assertIn(v[0], m.plan(n)["comparators"][x])

    def test_union_compares_halves(self):
        for n in (18, 19):
            v = m.INFO[n]["variants"]
            self.assertEqual(set(m.plan(n)["comparators"][v[-1]]), {"current_market", *v[:-1]})

    def test_180_not_a_final_holdout(self):
        for n in (18, 19):
            self.assertEqual(m.plan(n)["validation_origins"], 175)
            self.assertEqual(m.plan(n)["final_test_evaluations"], 0)

    def test_missing_preflight_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(m.p.Stop):
                m.test_gate(Path(d))

    def test_invalid_round_fails(self):
        for n in (True, 16, 20, "18"):
            with self.assertRaises(m.p.Stop):
                m.require(n)

    def test_heartbeat_regression_reused(self):
        out = io.StringIO()
        with redirect_stdout(out):
            m.p.emit("WORKER_HEARTBEAT", **m.p.heartbeat_fields({"stage": "FIT", "task": "x"}))
        self.assertEqual(json.loads(out.getvalue())["worker_stage"], "FIT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    a = parser.parse_args()
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    status = "TESTS_PASSED" if result.wasSuccessful() and not result.skipped else "TESTS_FAILED"
    r = dict(
        status=status,
        tests_run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        skipped=len(result.skipped),
        elapsed_seconds=round(time.monotonic() - start, 3),
        private_model_fits=0,
        synthetic_forecast_fits=0,
        helper_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
        tests_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=sys.version,
        numpy=np.__version__,
        pandas=pd.__version__,
    )
    m.p.atomic_json(a.receipt, r)
    print("RESULT:", status)
    raise SystemExit(0 if status == "TESTS_PASSED" else 1)
