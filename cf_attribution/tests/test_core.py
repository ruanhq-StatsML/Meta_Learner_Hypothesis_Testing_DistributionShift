"""Tests that lock the protocol math and the simulator's planted order."""

from __future__ import annotations

import unittest

import numpy as np

from cf_attribution.dgp import make_synthetic_panel, oracle_outcome
from cf_attribution.engine import precision_ok, project_support
from cf_attribution.metrics import (
    evalue_from_smd,
    median_bandwidth,
    mmd2_rff,
    pairwise_concordance,
    spearman,
)
from cf_attribution.models import ConditionalDDPM, ConditionalRealNVP
from cf_attribution.nets import MLP
from cf_attribution.semantics import RFF_PEAK_SLACK, Thresholds


class MetricTests(unittest.TestCase):
    def test_concordance_perfect_and_reversed(self):
        truth = [1, 2, 3, 4]
        self.assertEqual(pairwise_concordance(truth, truth), 1.0)
        self.assertEqual(pairwise_concordance(truth[::-1], truth), 0.0)

    def test_spearman_lag_sign(self):
        lag = [1, 2, 3, 4, 5]
        effect = [5, 4, 2, 1, 0.2]
        self.assertLess(spearman(lag, effect), -0.7)

    def test_mmd_detects_shift(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(80, 1))
        b = rng.normal(size=(80, 1))
        c = rng.normal(loc=1.5, size=(80, 1))
        near = mmd2_rff(a, b, n_freq=64, seed=1)
        far = mmd2_rff(a, c, n_freq=64, seed=1)
        self.assertLess(near, far)

    def test_evalue_grows_with_effect(self):
        self.assertGreater(evalue_from_smd(0.8), evalue_from_smd(0.2))
        self.assertGreater(evalue_from_smd(0.8), 2.0)

    def test_thresholds_are_the_protocol(self):
        t = Thresholds()
        self.assertEqual(t.l1_concordance, 0.8)
        self.assertEqual(t.l1_spearman, 0.7)
        self.assertEqual(t.l1_lag_spearman, -0.7)
        self.assertEqual(t.l1_min_passes, 3)
        self.assertEqual(t.l3_ks_p, 0.05)
        self.assertEqual(t.l3_hsic_low, 0.8)
        self.assertEqual(t.l3_hsic_high, 1.2)
        self.assertEqual(t.l5_evalue, 2.0)
        self.assertEqual(t.l5_evalue_fragile, 1.5)
        self.assertEqual(RFF_PEAK_SLACK, 1.05)


class PrecisionTests(unittest.TestCase):
    def test_peak_at_declared_lag(self):
        deltas = [0.1, 0.2, 0.4, 1.0, 0.5, 0.2]
        self.assertTrue(precision_ok(deltas, 4)["pass"])
        self.assertFalse(precision_ok(deltas, 2)["pass"])


class SupportTests(unittest.TestCase):
    def test_fallback_lands_on_a_reference_point(self):
        rng = np.random.default_rng(0)
        cloud = rng.normal(size=(40, 2))
        point = np.array([30.0, -30.0])
        adjusted, info = project_support(point, cloud, threshold=1.0)
        self.assertEqual(info["action"], "fallback")
        self.assertTrue(np.any(np.all(np.isclose(cloud, adjusted), axis=1)))


class GradientTests(unittest.TestCase):
    def test_mlp_matches_finite_difference(self):
        mlp = MLP([3, 6, 2], seed=0)
        x = np.random.default_rng(1).normal(size=(5, 3))
        y = mlp.forward(x)
        n = len(x)
        mlp.backward(y / n)
        analytic = float(mlp._dW[0][0, 0])
        base = float(mlp.W[0][0, 0])
        eps = 1e-5

        def loss(value: float) -> float:
            mlp.W[0][0, 0] = value
            out = mlp.forward(x)
            return float(0.5 * np.sum(out ** 2) / n)

        numeric = (loss(base + eps) - loss(base - eps)) / (2 * eps)
        self.assertAlmostEqual(analytic, numeric, places=4)


class OracleTests(unittest.TestCase):
    def test_planted_dose_orders_the_oracle(self):
        panel = make_synthetic_panel("long", seed=2026)
        members = np.where(panel.member_of == panel.target)[0][:16]
        y_scale = panel.outcome[: panel.train_end][:, members]
        sigma = median_bandwidth(y_scale[np.isfinite(y_scale)].reshape(-1, 1))
        scores = []
        doses = []
        for event in panel.queries["dose"]:
            entity = int(event["entity"])
            level = float(np.mean(panel.raw_block[panel.reference_index[entity], entity, 0]))
            y_cf = oracle_outcome(panel, int(event["t"]), entity, level)
            eval_t = int(event["t"]) + panel.peak_lag
            delta = mmd2_rff(
                panel.outcome[eval_t, members][:, None],
                y_cf[eval_t, members][:, None],
                sigma=sigma,
                n_freq=64,
                seed=0,
            )
            scores.append(delta)
            doses.append(event["dose"])
        self.assertGreater(pairwise_concordance(scores, doses), 0.8)
        self.assertGreater(spearman(scores, doses), 0.7)
        # A zero-alpha source must move the oracle outcome less than the
        # largest planted dose on the connected source.
        iso = panel.queries["isolated"][0]
        conn = max(panel.queries["dose"], key=lambda r: r["dose"])
        def delta_of(event):
            entity = int(event["entity"])
            level = float(np.mean(panel.raw_block[panel.reference_index[entity], entity, 0]))
            y_cf = oracle_outcome(panel, int(event["t"]), entity, level)
            eval_t = int(event["t"]) + panel.peak_lag
            return mmd2_rff(
                panel.outcome[eval_t, members][:, None],
                y_cf[eval_t, members][:, None],
                sigma=sigma,
                n_freq=64,
                seed=0,
            )
        self.assertLess(delta_of(iso), 0.5 * delta_of(conn))


class GenerativeTests(unittest.TestCase):
    def test_ddpm_and_flow_losses_drop(self):
        rng = np.random.default_rng(0)
        n, d = 500, 4
        cond = rng.normal(size=(n, 20))
        x = 0.8 * cond[:, :d] + rng.normal(scale=0.15, size=(n, d))
        slices = [(d, 2 * d), (2 * d, 3 * d), (3 * d, 4 * d)]
        # Pad cond so the three pools exist inside it.
        cond_ddpm = np.concatenate([cond[:, :d], cond[:, :d], cond[:, :d], cond[:, :d], cond[:, d:]], axis=1)
        ddpm = ConditionalDDPM(d, cond_ddpm.shape[1], slices, hidden=32, n_steps=6, seed=0)
        flow = ConditionalRealNVP(d, cond.shape[1], hidden=32, n_layers=3, seed=1)
        d_loss = ddpm.fit(x, cond_ddpm, epochs=8, batch=64, seed=0)
        f_loss = flow.fit(x, cond, epochs=12, batch=64, seed=1)
        self.assertLess(d_loss[-1], d_loss[0])
        self.assertLess(f_loss[-1], f_loss[0])


if __name__ == "__main__":
    unittest.main()
