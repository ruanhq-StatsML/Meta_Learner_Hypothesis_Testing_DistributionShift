"""Contract tests plus one strong-signal check that the two groups separate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from federated_vimp.aggregate import aggregate_uploads, loso_l1, pareto_mask
from federated_vimp.experiment import analyze_site
from federated_vimp.schema import validate_upload
from federated_vimp.statistics import StatConfig, dominance, placebo_std, robust_z, simplex


def _peaked(index, p=8):
    raw = np.zeros(p, dtype=float)
    raw[index] = 1.0
    raw[(index + 1) % p] = 0.05
    return simplex(raw)


def _flat(p=8):
    return simplex(np.ones(p))


def _packet(site_id, cov_index, con_index, z_mmd, z_r, n_k, sign):
    cov = _peaked(cov_index) if cov_index is not None else _flat()
    con = _peaked(con_index) if con_index is not None else _flat()
    # `sign` only documents intent; D is computed from the vectors.
    del sign
    d_value = dominance(con, cov)
    return {
        "site_id": site_id,
        "vimp_cov": cov.tolist(),
        "vimp_con": con.tolist(),
        "z_mmd": float(z_mmd),
        "z_r": float(z_r),
        "calibration_ok": True,
        "D_k": float(d_value),
        "D_k_ci": [float(d_value), float(d_value)],
        "n_k": int(n_k),
        "feature_mask": [True] * 8,
    }


def _fleet():
    return [
        _packet("P", None, 0, 0.0, 6.0, 1000, +1),
        _packet("M1", None, 0, 0.0, 0.5, 500, +1),
        _packet("M2", None, 0, 0.0, 0.4, 500, +1),
        _packet("B", None, 5, 0.0, 0.6, 3000, +1),
        _packet("Q", 2, None, 5.0, 0.0, 1000, -1),
        _packet("C1", 2, None, 0.4, 0.0, 500, -1),
        _packet("C2", 2, None, 0.3, 0.0, 500, -1),
        _packet("L", 7, None, 8.0, 0.0, 220, -1),
    ]


def test_simplex_and_dominance():
    uniform = simplex(np.zeros(4))
    assert uniform.sum() == pytest.approx(1.0)
    assert np.allclose(uniform, 0.25)
    masked = simplex(np.array([0.0, 3.0, 1.0, 9.0]), np.array([True, True, False, False]))
    assert masked[2] == 0.0 and masked[3] == 0.0
    assert masked.sum() == pytest.approx(1.0)
    assert dominance(_peaked(0), _flat()) > 0.0
    assert dominance(_flat(), _peaked(2)) < 0.0


def test_placebo_z_is_order_one():
    rng = np.random.default_rng(0)
    history = rng.normal(size=40)
    assert abs(placebo_std(history) - 1.0) < 0.35
    z, _, _ = robust_z(0.0, history)
    assert abs(z) < 3.0


def test_loso_matches_leave_one_out_mean():
    stack = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    weights = np.array([1.0, 1.0, 2.0])
    got = loso_l1(stack, weights)
    total_w = weights.sum()
    full = (weights[:, None] * stack).sum(axis=0) / total_w
    for k in range(3):
        mask = np.ones(3, dtype=bool)
        mask[k] = False
        held = (weights[mask, None] * stack[mask]).sum(axis=0) / weights[mask].sum()
        assert got[k] == pytest.approx(np.abs(full - held).sum())


def test_pareto_keeps_the_upper_right_chain():
    z = np.array([5.0, 1.0, 4.0, 3.0])
    delta = np.array([0.2, 0.9, 0.8, 0.3])
    assert pareto_mask(z, delta).tolist() == [True, True, True, False]


def test_schema_rejects_a_merged_scalar_and_a_non_simplex():
    packet = _fleet()[0]
    validate_upload(packet)
    broken = dict(packet)
    broken["causal_effect"] = 1.0
    with pytest.raises(ValueError):
        validate_upload(broken)
    broken = dict(packet)
    broken["vimp_cov"] = [0.5] * 8
    with pytest.raises(ValueError):
        validate_upload(broken)


def test_gated_router_separates_the_four_roles():
    report = aggregate_uploads(_fleet())
    routes = {row["site_id"]: row["route"] for row in report["sites"]}
    assert routes["P"] == "primary_driver"
    assert routes["Q"] == "primary_driver"
    assert routes["L"] == "local_anomaly"
    assert routes["B"] == "consensus_breaker"
    assert "M1" != "primary_driver"
    assert routes["M1"] != "primary_driver"
    assert set(report["consensus_con"]) >= {0}
    assert set(report["consensus_cov"]) >= {2}
    # The literal mixed-plane route is allowed to disagree; the test records
    # that the gate, not the disagreement LOSO, is what names both drivers.
    assert report["n_primary_gated"] == 2


def test_mechanism_separation_on_a_strong_shift():
    config = StatConfig(mmd_max_n=120, hsic_max_n=100, rf_trees=20, rf_depth=5)
    concept = {
        "site_id": "c", "role": "concept_primary", "kind": "concept", "n_k": 400,
        "delta_c": 2.8, "concept_idx": (0, 1), "concept_w": (1.0, 0.55),
        "delta_x": 0.0, "cov_idx": (), "noise": 1.0, "drop": (),
    }
    covariate = {
        "site_id": "x", "role": "cov_primary", "kind": "covariate", "n_k": 400,
        "delta_c": 0.0, "concept_idx": (), "concept_w": (),
        "delta_x": 2.2, "cov_idx": (2, 3), "noise": 1.0, "drop": (),
    }
    got_c = analyze_site(concept, seed=21, config=config, n_history=8, n_cap=280, n_boot=0)
    got_x = analyze_site(covariate, seed=22, config=config, n_history=8, n_cap=280, n_boot=0)
    assert got_c["D_k"] > 0.0
    assert int(np.argmax(got_c["vimp_con"])) in (0, 1)
    assert got_x["D_k"] < 0.0
    assert int(np.argmax(got_x["vimp_cov"])) in (2, 3)
    assert abs(sum(got_c["vimp_cov"]) - 1.0) < 1e-6
    assert abs(sum(got_x["vimp_con"]) - 1.0) < 1e-6
