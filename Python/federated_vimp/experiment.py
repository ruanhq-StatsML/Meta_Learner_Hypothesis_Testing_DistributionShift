"""20-site heterogeneous-shift experiment for aggregated OOD VIMP.

Ten sites change P(Y | X) only (concept drift). Ten change P(X) only
(covariate shift). Magnitude, sample size, and which coordinates move are
all heterogeneous, so three lookalike rules pick three different sites:

* argmax z picks the small local spike
* argmax n picks the large mild data center
* n-weighted simplex VIMP is pulled toward a large off-support site

The recommended server rule does not do any of those. It aggregates
strength-weighted group vectors, builds one Pareto frontier inside each
mechanism, and attaches a primary-driver label only when the site's leading
feature is a consensus feature of that aggregate.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from federated_vimp.aggregate import (
    aggregate_uploads,
    feature_mass,
    precision_at_k,
    render_executive,
    spearman_safe,
)
from federated_vimp.schema import validate_upload
from federated_vimp.statistics import (
    StatConfig,
    dominance,
    group_vectors,
    placebo_std,
    r_risk_signal,
    robust_z,
    scalar_mmd,
    simplex,
    z_inputs,
)


P = 8
CONCEPT_SUPPORT = (0, 1)
COVARIATE_SUPPORT = (2, 3)
OFF_SUPPORT = 5
LOCAL_FEATURE = 7
RHO = 0.25

# Population sizes differ on purpose. z-scores are computed after capping
# every site at the same analysis size; n_k is only an influence weight.
CATALOG = [
    {"site_id": "site-00", "role": "concept_primary", "kind": "concept", "n_k": 900,
     "delta_c": 0.70, "concept_idx": CONCEPT_SUPPORT, "concept_w": (1.0, 0.55),
     "delta_x": 0.0, "cov_idx": (), "noise": 1.0, "drop": ()},
    {"site_id": "site-01", "role": "concept_mild", "kind": "concept", "n_k": 520,
     "delta_c": 0.30, "concept_idx": CONCEPT_SUPPORT, "concept_w": (1.0, 0.55),
     "delta_x": 0.0, "cov_idx": (), "noise": 1.0, "drop": ()},
    {"site_id": "site-02", "role": "concept_mild", "kind": "concept", "n_k": 480,
     "delta_c": 0.30, "concept_idx": CONCEPT_SUPPORT, "concept_w": (1.0, 0.55),
     "delta_x": 0.0, "cov_idx": (), "noise": 1.0, "drop": ()},
    {"site_id": "site-03", "role": "concept_mild", "kind": "concept", "n_k": 610,
     "delta_c": 0.30, "concept_idx": CONCEPT_SUPPORT, "concept_w": (1.0, 0.55),
     "delta_x": 0.0, "cov_idx": (), "noise": 1.0, "drop": ()},
    {"site_id": "site-04", "role": "concept_mild", "kind": "concept", "n_k": 450,
     "delta_c": 0.30, "concept_idx": CONCEPT_SUPPORT, "concept_w": (1.0, 0.55),
     "delta_x": 0.0, "cov_idx": (), "noise": 1.50, "drop": ()},
    {"site_id": "site-05", "role": "concept_mild", "kind": "concept", "n_k": 540,
     "delta_c": 0.30, "concept_idx": CONCEPT_SUPPORT, "concept_w": (1.0, 0.55),
     "delta_x": 0.0, "cov_idx": (), "noise": 1.0, "drop": ()},
    {"site_id": "site-06", "role": "concept_mild", "kind": "concept", "n_k": 500,
     "delta_c": 0.30, "concept_idx": CONCEPT_SUPPORT, "concept_w": (1.0, 0.55),
     "delta_x": 0.0, "cov_idx": (), "noise": 1.0, "drop": ()},
    {"site_id": "site-07", "role": "concept_breaker", "kind": "concept", "n_k": 2400,
     "delta_c": 0.48, "concept_idx": (OFF_SUPPORT,), "concept_w": (1.0,),
     "delta_x": 0.0, "cov_idx": (), "noise": 1.0, "drop": ()},
    {"site_id": "site-08", "role": "concept_mild", "kind": "concept", "n_k": 430,
     "delta_c": 0.30, "concept_idx": CONCEPT_SUPPORT, "concept_w": (1.0, 0.55),
     "delta_x": 0.0, "cov_idx": (), "noise": 1.0, "drop": ()},
    {"site_id": "site-09", "role": "concept_mild", "kind": "concept", "n_k": 390,
     "delta_c": 0.30, "concept_idx": CONCEPT_SUPPORT, "concept_w": (1.0, 0.55),
     "delta_x": 0.0, "cov_idx": (), "noise": 1.0, "drop": ()},
    {"site_id": "site-10", "role": "cov_primary", "kind": "covariate", "n_k": 880,
     "delta_c": 0.0, "concept_idx": (), "concept_w": (),
     "delta_x": 0.75, "cov_idx": COVARIATE_SUPPORT, "noise": 1.0, "drop": ()},
    {"site_id": "site-11", "role": "cov_mild", "kind": "covariate", "n_k": 510,
     "delta_c": 0.0, "concept_idx": (), "concept_w": (),
     "delta_x": 0.28, "cov_idx": COVARIATE_SUPPORT, "noise": 1.0, "drop": ()},
    {"site_id": "site-12", "role": "cov_mild", "kind": "covariate", "n_k": 470,
     "delta_c": 0.0, "concept_idx": (), "concept_w": (),
     "delta_x": 0.28, "cov_idx": COVARIATE_SUPPORT, "noise": 1.0, "drop": ()},
    {"site_id": "site-13", "role": "cov_mild", "kind": "covariate", "n_k": 560,
     "delta_c": 0.0, "concept_idx": (), "concept_w": (),
     "delta_x": 0.28, "cov_idx": COVARIATE_SUPPORT, "noise": 1.0, "drop": ()},
    {"site_id": "site-14", "role": "cov_mild", "kind": "covariate", "n_k": 500,
     "delta_c": 0.0, "concept_idx": (), "concept_w": (),
     "delta_x": 0.28, "cov_idx": COVARIATE_SUPPORT, "noise": 1.40, "drop": ()},
    {"site_id": "site-15", "role": "cov_mild", "kind": "covariate", "n_k": 530,
     "delta_c": 0.0, "concept_idx": (), "concept_w": (),
     "delta_x": 0.28, "cov_idx": COVARIATE_SUPPORT, "noise": 1.0, "drop": ()},
    {"site_id": "site-16", "role": "cov_mild", "kind": "covariate", "n_k": 490,
     "delta_c": 0.0, "concept_idx": (), "concept_w": (),
     "delta_x": 0.28, "cov_idx": COVARIATE_SUPPORT, "noise": 1.0, "drop": ()},
    {"site_id": "site-17", "role": "cov_large_mild", "kind": "covariate", "n_k": 2600,
     "delta_c": 0.0, "concept_idx": (), "concept_w": (),
     "delta_x": 0.06, "cov_idx": COVARIATE_SUPPORT, "noise": 1.0, "drop": ()},
    {"site_id": "site-18", "role": "cov_local", "kind": "covariate", "n_k": 260,
     "delta_c": 0.0, "concept_idx": (), "concept_w": (),
     "delta_x": 1.80, "cov_idx": (LOCAL_FEATURE,), "noise": 1.0, "drop": ()},
    {"site_id": "site-19", "role": "cov_mild", "kind": "covariate", "n_k": 440,
     "delta_c": 0.0, "concept_idx": (), "concept_w": (),
     "delta_x": 0.28, "cov_idx": COVARIATE_SUPPORT, "noise": 1.0, "drop": (6,)},
]


def sample_x(rng: np.random.Generator, n: int, mean: np.ndarray, rho: float = RHO) -> np.ndarray:
    p = mean.size
    latent = rng.normal(size=(n, p))
    x = np.empty_like(latent)
    x[:, 0] = latent[:, 0]
    scale = float(np.sqrt(max(1.0 - rho ** 2, 0.0)))
    for j in range(1, p):
        x[:, j] = rho * x[:, j - 1] + scale * latent[:, j]
    return x + mean


def outcome_mean(x: np.ndarray, spec: dict, drifted: bool) -> np.ndarray:
    mean = 1.0 * x[:, 0] + 0.8 * x[:, 1] + 0.25 * x[:, 4]
    if drifted and spec["delta_c"]:
        for index, weight in zip(spec["concept_idx"], spec["concept_w"]):
            mean = mean + float(spec["delta_c"]) * float(weight) * x[:, index]
    return mean


def draw_batch(spec: dict, n: int, drifted: bool, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mean = np.zeros(P, dtype=float)
    if drifted:
        for index in spec["cov_idx"]:
            mean[int(index)] = float(spec["delta_x"])
    x = sample_x(rng, n, mean)
    y = outcome_mean(x, spec, drifted) + rng.normal(scale=float(spec["noise"]), size=n)
    return x, y


def feature_mask_of(spec: dict) -> np.ndarray:
    mask = np.ones(P, dtype=bool)
    for index in spec["drop"]:
        mask[int(index)] = False
    return mask


def _history_scalars(
    spec: dict,
    n: int,
    n_history: int,
    config: StatConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = feature_mask_of(spec)
    observed = np.flatnonzero(mask)
    hist_mmd = np.zeros(n_history, dtype=float)
    hist_r = np.zeros(n_history, dtype=float)
    for h in range(n_history):
        x_ref, y_ref = draw_batch(spec, n, False, seed + 1000 + 2 * h)
        x_new, y_new = draw_batch(spec, n, False, seed + 1000 + 2 * h + 1)
        mmd_value = scalar_mmd(
            x_ref[:, observed], x_new[:, observed], config.mmd_max_n, seed + 3000 + h
        )
        x_all = np.vstack([x_ref[:, observed], x_new[:, observed]])
        y_all = np.concatenate([y_ref, y_new])
        w_all = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
        r_value = r_risk_signal(x_all, y_all, w_all, config.ridge_alpha, seed + 4000 + h)
        hist_mmd[h], hist_r[h] = z_inputs(mmd_value, r_value)
    return hist_mmd, hist_r


def _bootstrap_d(
    x_ref: np.ndarray,
    x_new: np.ndarray,
    y_ref: np.ndarray,
    y_new: np.ndarray,
    mask: np.ndarray,
    config: StatConfig,
    seed: int,
    n_boot: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = np.zeros(n_boot, dtype=float)
    for b in range(n_boot):
        i_ref = rng.integers(0, len(x_ref), size=len(x_ref))
        i_new = rng.integers(0, len(x_new), size=len(x_new))
        grouped = group_vectors(
            x_ref[i_ref], x_new[i_new], y_ref[i_ref], y_new[i_new],
            mask, config, seed + 17 * (b + 1),
        )
        scores[b] = dominance(grouped["vimp_con"], grouped["vimp_cov"])
    lo, hi = np.quantile(scores, [0.025, 0.975])
    return float(lo), float(hi)


def analyze_site(
    spec: dict,
    seed: int,
    config: StatConfig,
    n_history: int,
    n_cap: int,
    n_boot: int = 0,
) -> dict:
    """One site, one window. The returned object is the upload packet."""
    n_analysis = int(min(spec["n_k"], n_cap))
    if n_analysis < 40:
        raise ValueError("analysis size too small for site {0}".format(spec["site_id"]))
    mask = feature_mask_of(spec)
    hist_mmd, hist_r = _history_scalars(spec, n_analysis, n_history, config, seed)
    x_ref, y_ref = draw_batch(spec, n_analysis, False, seed + 11)
    x_new, y_new = draw_batch(spec, n_analysis, True, seed + 29)
    grouped = group_vectors(x_ref, x_new, y_ref, y_new, mask, config, seed + 7)
    mmd_input, r_input = z_inputs(grouped["mmd"], grouped["r_signal"])
    z_mmd, _, _ = robust_z(mmd_input, hist_mmd)
    z_r, _, _ = robust_z(r_input, hist_r)
    std_mmd = placebo_std(hist_mmd)
    std_r = placebo_std(hist_r)
    # 0.30 is the written gate. At H≈24 the sd of a sd is about 0.15, so that
    # gate flags stationary sites. 0.80 keeps a clearly wild history out and
    # is what `calibration_*_ok` uses when config.calibration_tol is left open.
    strict = 0.30
    tol = config.calibration_tol
    cal_mmd = bool(np.isfinite(std_mmd) and abs(std_mmd - 1.0) <= tol)
    cal_r = bool(np.isfinite(std_r) and abs(std_r - 1.0) <= tol)
    strict_mmd = bool(np.isfinite(std_mmd) and abs(std_mmd - 1.0) <= strict)
    strict_r = bool(np.isfinite(std_r) and abs(std_r - 1.0) <= strict)
    d_value = dominance(grouped["vimp_con"], grouped["vimp_cov"])
    if n_boot > 0:
        ci = _bootstrap_d(x_ref, x_new, y_ref, y_new, mask, config, seed + 90, n_boot)
    else:
        ci = (d_value, d_value)
    packet = {
        "site_id": spec["site_id"],
        "vimp_cov": [float(v) for v in grouped["vimp_cov"]],
        "vimp_con": [float(v) for v in grouped["vimp_con"]],
        "z_mmd": float(z_mmd),
        "z_r": float(z_r),
        "calibration_ok": bool(cal_mmd and cal_r),
        "D_k": float(d_value),
        "D_k_ci": [float(ci[0]), float(ci[1])],
        "n_k": int(spec["n_k"]),
        "feature_mask": [bool(v) for v in mask],
        "diagnostics": {
            "n_analysis": n_analysis,
            "placebo_std_mmd": float(std_mmd),
            "placebo_std_r": float(std_r),
            "calibration_mmd_ok": cal_mmd,
            "calibration_r_ok": cal_r,
            "strict_mmd_ok": strict_mmd,
            "strict_r_ok": strict_r,
            "z_transform": "log_mmd+asinh_r",
            "mmd": float(grouped["mmd"]),
            "r_signal": float(grouped["r_signal"]),
            "bootstrap_B": int(n_boot),
            "top_con": int(np.argmax(grouped["vimp_con"])),
            "top_cov": int(np.argmax(grouped["vimp_cov"])),
        },
    }
    validate_upload(packet)
    return packet


def truth_table() -> List[dict]:
    rows = []
    for spec in CATALOG:
        rows.append({
            "site_id": spec["site_id"],
            "role": spec["role"],
            "kind": spec["kind"],
            "n_k": int(spec["n_k"]),
            "concept_idx": list(spec["concept_idx"]),
            "cov_idx": list(spec["cov_idx"]),
        })
    return rows


def oracle_vector(indices: Sequence[int], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    vec = np.zeros(P, dtype=float)
    if weights is None:
        weights = np.ones(len(indices))
    for index, weight in zip(indices, weights):
        vec[int(index)] = float(weight)
    return simplex(vec)


def pooled_vimp(seed: int, config: StatConfig, n_pool: int) -> dict:
    """One VIMP on a mixture drawn proportional to n_k. This is the raw-data baseline.

    It can rank features. It cannot name a site, and it spends O(sum n_k) rows.
    """
    weights = np.asarray([spec["n_k"] for spec in CATALOG], dtype=float)
    weights = weights / weights.sum()
    counts = np.maximum(12, np.floor(weights * n_pool).astype(int))
    # Put the remainder on the largest site so the pool has about n_pool rows.
    counts[int(np.argmax(weights))] += max(0, n_pool - int(counts.sum()))
    shared = np.ones(P, dtype=bool)
    for spec in CATALOG:
        shared &= feature_mask_of(spec)
    ref_x, ref_y, new_x, new_y = [], [], [], []
    for spec, count in zip(CATALOG, counts):
        x_ref, y_ref = draw_batch(spec, int(count), False, seed + 8000 + spec["n_k"])
        x_new, y_new = draw_batch(spec, int(count), True, seed + 9000 + spec["n_k"])
        ref_x.append(x_ref[:, shared])
        new_x.append(x_new[:, shared])
        ref_y.append(y_ref)
        new_y.append(y_new)
    mask = np.ones(int(shared.sum()), dtype=bool)
    grouped = group_vectors(
        np.vstack(ref_x), np.vstack(new_x),
        np.concatenate(ref_y), np.concatenate(new_y),
        mask, config, seed + 77,
    )
    full_con = np.zeros(P, dtype=float)
    full_cov = np.zeros(P, dtype=float)
    full_con[shared] = grouped["vimp_con"]
    full_cov[shared] = grouped["vimp_cov"]
    return {"vimp_con": simplex(full_con).tolist(), "vimp_cov": simplex(full_cov).tolist()}


def _mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def score_replicate(uploads: Sequence[dict], report: dict, pooled: Optional[dict] = None) -> dict:
    by_id = {packet["site_id"]: packet for packet in uploads}
    truth = {row["site_id"]: row for row in truth_table()}
    routed = {row["site_id"]: row for row in report["sites"]}
    oracle_con = oracle_vector(CONCEPT_SUPPORT, (1.0, 0.55))
    oracle_cov = oracle_vector(COVARIATE_SUPPORT, (1.0, 1.0))
    agg = report["aggregates"]
    out = {
        "D_agg": float(report["D_agg"]),
        "mean_abs_D": float(np.mean([abs(site["D_k"]) for site in report["sites"]])),
        "frontier_size": int(report["frontier_size_strength"]),
        "frontier_size_mixed_disagreement": int(report["frontier_size_disagreement"]),
        "n_primary_gated": int(report["n_primary_gated"]),
        "n_primary_literal": int(report["n_primary_literal"]),
        "shared_fraction": float(report["shared_fraction"]),
        "consistency_con_within": report["consistency"]["vimp_con_within_concept"],
        "consistency_con_all": report["consistency"]["vimp_con_all_sites"],
        "consistency_cov_within": report["consistency"]["vimp_cov_within_covariate"],
        "consistency_cov_all": report["consistency"]["vimp_cov_all_sites"],
        "spearman_mean_median_con": report["mean_median_spearman"]["vimp_con"],
        "spearman_mean_median_cov": report["mean_median_spearman"]["vimp_cov"],
        "calibration_fail_frac": float(np.mean([not packet["calibration_ok"] for packet in uploads])),
        "strict_calibration_fail_frac": float(np.mean([
            not (packet["diagnostics"]["strict_mmd_ok"] and packet["diagnostics"]["strict_r_ok"])
            for packet in uploads
        ])),
        "consensus_con": list(report["consensus_con"]),
        "consensus_cov": list(report["consensus_cov"]),
    }
    for name, key, oracle, truth_idx in (
        ("con_n", "vimp_con_n", oracle_con, CONCEPT_SUPPORT),
        ("con_median", "vimp_con_median", oracle_con, CONCEPT_SUPPORT),
        ("con_strength", "vimp_con_strength", oracle_con, CONCEPT_SUPPORT),
        ("cov_n", "vimp_cov_n", oracle_cov, COVARIATE_SUPPORT),
        ("cov_median", "vimp_cov_median", oracle_cov, COVARIATE_SUPPORT),
        ("cov_strength", "vimp_cov_strength", oracle_cov, COVARIATE_SUPPORT),
    ):
        vec = np.asarray(agg[key], dtype=float)
        out["prec_" + name] = precision_at_k(vec, truth_idx, 2)
        out["top1_" + name] = float(int(np.argmax(vec)) in set(truth_idx))
        out["mass_" + name] = feature_mass(vec, truth_idx)
        out["spear_" + name] = spearman_safe(vec, oracle)
    out["mass5_con_n"] = feature_mass(np.asarray(agg["vimp_con_n"]), (OFF_SUPPORT,))
    out["mass5_con_median"] = feature_mass(np.asarray(agg["vimp_con_median"]), (OFF_SUPPORT,))
    out["mass5_con_strength"] = feature_mass(np.asarray(agg["vimp_con_strength"]), (OFF_SUPPORT,))
    out["mass7_cov_n"] = feature_mass(np.asarray(agg["vimp_cov_n"]), (LOCAL_FEATURE,))
    out["mass7_cov_strength"] = feature_mass(np.asarray(agg["vimp_cov_strength"]), (LOCAL_FEATURE,))

    mild_con_prec, mild_con_mass = [], []
    mild_cov_prec, mild_cov_mass = [], []
    type_hit = []
    for row in truth_table():
        packet = by_id[row["site_id"]]
        if row["role"] == "concept_mild":
            mild_con_prec.append(precision_at_k(packet["vimp_con"], CONCEPT_SUPPORT, 2))
            mild_con_mass.append(feature_mass(packet["vimp_con"], CONCEPT_SUPPORT))
        if row["role"] == "cov_mild":
            mild_cov_prec.append(precision_at_k(packet["vimp_cov"], COVARIATE_SUPPORT, 2))
            mild_cov_mass.append(feature_mass(packet["vimp_cov"], COVARIATE_SUPPORT))
        if row["site_id"] in routed:
            type_hit.append(int(routed[row["site_id"]]["mechanism"] == row["kind"]))
    out["prec_con_mild_mean"] = _mean(mild_con_prec)
    out["mass_con_mild_mean"] = _mean(mild_con_mass)
    out["prec_cov_mild_mean"] = _mean(mild_cov_prec)
    out["mass_cov_mild_mean"] = _mean(mild_cov_mass)
    out["type_accuracy"] = _mean([float(v) for v in type_hit])

    def _route(site_id: str, field: str = "route") -> str:
        if site_id not in routed:
            return "excluded"
        return str(routed[site_id][field])

    named = {
        "concept_primary": "site-00",
        "cov_primary": "site-10",
        "local": "site-18",
        "breaker": "site-07",
        "large_mild": "site-17",
    }
    for label, site_id in named.items():
        out["route_" + label] = _route(site_id)
        out["literal_" + label] = _route(site_id, "route_literal")
        packet = by_id[site_id]
        out["z_" + label] = float(max(packet["z_mmd"], packet["z_r"]))
        out["D_" + label] = float(packet["D_k"])
        out["top_con_" + label] = int(np.argmax(packet["vimp_con"]))
        out["top_cov_" + label] = int(np.argmax(packet["vimp_cov"]))
        if site_id in routed:
            out["delta_strength_" + label] = float(routed[site_id]["delta_strength"])
            out["delta_dis_" + label] = float(routed[site_id]["delta_disagreement"])
        else:
            out["delta_strength_" + label] = float("nan")
            out["delta_dis_" + label] = float("nan")
    out["hit_concept_primary"] = out["route_concept_primary"] == "primary_driver"
    out["hit_cov_primary"] = out["route_cov_primary"] == "primary_driver"
    out["local_not_primary"] = out["route_local"] != "primary_driver"
    out["breaker_not_primary"] = out["route_breaker"] != "primary_driver"
    out["large_mild_not_primary"] = out["route_large_mild"] != "primary_driver"
    out["literal_hit_concept_primary"] = out["literal_concept_primary"] == "primary_driver"
    out["literal_hit_cov_primary"] = out["literal_cov_primary"] == "primary_driver"

    sites = report["sites"]
    out["argmax_z"] = max(sites, key=lambda s: s["z_k"])["site_id"]
    out["argmax_n"] = max(sites, key=lambda s: s["n_k"])["site_id"]
    out["argmax_delta_strength"] = max(sites, key=lambda s: s["delta_strength"])["site_id"]
    out["argmax_delta_disagreement"] = max(sites, key=lambda s: s["delta_disagreement"])["site_id"]
    out["argmax_z_role"] = truth[out["argmax_z"]]["role"]
    out["argmax_n_role"] = truth[out["argmax_n"]]["role"]
    out["argmax_delta_strength_role"] = truth[out["argmax_delta_strength"]]["role"]
    out["argmax_delta_disagreement_role"] = truth[out["argmax_delta_disagreement"]]["role"]

    primary_packet = by_id["site-00"]
    out["prec_con_primary_site"] = precision_at_k(primary_packet["vimp_con"], CONCEPT_SUPPORT, 2)
    out["mass_con_primary_site"] = feature_mass(primary_packet["vimp_con"], CONCEPT_SUPPORT)
    out["mass5_breaker_site"] = feature_mass(by_id["site-07"]["vimp_con"], (OFF_SUPPORT,))
    if pooled is not None:
        out["prec_con_pooled"] = precision_at_k(pooled["vimp_con"], CONCEPT_SUPPORT, 2)
        out["prec_cov_pooled"] = precision_at_k(pooled["vimp_cov"], COVARIATE_SUPPORT, 2)
        out["mass_con_pooled"] = feature_mass(np.asarray(pooled["vimp_con"]), CONCEPT_SUPPORT)
        out["mass_cov_pooled"] = feature_mass(np.asarray(pooled["vimp_cov"]), COVARIATE_SUPPORT)
        out["mass5_con_pooled"] = feature_mass(np.asarray(pooled["vimp_con"]), (OFF_SUPPORT,))
        out["spear_con_pooled"] = spearman_safe(np.asarray(pooled["vimp_con"]), oracle_con)
        out["spear_cov_pooled"] = spearman_safe(np.asarray(pooled["vimp_cov"]), oracle_cov)
    raw_bytes = sum(spec["n_k"] * P * 8 * 2 * 2 for spec in CATALOG)  # X ref/new, Y ignored in the 2*2
    # X ref, X new, Y ref, Y new:
    raw_bytes = sum(spec["n_k"] * (P + 1) * 8 * 2 for spec in CATALOG)
    upload_bytes = len(uploads) * (P * 2 * 8 + 8 * 8)
    out["raw_bytes"] = int(raw_bytes)
    out["upload_bytes"] = int(upload_bytes)
    return out


def run_replicate(
    seed: int,
    config: Optional[StatConfig] = None,
    n_history: int = 24,
    n_cap: int = 420,
    n_boot: int = 0,
    with_pooled: bool = True,
    progress: bool = False,
) -> dict:
    config = config or StatConfig()
    t0 = time.perf_counter()
    uploads = []
    for index, spec in enumerate(CATALOG):
        t_site = time.perf_counter()
        packet = analyze_site(spec, seed + 1009 * (index + 1), config, n_history, n_cap, n_boot=n_boot)
        uploads.append(packet)
        if progress:
            print(
                "{sid} z_mmd={zm:.2f} z_r={zr:.2f} D={d:.2f} top_con={tc} top_cov={tv} cal={cal} std={sm:.2f}/{sr:.2f} ({sec:.1f}s)".format(
                    sid=packet["site_id"],
                    zm=packet["z_mmd"],
                    zr=packet["z_r"],
                    d=packet["D_k"],
                    tc=packet["diagnostics"]["top_con"],
                    tv=packet["diagnostics"]["top_cov"],
                    cal="{0}/{1}".format(
                        int(packet["diagnostics"]["calibration_mmd_ok"]),
                        int(packet["diagnostics"]["calibration_r_ok"]),
                    ),
                    sm=packet["diagnostics"]["placebo_std_mmd"],
                    sr=packet["diagnostics"]["placebo_std_r"],
                    sec=time.perf_counter() - t_site,
                ),
                flush=True,
            )
    report = aggregate_uploads(uploads)
    pooled = pooled_vimp(seed, config, n_pool=n_cap) if with_pooled else None
    metrics = score_replicate(uploads, report, pooled)
    metrics["seconds"] = float(time.perf_counter() - t0)
    return {"uploads": uploads, "report": report, "pooled": pooled, "metrics": metrics}


def ablate_z_scaling(config: Optional[StatConfig] = None, n_history: int = 16) -> dict:
    """Show that a test-style z at the full sample size promotes the large mild site.

    The same mild concept shift is scored at n=400 and n=2000. History uses
    the same n as the current window, which is what a per-site test does
    when it is not capped. The population shift does not change.
    """
    config = config or StatConfig()
    spec = {
        "site_id": "ablation", "role": "ablation", "kind": "concept", "n_k": 2000,
        "delta_c": 0.35, "concept_idx": CONCEPT_SUPPORT, "concept_w": (1.0, 0.55),
        "delta_x": 0.0, "cov_idx": (), "noise": 1.0, "drop": (),
    }
    out = {}
    for n in (400, 2000):
        hist_mmd, hist_r = _history_scalars(spec, n, n_history, config, seed=91000 + n)
        x_ref, y_ref = draw_batch(spec, n, False, 92000 + n)
        x_new, y_new = draw_batch(spec, n, True, 93000 + n)
        observed = np.arange(P)
        mmd = scalar_mmd(x_ref[:, observed], x_new[:, observed], config.mmd_max_n, 94000 + n)
        x_all = np.vstack([x_ref, x_new])
        y_all = np.concatenate([y_ref, y_new])
        w_all = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
        # R-risk sees the full n; MMD is kernel-capped inside scalar_mmd.
        r_value = r_risk_signal(x_all, y_all, w_all, config.ridge_alpha, 95000 + n)
        z_mmd_input, z_r_input = z_inputs(mmd, r_value)
        z_mmd, _, _ = robust_z(z_mmd_input, hist_mmd)
        z_r, _, _ = robust_z(z_r_input, hist_r)
        out["n{0}_z_mmd".format(n)] = float(z_mmd)
        out["n{0}_z_r".format(n)] = float(z_r)
        out["n{0}_z".format(n)] = float(max(z_mmd, z_r))
    return out


def save_scatter(report: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sites = report["sites"]
    route_color = {
        "primary_driver": "#b00020",
        "local_anomaly": "#e07a00",
        "consensus_breaker": "#5b2c6f",
        "stable_anchor": "#1b7f4e",
        "no_action": "#8a8f98",
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for mechanism, marker in (("concept", "o"), ("covariate", "s")):
        rows = [s for s in sites if s["mechanism"] == mechanism]
        ax.scatter(
            [s["z_match"] for s in rows],
            [s["delta_strength"] for s in rows],
            c=[route_color.get(s["route"], "#555555") for s in rows],
            marker=marker,
            s=70,
            label=mechanism,
            zorder=3,
        )
    labeled = {"primary_driver", "local_anomaly", "consensus_breaker"}
    for site in sites:
        if site["route"] in labeled or site["on_frontier_strength"]:
            ax.annotate(site["site_id"].replace("site-", ""), (site["z_match"], site["delta_strength"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
    z_cut = 2.0
    deltas = [s["delta_strength"] for s in sites]
    ax.axvline(z_cut, color="#444", lw=0.8, ls="--")
    ax.axhline(float(np.median(deltas)), color="#444", lw=0.8, ls=":")
    ax.set_xlabel("mechanism-matched z")
    ax.set_ylabel("strength  n × z_match")
    ax.set_title("Gated routes. Color is the action, marker is the estimated mechanism.")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def summarize_metrics(rows: Sequence[dict]) -> dict:
    keys = [key for key in rows[0] if isinstance(rows[0][key], (int, float, bool))]
    summary = {"n_reps": len(rows)}
    for key in keys:
        vals = []
        for row in rows:
            value = row[key]
            if isinstance(value, bool):
                vals.append(float(value))
            elif isinstance(value, (int, float)) and np.isfinite(value):
                vals.append(float(value))
        if not vals:
            continue
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)}
    # Modal site picked by the argmax rules.
    for key in ("argmax_z", "argmax_n", "argmax_delta_strength", "argmax_delta_disagreement",
                "route_concept_primary", "route_cov_primary", "route_local", "route_breaker", "route_large_mild"):
        if key not in rows[0]:
            continue
        counts: Dict[str, int] = {}
        for row in rows:
            counts[str(row[key])] = counts.get(str(row[key]), 0) + 1
        summary[key] = counts
    return summary
