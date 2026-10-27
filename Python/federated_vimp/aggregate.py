"""Server-side aggregation, diagnostics, and primary-driver routing.

Two influence numbers are kept on purpose.

* ``delta_disagreement`` is the L1 change in the n-weighted simplex when a
  site is removed. A site that shares the fleet ranking has a small value
  even if its shift is large, because simplex vectors have no magnitude.
* ``delta_strength`` is ``n_k * max(z_k, 0)``. It is the L1 of that site's
  contribution to the strength-weighted sum, and it does carry magnitude.

The literal roadmap route uses disagreement. The recommended route uses
strength, then refuses the primary-driver label unless the site's leading
feature sits in the consensus set of the strength-weighted aggregate.
That gate is what stops a small private spike, and a large off-support
site, from being called the fleet driver.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr


def _as_float_matrix(rows: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=float)


def weighted_mean(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return (w[:, None] * np.asarray(stack, dtype=float)).sum(axis=0)


def loso_l1(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """L1 change of the weighted mean when each row is removed."""
    stack = np.asarray(stack, dtype=float)
    w = np.asarray(weights, dtype=float)
    out = np.zeros(len(stack), dtype=float)
    full = weighted_mean(stack, w)
    for k in range(len(stack)):
        mask = np.ones(len(stack), dtype=bool)
        mask[k] = False
        if not np.any(mask) or float(w[mask].sum()) <= 0:
            out[k] = float(np.abs(full).sum())
            continue
        held = weighted_mean(stack[mask], w[mask])
        out[k] = float(np.abs(full - held).sum())
    return out


def spearman_safe(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    rho = spearmanr(left, right).statistic
    if rho is None or not np.isfinite(rho):
        return float("nan")
    return float(rho)


def mean_offdiag_spearman(stack: np.ndarray) -> float:
    stack = np.asarray(stack, dtype=float)
    k = stack.shape[0]
    if k < 2:
        return float("nan")
    vals = []
    for i in range(k):
        for j in range(i + 1, k):
            rho = spearman_safe(stack[i], stack[j])
            if np.isfinite(rho):
                vals.append(rho)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def pareto_mask(z_values: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """True where no other point is at least as high on both axes and higher on one.

    Ineligible rows (non-finite) are left False. Higher z and higher delta
    are both better.
    """
    z_values = np.asarray(z_values, dtype=float)
    delta = np.asarray(delta, dtype=float)
    n = len(z_values)
    on = np.zeros(n, dtype=bool)
    finite = np.isfinite(z_values) & np.isfinite(delta)
    for i in range(n):
        if not finite[i]:
            continue
        dominated = False
        for j in range(n):
            if i == j or not finite[j]:
                continue
            ge = z_values[j] >= z_values[i] and delta[j] >= delta[i]
            gt = z_values[j] > z_values[i] or delta[j] > delta[i]
            if ge and gt:
                dominated = True
                break
        on[i] = not dominated
    return on


def precision_at_k(scores: np.ndarray, truth: Sequence[int], k: int = 2) -> float:
    top = set(np.argsort(-np.asarray(scores, dtype=float))[:k].tolist())
    return len(top & set(int(j) for j in truth)) / float(k)


def feature_mass(scores: np.ndarray, indices: Sequence[int]) -> float:
    if not len(indices):
        return 0.0
    return float(np.sum(np.asarray(scores, dtype=float)[list(indices)]))


def _top_index(scores: np.ndarray) -> int:
    return int(np.argmax(scores))


def aggregate_uploads(
    uploads: Sequence[Mapping],
    *,
    min_n: int = 200,
    z_primary: float = 2.0,
    z_quiet: float = 1.0,
    top_k: int = 2,
    freq_consensus: float = 0.5,
    freq_local: float = 0.2,
    anchor_abs_d: float = 0.10,
) -> Dict:
    """Pool eligible uploads. Does not read ground-truth roles or mechanisms."""
    if not uploads:
        raise ValueError("no uploads")
    p = len(uploads[0]["vimp_cov"])
    eligible_idx = []
    for i, packet in enumerate(uploads):
        n_k = int(packet["n_k"])
        # Sample size gates the pool. Calibration gates the frontier only:
        # a skewed null on the unused mechanism must not drop the site.
        ok = n_k >= min_n
        if ok and len(packet["vimp_cov"]) == p and len(packet["feature_mask"]) == p:
            eligible_idx.append(i)
    if len(eligible_idx) < 2:
        raise ValueError("fewer than two eligible sites")

    mask = np.ones(p, dtype=bool)
    for i in eligible_idx:
        mask &= np.asarray(uploads[i]["feature_mask"], dtype=bool)
    shared = np.flatnonzero(mask)
    shared_fraction = float(len(shared) / p)
    alarm_shared = shared_fraction < 0.5

    def _take(i: int, key: str) -> np.ndarray:
        vec = np.asarray(uploads[i][key], dtype=float)[shared]
        total = float(vec.sum())
        if total <= 1e-12:
            return np.full(len(shared), 1.0 / len(shared))
        return vec / total

    cov = np.vstack([_take(i, "vimp_cov") for i in eligible_idx])
    con = np.vstack([_take(i, "vimp_con") for i in eligible_idx])
    n_k = np.asarray([float(uploads[i]["n_k"]) for i in eligible_idx])
    z_mmd = np.asarray([float(uploads[i]["z_mmd"]) for i in eligible_idx])
    z_r = np.asarray([float(uploads[i]["z_r"]) for i in eligible_idx])
    z_k = np.maximum(z_mmd, z_r)
    d_k = np.asarray([float(uploads[i]["D_k"]) for i in eligible_idx])
    site_ids = [str(uploads[i]["site_id"]) for i in eligible_idx]

    con_n = weighted_mean(con, n_k)
    cov_n = weighted_mean(cov, n_k)
    con_med = np.median(con, axis=0)
    cov_med = np.median(cov, axis=0)
    con_uniform = con.mean(axis=0)
    cov_uniform = cov.mean(axis=0)

    # Strength-weighted contributions. z_r gates concept mass, z_mmd gates
    # covariate mass, so a pure concept site does not vote on vimp_cov.
    strength_con = n_k * np.maximum(z_r, 0.0)
    strength_cov = n_k * np.maximum(z_mmd, 0.0)
    con_strength = weighted_mean(con, np.maximum(strength_con, 1e-8))
    cov_strength = weighted_mean(cov, np.maximum(strength_cov, 1e-8))

    delta_dis_con = loso_l1(con, n_k)
    delta_dis_cov = loso_l1(cov, n_k)
    delta_disagreement = np.maximum(delta_dis_con, delta_dis_cov)
    # Matched strength: a concept site is scored on z_r, a covariate site on
    # z_mmd. One mixed axis lets the louder mechanism dominate the other.
    z_match = np.where(d_k >= 0.0, z_r, z_mmd)
    delta_strength = n_k * np.maximum(z_match, 0.0)
    delta_strength_mixed = n_k * np.maximum(z_k, 0.0)

    # Consensus from frequency inside the mechanism the site itself claims,
    # and from the strength-weighted vector. A feature must clear the
    # frequency bar; otherwise the strength vector's own top coordinate is
    # the fallback so the gate is never empty.
    mechanism = np.where(d_k >= 0.0, "concept", "covariate")
    freq_con = _topk_frequency(con, mechanism == "concept", top_k)
    freq_cov = _topk_frequency(cov, mechanism == "covariate", top_k)
    freq_con_all = _topk_frequency(con, np.ones(len(con), dtype=bool), top_k)
    freq_cov_all = _topk_frequency(cov, np.ones(len(cov), dtype=bool), top_k)
    consensus_con = _consensus_indices(freq_con, con_strength, freq_consensus)
    consensus_cov = _consensus_indices(freq_cov, cov_strength, freq_consensus)

    contrib_con = strength_con[:, None] * con
    contrib_cov = strength_cov[:, None] * cov
    influence_con = contrib_con[:, consensus_con].sum(axis=1) if len(consensus_con) else np.zeros(len(con))
    influence_cov = contrib_cov[:, consensus_cov].sum(axis=1) if len(consensus_cov) else np.zeros(len(cov))
    consensus_influence = influence_con + influence_cov
    total_strength_mass = contrib_con.sum(axis=1) + contrib_cov.sum(axis=1)
    discordant_mass = np.maximum(total_strength_mass - consensus_influence, 0.0)

    mean_median_spearman = {
        "vimp_con": spearman_safe(con_n, con_med),
        "vimp_cov": spearman_safe(cov_n, cov_med),
    }
    consistency = {
        "vimp_con_within_concept": mean_offdiag_spearman(con[mechanism == "concept"]),
        "vimp_cov_within_covariate": mean_offdiag_spearman(cov[mechanism == "covariate"]),
        "vimp_con_all_sites": mean_offdiag_spearman(con),
        "vimp_cov_all_sites": mean_offdiag_spearman(cov),
    }
    outlier = _outlier_site(con, cov, site_ids)

    cal_match = np.asarray([
        _matched_calibration(uploads[eligible_idx[i]], str(mechanism[i]))
        for i in range(len(site_ids))
    ], dtype=bool)
    z_frontier = np.where(cal_match, z_match, np.nan)
    delta_frontier = np.where(cal_match, delta_strength, np.nan)
    med_strength = float(np.median(delta_strength[cal_match])) if np.any(cal_match) else float(np.median(delta_strength))
    med_dis = float(np.median(delta_disagreement))
    med_discord = float(np.median(discordant_mass[cal_match])) if np.any(cal_match) else float(np.median(discordant_mass))
    med_n = float(np.median(n_k))
    frontier_strength = _frontier_within(mechanism, z_frontier, delta_frontier)
    frontier_disagreement = pareto_mask(z_k, delta_disagreement)

    lead_feature = []
    lead_in_consensus = []
    for i in range(len(site_ids)):
        if z_r[i] >= z_mmd[i]:
            local = con[i]
            ok = _top_index(local) in set(consensus_con.tolist())
            lead_feature.append(int(shared[_top_index(local)]))
        else:
            local = cov[i]
            ok = _top_index(local) in set(consensus_cov.tolist())
            lead_feature.append(int(shared[_top_index(local)]))
        lead_in_consensus.append(bool(ok))

    literal = [
        "unreliable" if not bool(cal_match[i]) else _route_literal(
            z=float(z_k[i]),
            delta=float(delta_disagreement[i]),
            on_frontier=bool(frontier_disagreement[i]),
            d_value=float(d_k[i]),
            med=med_dis,
            z_primary=z_primary,
            z_quiet=z_quiet,
            anchor_abs_d=anchor_abs_d,
        )
        for i in range(len(site_ids))
    ]
    gated = [
        "unreliable" if not bool(cal_match[i]) else _route_gated(
            z=float(z_match[i]),
            delta=float(delta_strength[i]),
            discordant=float(discordant_mass[i]),
            on_frontier=bool(frontier_strength[i]),
            lead_ok=bool(lead_in_consensus[i]),
            d_value=float(d_k[i]),
            n_k=float(n_k[i]),
            med_delta=med_strength,
            med_discord=med_discord,
            med_n=med_n,
            z_primary=z_primary,
            z_quiet=z_quiet,
            anchor_abs_d=anchor_abs_d,
        )
        for i in range(len(site_ids))
    ]

    n_concept = int(np.sum(mechanism == "concept"))
    n_covariate = int(np.sum(mechanism == "covariate"))
    n_joint = int(np.sum((z_mmd > z_primary) & (z_r > z_primary)))
    d_agg = float(
        (np.max(con_strength) - np.max(cov_strength))
        / (np.max(con_strength) + np.max(cov_strength) + 1e-8)
    )

    def _full(shared_vec: np.ndarray) -> List[float]:
        full = np.zeros(p, dtype=float)
        full[shared] = shared_vec
        total = float(full.sum())
        if total > 0:
            full = full / total
        return full.tolist()

    sites = []
    for i in range(len(site_ids)):
        sites.append({
            "site_id": site_ids[i],
            "n_k": int(n_k[i]),
            "z_mmd": float(z_mmd[i]),
            "z_r": float(z_r[i]),
            "z_k": float(z_k[i]),
            "D_k": float(d_k[i]),
            "mechanism": str(mechanism[i]),
            "delta_disagreement": float(delta_disagreement[i]),
            "delta_strength": float(delta_strength[i]),
            "delta_strength_mixed": float(delta_strength_mixed[i]),
            "z_match": float(z_match[i]),
            "consensus_influence": float(consensus_influence[i]),
            "discordant_mass": float(discordant_mass[i]),
            "on_frontier_strength": bool(frontier_strength[i]),
            "on_frontier_disagreement": bool(frontier_disagreement[i]),
            "lead_feature": int(lead_feature[i]),
            "lead_in_consensus": bool(lead_in_consensus[i]),
            "calibration_match": bool(cal_match[i]),
            "route_literal": literal[i],
            "route": gated[i],
            "top_con": int(shared[int(np.argmax(con[i]))]),
            "top_cov": int(shared[int(np.argmax(cov[i]))]),
        })

    excluded = []
    for i, packet in enumerate(uploads):
        if i in eligible_idx:
            continue
        reason = "insufficient_sample" if int(packet["n_k"]) < min_n else "malformed"
        excluded.append({"site_id": str(packet["site_id"]), "reason": reason, "n_k": int(packet["n_k"])})

    return {
        "semantic": (
            "Conditional association only. D_agg is a fleet label, not a site share. "
            "vimp_cov and vimp_con are not merged."
        ),
        "p": p,
        "shared_features": shared.astype(int).tolist(),
        "shared_fraction": shared_fraction,
        "alarm_shared_features": alarm_shared,
        "alarm_mean_median": bool(
            (not np.isfinite(mean_median_spearman["vimp_con"]))
            or mean_median_spearman["vimp_con"] < 0.8
            or (not np.isfinite(mean_median_spearman["vimp_cov"]))
            or mean_median_spearman["vimp_cov"] < 0.8
        ),
        "mean_median_spearman": mean_median_spearman,
        "consistency": consistency,
        "alarm_consistency_within": bool(
            (np.isfinite(consistency["vimp_con_within_concept"]) and consistency["vimp_con_within_concept"] < 0.5)
            or (np.isfinite(consistency["vimp_cov_within_covariate"]) and consistency["vimp_cov_within_covariate"] < 0.5)
        ),
        "outlier_site": outlier,
        "n_concept": n_concept,
        "n_covariate": n_covariate,
        "n_joint": n_joint,
        "D_agg": d_agg,
        "z_values": z_k.tolist(),
        "aggregates": {
            "vimp_con_n": _full(con_n),
            "vimp_cov_n": _full(cov_n),
            "vimp_con_median": _full(con_med),
            "vimp_cov_median": _full(cov_med),
            "vimp_con_uniform": _full(con_uniform),
            "vimp_cov_uniform": _full(cov_uniform),
            "vimp_con_strength": _full(con_strength),
            "vimp_cov_strength": _full(cov_strength),
        },
        "freq_con_within": _freq_full(p, shared, freq_con),
        "freq_cov_within": _freq_full(p, shared, freq_cov),
        "freq_con_all": _freq_full(p, shared, freq_con_all),
        "freq_cov_all": _freq_full(p, shared, freq_cov_all),
        "consensus_con": shared[consensus_con].astype(int).tolist(),
        "consensus_cov": shared[consensus_cov].astype(int).tolist(),
        "local_con": shared[np.flatnonzero(freq_con < freq_local)].astype(int).tolist(),
        "local_cov": shared[np.flatnonzero(freq_cov < freq_local)].astype(int).tolist(),
        "sites": sites,
        "excluded": excluded,
        "frontier_size_strength": int(frontier_strength.sum()),
        "frontier_size_disagreement": int(frontier_disagreement.sum()),
        "n_primary_gated": int(sum(route == "primary_driver" for route in gated)),
        "n_primary_literal": int(sum(route == "primary_driver" for route in literal)),
        "route_counts": _counts(gated),
        "literal_route_counts": _counts(literal),
    }


def _topk_frequency(stack: np.ndarray, row_mask: np.ndarray, top_k: int) -> np.ndarray:
    freq = np.zeros(stack.shape[1], dtype=float)
    rows = np.flatnonzero(row_mask)
    if rows.size == 0:
        return freq
    for i in rows:
        top = np.argsort(-stack[i])[:top_k]
        freq[top] += 1.0
    freq /= float(rows.size)
    return freq


def _consensus_indices(freq: np.ndarray, strength: np.ndarray, cut: float) -> np.ndarray:
    chosen = np.flatnonzero(freq >= cut)
    if chosen.size == 0:
        chosen = np.asarray([int(np.argmax(strength))], dtype=int)
    return chosen.astype(int)


def _freq_full(p: int, shared: np.ndarray, freq_shared: np.ndarray) -> List[float]:
    full = np.zeros(p, dtype=float)
    full[shared] = freq_shared
    return full.tolist()


def _outlier_site(con: np.ndarray, cov: np.ndarray, site_ids: Sequence[str]) -> Optional[str]:
    # Lowest average correlation of the concatenated profile with the other sites.
    profile = np.hstack([con, cov])
    k = len(site_ids)
    if k < 3:
        return None
    mean_rho = np.zeros(k, dtype=float)
    for i in range(k):
        vals = []
        for j in range(k):
            if i == j:
                continue
            rho = spearman_safe(profile[i], profile[j])
            if np.isfinite(rho):
                vals.append(rho)
        mean_rho[i] = float(np.mean(vals)) if vals else float("nan")
    if not np.any(np.isfinite(mean_rho)):
        return None
    return str(site_ids[int(np.nanargmin(mean_rho))])


def _matched_calibration(packet: Mapping, mechanism: str) -> bool:
    """Trust z only for the mechanism whose own placebo check passed.

    ``calibration_ok`` is the strict AND of both checks and stays on the
    upload. The server does not drop a concept site because its null MMD
    history is skewed.
    """
    diag = packet.get("diagnostics") or {}
    if mechanism == "concept" and "calibration_r_ok" in diag:
        return bool(diag["calibration_r_ok"])
    if mechanism == "covariate" and "calibration_mmd_ok" in diag:
        return bool(diag["calibration_mmd_ok"])
    return bool(packet["calibration_ok"])


def _frontier_within(mechanism: np.ndarray, z_axis: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Pareto frontier computed separately inside each estimated mechanism."""
    on = np.zeros(len(mechanism), dtype=bool)
    for label in ("concept", "covariate"):
        idx = np.flatnonzero(mechanism == label)
        if idx.size == 0:
            continue
        local = pareto_mask(z_axis[idx], delta[idx])
        on[idx] = local
    return on


def _route_literal(
    z: float,
    delta: float,
    on_frontier: bool,
    d_value: float,
    med: float,
    z_primary: float,
    z_quiet: float,
    anchor_abs_d: float,
) -> str:
    if on_frontier and z > z_primary and delta > med:
        return "primary_driver"
    if z > z_primary and delta <= med:
        return "local_anomaly"
    if z < z_quiet and delta > med:
        return "consensus_breaker"
    if z <= z_quiet and delta <= med and abs(d_value) <= anchor_abs_d:
        return "stable_anchor"
    return "no_action"


def _route_gated(
    z: float,
    delta: float,
    discordant: float,
    on_frontier: bool,
    lead_ok: bool,
    d_value: float,
    n_k: float,
    med_delta: float,
    med_discord: float,
    med_n: float,
    z_primary: float,
    z_quiet: float,
    anchor_abs_d: float,
) -> str:
    """Strength Pareto, then a consensus-feature gate.

    High strength on a private feature is not the fleet driver. If that
    private mass comes from a large site, the route is consensus breaker
    (split axis / feature definition). If it comes from a small site, the
    route is local anomaly (watch this site, do not retrain the fleet).
    """
    if on_frontier and z > z_primary and delta > med_delta and lead_ok:
        return "primary_driver"
    private = (not lead_ok) and (discordant > med_discord or delta > med_delta)
    if private and z > z_primary and n_k < med_n:
        return "local_anomaly"
    if private and n_k >= med_n:
        return "consensus_breaker"
    # A detectable but low-influence site is a local anomaly only when its
    # leading feature is private. An aligned mild site stays no_action.
    if z > z_primary and delta <= med_delta and not lead_ok:
        return "local_anomaly"
    if z < z_quiet and discordant > med_discord:
        return "consensus_breaker"
    if z <= z_quiet and delta <= med_delta and abs(d_value) <= anchor_abs_d:
        return "stable_anchor"
    return "no_action"


def _counts(routes: Sequence[str]) -> Dict[str, int]:
    keys = (
        "primary_driver",
        "local_anomaly",
        "consensus_breaker",
        "stable_anchor",
        "no_action",
        "unreliable",
    )
    return {key: int(sum(route == key for route in routes)) for key in keys}


def render_executive(report: Mapping) -> str:
    """One page a business reader can use without the ground truth."""
    primaries = [s for s in report["sites"] if s["route"] == "primary_driver"]
    locals_ = [s for s in report["sites"] if s["route"] == "local_anomaly"]
    breakers = [s for s in report["sites"] if s["route"] == "consensus_breaker"]
    lines = [
        report["semantic"],
        "",
        "先处理的站点（gated primary driver）",
    ]
    if not primaries:
        lines.append("- 没有站点同时满足：Pareto 前沿、z>2、强度高于中位数、主导特征落在共识集。")
    for site in sorted(primaries, key=lambda s: -s["z_k"]):
        action = _action_for(site["mechanism"])
        lines.append(
            "- {sid}: 机制={mech}，z={z:.2f}，强度 Δ={d:.0f}，主导特征指数={feat}。动作：{act}".format(
                sid=site["site_id"],
                mech=site["mechanism"],
                z=site["z_match"],
                d=site["delta_strength"],
                feat=site["lead_feature"],
                act=action,
            )
        )
    lines.append("")
    lines.append("不要当成机群主驱动")
    if locals_:
        lines.append(
            "- 局部异常（z 高，但强度不够或主导特征不在共识集）: "
            + ", ".join(s["site_id"] for s in locals_)
            + "。动作：本站监控，不要全网重训。"
        )
    if breakers:
        lines.append(
            "- 共识破坏者（贡献主要落在非共识特征）: "
            + ", ".join(s["site_id"] for s in breakers)
            + "。动作：查 split 轴和特征定义，不要用它的 VIMP 代表机群。"
        )
    if not locals_ and not breakers:
        lines.append("- 无。")
    lines.append("")
    lines.append(
        "共识特征指数 concept={con}，covariate={cov}。D_agg={d:.3f}（机群标签，不是份额）。".format(
            con=report["consensus_con"],
            cov=report["consensus_cov"],
            d=report["D_agg"],
        )
    )
    lines.append(
        "漂移类型计数（由 D_k 符号，不是由真值）: concept={c}, covariate={v}, joint(z_mmd>2 且 z_r>2)={j}。".format(
            c=report["n_concept"],
            v=report["n_covariate"],
            j=report["n_joint"],
        )
    )
    if report["alarm_shared_features"]:
        lines.append("警报：共享特征比例 {0:.2f} < 0.5，聚合排序不可靠。".format(report["shared_fraction"]))
    if report["alarm_mean_median"]:
        lines.append("警报：n-加权均值和中位数的 Spearman < 0.8，少数站点在拉均值。以中位数和强度加权为准，不要只报 n-加权。")
    if report["alarm_consistency_within"]:
        lines.append("警报：机制组内的平均 Spearman < 0.5。")
    if report["excluded"]:
        lines.append(
            "未进排序: " + ", ".join("{0}({1})".format(s["site_id"], s["reason"]) for s in report["excluded"])
        )
    return "\n".join(lines) + "\n"


def _action_for(mechanism: str) -> str:
    if mechanism == "concept":
        return "触发该站局部重训。概念漂移是 P(Y|X) 变了，上游管道通常是干净的。"
    if mechanism == "covariate":
        return "检查该站上游数据管道。协变量漂移是 P(X) 变了，先不要全网重训。"
    return "两种漂移都强，先拆开再决定重训还是查管道。"
