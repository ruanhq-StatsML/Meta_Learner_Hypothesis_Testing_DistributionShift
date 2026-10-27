"""Frozen semantics and acceptance thresholds.

Thresholds are the evaluation protocol. They are not adjusted after
results are seen.
"""

from __future__ import annotations

SEMANTIC_DECLARATION = (
    "This is a conditional-distribution counterfactual, not a structural "
    "causal effect. Path channels (the separate routes that could carry "
    "the association) are not identifiable. MMD has no physical unit and "
    "is reported as a ratio to the no-intervention baseline. Steps that "
    "leave the reference support are excluded from the final attribution. "
    "Aggregate scores describe the queried cohort distribution, not a "
    "site-level causal share."
)

SEMANTIC_BULLETS_ZH = [
    "这是条件分布反事实，不是结构因果效应。",
    "路径不可分解：风控、推荐、社交或其他传导通道不能从该分数里分开。",
    "MMD 无单位，报告相对无干预基线的倍数。",
    "支持集外的步骤不进入最终归因。",
    "分数描述的是被查询客群的分布对比，不是站点级因果份额。",
]

# Monte Carlo slack for the random-Fourier MMD peak test. Fixed here,
# not tuned on the experimental results.
RFF_PEAK_SLACK = 1.05

# Bootstrap replicates for the reported interval. The resample redraws
# held-out members and conditional outcome draws; it does not retrain.
N_BOOT = 200

# Primary generative model is the conditional DDPM. The conditional
# RealNVP is the architecture control, not a second chance to pass L3.
PRIMARY_MODEL = "ddpm"


class Thresholds:
    """Acceptance cuts copied from the evaluation protocol."""

    mvp_stop_concordance = 0.6
    mvp_go_concordance = 0.7
    mvp_noise_ratio = 1.25

    l1_concordance = 0.8
    l1_spearman = 0.7
    l1_lag_spearman = -0.7
    l1_isolation_effect = -1.0
    l1_isolation_p = 0.05
    l1_min_passes = 3
    l1_min_events = 4

    l2_ratio = 2.0

    l3_ks_p = 0.05
    l3_wasserstein_std = 0.1
    l3_hsic_low = 0.8
    l3_hsic_high = 1.2

    l4_min_agree = 3
    l4_max_disagree = 2

    l5_cv_baseline = 0.30
    l5_cv_kernel = 0.20
    l5_cv_architecture = 0.30
    l5_evalue = 2.0
    l5_evalue_fragile = 1.5


def coefficient_of_variation(values) -> float:
    import numpy as np

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    mu = float(np.mean(x))
    if abs(mu) < 1e-12:
        return float("inf") if float(np.std(x)) > 0 else 0.0
    return float(np.std(x, ddof=1) / abs(mu))
