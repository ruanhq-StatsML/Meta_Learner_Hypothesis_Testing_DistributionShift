"""Rollout, attribution, and the five-layer evaluation protocol."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from cf_attribution.dgp import Panel, oracle_outcome
from cf_attribution.metrics import (
    cohens_d,
    evalue_from_smd,
    hsic_ratio,
    ks_uniform_p,
    median_bandwidth,
    mmd2_rff,
    pairwise_concordance,
    randomized_pit,
    spearman,
    split_noise_floor,
    wasserstein_nd,
    welch_p,
)
from cf_attribution.models import ConditionalDDPM, ConditionalRealNVP, QuantileOutcome
from cf_attribution.semantics import (
    N_BOOT,
    PRIMARY_MODEL,
    RFF_PEAK_SLACK,
    SEMANTIC_DECLARATION,
    Thresholds,
    coefficient_of_variation,
)

LAGS = list(range(1, 15))
T = Thresholds()


@dataclass
class Config:
    ddpm_epochs: int = 36
    flow_epochs: int = 36
    diff_steps: int = 16
    hidden: int = 96
    gbr_iter: int = 50
    n_boot: int = N_BOOT
    rff: int = 96
    batch: int = 128
    seed: int = 2026
    panels: tuple[str, ...] = ("long", "short", "beijing", "covid")
    # Median pairwise distance of the training outcome, not of the
    # observed-versus-counterfactual pair. Pairing the counterfactual into
    # the bandwidth makes large shifts look smaller.
    sigma: float | None = None


@dataclass
class Estimate:
    delta: float
    delta_base: float
    ratio: float
    ci: tuple[float, float]
    support_pass_rate: float
    support_adjusted: int
    y_obs: np.ndarray
    y_cf: np.ndarray
    y_base: np.ndarray
    gates: list[list[float]] = field(default_factory=list)
    path: list[float] = field(default_factory=list)
    path_ratio: list[float] = field(default_factory=list)
    localizations: list[dict] = field(default_factory=list)
    eval_t: int = -1
    t0: int = -1


def split_indices(panel: Panel, seed: int):
    members = np.where(panel.member_of == panel.target)[0]
    times = np.arange(16, panel.train_end)
    if panel.split_mode == "time":
        return members, members, times
    rng = np.random.default_rng(seed)
    perm = rng.permutation(members)
    mid = max(len(perm) // 2, 1)
    return perm[:mid], perm[mid:], times


def _entity_x(panel: Panel, t: int, entity: int) -> np.ndarray:
    if panel.entity_covariates is None:
        d = panel.covariates.shape[-1]
        return np.zeros(d)
    t = min(max(t, 0), len(panel.entity_covariates) - 1)
    return panel.entity_covariates[t, entity]


def condition_at(panel: Panel, history: np.ndarray, t: int, entity: int) -> np.ndarray:
    """history: (T, E, d). Pools use only times strictly before t."""
    d = history.shape[-1]
    prev = history[t - 1, entity] if t >= 1 else np.zeros(d)

    def pool(h: int) -> np.ndarray:
        lo = max(0, t - h)
        sl = history[lo:t, entity]
        if len(sl) == 0:
            return np.zeros(d)
        return sl.mean(axis=0)

    ps, pm, pl = pool(2), pool(7), pool(14)
    x = _entity_x(panel, t, entity)
    driver = int(panel.driver_of[entity])
    if driver >= 0:
        flag = np.array([1.0])
        dprev = history[t - 1, driver] if t >= 1 else np.zeros(d)
        lo = max(0, t - 14)
        dsl = history[lo:t, driver]
        dpool = dsl.mean(axis=0) if len(dsl) else np.zeros(d)
    else:
        flag = np.array([0.0])
        dprev = np.zeros(d)
        dpool = np.zeros(d)
    emb = panel.embedding[entity]
    return np.concatenate([prev, ps, pm, pl, x, flag, dprev, dpool, emb])


def pool_slices(d: int) -> list[tuple[int, int]]:
    return [(d, 2 * d), (2 * d, 3 * d), (3 * d, 4 * d)]


def transition_matrices(panel: Panel):
    xs, cs = [], []
    hist = panel.block
    for e in range(hist.shape[1]):
        for t in range(16, panel.train_end):
            xs.append(hist[t, e])
            cs.append(condition_at(panel, hist, t, e))
    return np.asarray(xs, dtype=np.float64), np.asarray(cs, dtype=np.float64)


def design_matrix(panel: Panel, history: np.ndarray, times, members, source: int):
    rows, ys, used_t, used_m = [], [], [], []
    d_src = []
    for t in times:
        src = [history[t - k, source, 0] if t - k >= 0 else 0.0 for k in LAGS]
        med = [history[t - k, panel.mediator, 0] if t - k >= 0 else 0.0 for k in LAGS]
        own = history[t - 1, panel.target, 0] if t >= 1 else 0.0
        base = np.asarray(src + med + [own], dtype=np.float64)
        for m in members:
            cov = panel.covariates[t, m]
            if not np.isfinite(cov).all() or not np.isfinite(panel.outcome[t, m]):
                continue
            rows.append(np.concatenate([base, cov]))
            ys.append(panel.outcome[t, m])
            used_t.append(t)
            used_m.append(m)
            d_src.append(base)
    if not rows:
        return np.zeros((0, 1)), np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(used_t, dtype=int),
        np.asarray(used_m, dtype=int),
    )


def fit_outcome(panel: Panel, train_times, train_members, source: int, cfg: Config) -> QuantileOutcome:
    x, y, _, _ = design_matrix(panel, panel.block, train_times, train_members, source)
    model = QuantileOutcome(max_iter=cfg.gbr_iter, seed=cfg.seed)
    model.fit(x, y)
    return model


def _cloud(panel: Panel, entity: int) -> np.ndarray:
    idx = panel.reference_index[entity]
    cloud = panel.block[idx, entity]
    finite = cloud[np.isfinite(cloud).all(axis=1)]
    if len(finite) < 5:
        finite = panel.block[: panel.train_end, entity]
    return finite


def reference_dim0(panel: Panel, entity: int, mode: str, rng: np.random.Generator) -> float:
    cloud = _cloud(panel, entity)
    if mode == "zero":
        return 0.0
    if mode == "reference_sample":
        return float(cloud[int(rng.integers(len(cloud))), 0])
    return float(np.mean(cloud[:, 0]))


def _support_threshold(cloud: np.ndarray) -> float:
    rng = np.random.default_rng(0)
    if len(cloud) < 8:
        return float("inf")
    take = cloud if len(cloud) <= 80 else cloud[rng.choice(len(cloud), 80, replace=False)]
    dists = []
    for i in range(len(take)):
        diff = take[i] - cloud
        # Exclude a near-copy of itself by dropping the closest single neighbor
        # only when the cloud contains the point. Using the 2nd neighbor is enough.
        dist = np.linalg.norm(diff, axis=1)
        dist.sort()
        dists.append(dist[1] if len(dist) > 1 else dist[0])
    return float(np.quantile(dists, 0.95))


def _localize(point: np.ndarray, cloud: np.ndarray) -> dict:
    mu = cloud.mean(axis=0)
    sd = np.clip(cloud.std(axis=0), 1e-6, None)
    z = (point - mu) / sd
    order = np.argsort(-np.abs(z))
    return {"z": z.round(3).tolist(), "top_dims": order[:3].tolist()}


def project_support(
    point: np.ndarray,
    cloud: np.ndarray,
    threshold: float,
    fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    dist = np.linalg.norm(cloud - point, axis=1)
    nearest = cloud[int(np.argmin(dist))]
    nn = float(dist.min())
    ratio = nn / max(threshold, 1e-6)
    info = {"nn": nn, "ratio": ratio, "action": "keep", "localization": None}
    if ratio <= 1.0:
        return point, info
    info["localization"] = _localize(point, cloud)
    if ratio <= 1.5:
        info["action"] = "reweight"
        return 0.7 * point + 0.3 * nearest, info
    if ratio <= 2.5:
        info["action"] = "clamp"
        lo = np.quantile(cloud, 0.01, axis=0)
        hi = np.quantile(cloud, 0.99, axis=0)
        return np.clip(point, lo, hi), info
    # A residual step outside the training support is dropped: the factual
    # state is kept, so the step does not enter the attribution.
    info["action"] = "fallback"
    if fallback is not None:
        return np.asarray(fallback, dtype=float).copy(), info
    return nearest.copy(), info


def rollout(
    panel: Panel,
    model,
    source: int,
    t0: int,
    n_steps: int,
    level: float,
    roll_mediator: bool,
) -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
    """Residual rollout. Future shocks stay in the factual path.

    Only dimension 0 is replaced at the intervention time. Later states move
    by the difference between the conditional mean under the counterfactual
    history and the conditional mean under the factual history, using the
    same noise seed. Out-of-support states are reweighted, clamped, or
    replaced by the nearest reference state and are flagged.
    """
    hist = panel.block.copy()
    hist[t0, source] = panel.block[t0, source].copy()
    hist[t0, source, 0] = level
    gates: list[np.ndarray] = []
    notes: list[dict] = []
    entities = [source]
    if roll_mediator and int(panel.driver_of[panel.mediator]) == source and panel.mediator != source:
        entities.append(panel.mediator)
    # Support is the training range, including anomalies. The calm reference
    # set is only the intervention baseline; using it as support deletes
    # impulses the rollout was not asked to remove.
    clouds = {e: panel.block[: panel.train_end, e] for e in entities}
    thresholds = {e: _support_threshold(clouds[e]) for e in entities}
    for step in range(n_steps):
        t_next = t0 + step + 1
        if t_next >= len(hist):
            break
        for e in entities:
            # The intervened state itself is the first path point and is not
            # a model draw. Support is checked on generated steps.
            c_cf = condition_at(panel, hist, t_next, e)[None, :]
            c_f = condition_at(panel, panel.block, t_next, e)[None, :]
            seed = 10_000 + 97 * int(e) + 13 * int(t_next)
            mu_cf = model.mean(c_cf, seed)[0]
            gate_cf = np.asarray(getattr(model, "last_gate", np.full(3, np.nan)), dtype=float).copy()
            mu_f = model.mean(c_f, seed)[0]
            if e == source:
                gates.append(gate_cf)
            factual_state = panel.block[t_next, e]
            proposed = factual_state + (mu_cf - mu_f)
            adjusted, info = project_support(proposed, clouds[e], thresholds[e], fallback=factual_state)
            info["t"] = int(t_next)
            info["entity"] = int(e)
            notes.append(info)
            hist[t_next, e] = adjusted
    return hist, gates, notes


def _ar1_phi(panel: Panel, entity: int) -> float:
    series = panel.block[: panel.train_end, entity, 0]
    series = series[np.isfinite(series)]
    denom = float(np.dot(series[:-1], series[:-1])) if len(series) > 5 else 0.0
    if denom < 1e-8:
        return 0.0
    return float(np.clip(np.dot(series[:-1], series[1:]) / denom, -0.95, 0.95))


def _cross_section(outcome: QuantileOutcome, x: np.ndarray, rng: np.random.Generator, n_draws: int = 8, draws: bool = True):
    point = outcome.median(x)
    samples = outcome.sample(x, rng, n_draws=n_draws) if draws else None
    return point, samples


def estimate_at(
    panel: Panel,
    model,
    outcome: QuantileOutcome,
    source: int,
    t0: int,
    lag: int,
    level: float,
    eval_members: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    roll_mediator: bool = True,
    with_path: bool = False,
    with_boot: bool = False,
) -> Estimate | None:
    eval_t = t0 + lag
    if t0 < 16 or eval_t >= len(panel.block) or eval_t <= t0:
        return None
    n_steps = max(lag - 1, 0)
    if with_path:
        n_steps = max(n_steps, 14)
    hist, gates, notes = rollout(panel, model, source, t0, n_steps, level, roll_mediator)
    x_cf, _, _, _ = design_matrix(panel, hist, [eval_t], eval_members, source)
    x_f, y_obs, _, _ = design_matrix(panel, panel.block, [eval_t], eval_members, source)
    if len(y_obs) < 4:
        return None
    y_cf, _ = _cross_section(outcome, x_cf, rng, draws=False)
    y_base, _ = _cross_section(outcome, x_f, rng, draws=False)
    sigma = cfg.sigma
    delta = mmd2_rff(y_obs[:, None], y_cf[:, None], sigma=sigma, n_freq=cfg.rff, seed=cfg.seed)
    delta_base = mmd2_rff(y_obs[:, None], y_base[:, None], sigma=sigma, n_freq=cfg.rff, seed=cfg.seed)
    floor = split_noise_floor(y_obs[:, None], seed=cfg.seed, n_freq=min(64, cfg.rff), sigma=sigma)
    ratio = float(delta / max(delta_base, floor, 1e-8))
    kept = [n for n in notes if n["action"] == "keep"]
    pass_rate = float(len(kept) / max(len(notes), 1))
    ci = (float("nan"), float("nan"))
    if with_boot and len(y_obs) >= 4:
        # Member resample of the conditional-median cross-section. Same estimand as the point.
        boots = []
        for _ in range(cfg.n_boot):
            take = rng.integers(0, len(y_obs), size=len(y_obs))
            boots.append(mmd2_rff(
                y_obs[take, None], y_cf[take, None], sigma=cfg.sigma, n_freq=cfg.rff, seed=cfg.seed,
            ))
        ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    path, path_ratio = [], []
    if with_path:
        for h in range(1, min(16, len(panel.block) - t0)):
            xh, _, _, _ = design_matrix(panel, hist, [t0 + h], eval_members, source)
            xf, yo, _, _ = design_matrix(panel, panel.block, [t0 + h], eval_members, source)
            if len(yo) < 4:
                continue
            yc, _ = _cross_section(outcome, xh, rng, draws=False)
            yb, _ = _cross_section(outcome, xf, rng, draws=False)
            dh = mmd2_rff(yo[:, None], yc[:, None], sigma=cfg.sigma, n_freq=cfg.rff, seed=cfg.seed)
            db = mmd2_rff(yo[:, None], yb[:, None], sigma=cfg.sigma, n_freq=cfg.rff, seed=cfg.seed)
            fl = split_noise_floor(yo[:, None], seed=cfg.seed, sigma=cfg.sigma)
            path.append(float(dh))
            path_ratio.append(float(dh / max(db, fl, 1e-8)))
    locs = [n["localization"] for n in notes if n["localization"] is not None]
    return Estimate(
        delta=float(delta),
        delta_base=float(delta_base),
        ratio=ratio,
        ci=ci,
        support_pass_rate=pass_rate,
        support_adjusted=int(sum(n["action"] != "keep" for n in notes)),
        y_obs=y_obs,
        y_cf=y_cf,
        y_base=y_base,
        gates=[g.tolist() for g in gates],
        path=path,
        path_ratio=path_ratio,
        localizations=locs,
        eval_t=int(eval_t),
        t0=int(t0),
    )


def _score_events(panel, model, outcomes, events, eval_members, cfg, rng, lag=None):
    scores = []
    truths = []
    usable = []
    for event in events:
        source = int(event["entity"])
        level = reference_dim0(panel, source, "reference_mean", rng)
        est = estimate_at(
            panel, model, outcomes[source], source, int(event["t"]),
            lag or panel.peak_lag, level, eval_members, cfg, rng,
            roll_mediator=(int(panel.driver_of[panel.mediator]) == source),
        )
        if est is None:
            continue
        scores.append(est.delta)
        if "overlap" in event:
            truths.append(float(event["overlap"]))
        else:
            truths.append(float(event["dose"]))
        usable.append(event)
    return np.asarray(scores), np.asarray(truths), usable


def mvp_gate(panel, outcomes, eval_members, cfg, rng) -> dict:
    """Quantile-regression counterfactual, no generative rollout.

    The lagged source value at the impulse is replaced by the reference
    level and every other feature stays factual. This is the signal check
    that decides whether a diffusion model is worth training.
    """
    events = panel.queries.get("dose") or []
    scores, truths = [], []
    main_ratio = float("nan")
    main_delta = float("nan")
    for event in events:
        source = int(event["entity"])
        t0 = int(event["t"])
        lag = panel.peak_lag
        if t0 + lag >= len(panel.block):
            continue
        level = reference_dim0(panel, source, "reference_mean", rng)
        # One-day replacement leaves the persistent echo in later lags, so the
        # MVP carries the intervention with an AR(1) fit on the training window.
        # The diffusion model replaces this propagation in stage 2.
        hist = panel.block.copy()
        delta0 = level - hist[t0, source, 0]
        phi = _ar1_phi(panel, source)
        hist[t0, source, 0] = level
        for step in range(1, lag):
            if t0 + step >= len(hist):
                break
            hist[t0 + step, source, 0] = panel.block[t0 + step, source, 0] + (phi ** step) * delta0
        x_cf, _, _, _ = design_matrix(panel, hist, [t0 + lag], eval_members, source)
        x_f, y_obs, _, _ = design_matrix(panel, panel.block, [t0 + lag], eval_members, source)
        if len(y_obs) < 4 or source not in outcomes:
            continue
        y_cf = outcomes[source].median(x_cf)
        y_base = outcomes[source].median(x_f)
        delta = mmd2_rff(y_obs[:, None], y_cf[:, None], sigma=cfg.sigma, n_freq=cfg.rff, seed=cfg.seed)
        base = mmd2_rff(y_obs[:, None], y_base[:, None], sigma=cfg.sigma, n_freq=cfg.rff, seed=cfg.seed)
        floor = split_noise_floor(y_obs[:, None], seed=cfg.seed, sigma=cfg.sigma)
        ratio = float(delta / max(base, floor, 1e-8))
        scores.append(delta)
        truths.append(float(event["dose"]))
        if panel.queries.get("main") and event["t"] == panel.queries["main"]["t"]:
            main_ratio = ratio
            main_delta = float(delta)
    scores_a = np.asarray(scores, dtype=float)
    truths_a = np.asarray(truths, dtype=float)
    conc = pairwise_concordance(scores_a, truths_a) if len(scores_a) >= 4 else float("nan")
    rho = spearman(scores_a, truths_a) if len(scores_a) >= 4 else float("nan")
    if len(scores_a) and not np.isfinite(main_ratio):
        main_ratio = float(scores_a[-1] / max(np.median(scores_a), 1e-8))
        main_delta = float(scores_a[-1])
    near_noise = (not np.isfinite(main_ratio)) or main_ratio < T.mvp_noise_ratio
    low_rank = (not np.isfinite(conc)) or conc < T.mvp_stop_concordance
    if len(scores_a) < 4 and np.isfinite(main_ratio):
        # Not enough contrasts to rank. The level of the main contrast decides.
        decision = "stop" if near_noise else "proceed_unranked"
    elif low_rank and near_noise:
        decision = "stop"
    elif np.isfinite(conc) and conc > T.mvp_go_concordance:
        decision = "proceed"
    elif not near_noise:
        decision = "proceed_unstable_ranking"
    else:
        decision = "stop"
    return {
        "concordance": None if not np.isfinite(conc) else float(conc),
        "spearman": None if not np.isfinite(rho) else float(rho),
        "main_delta": None if not np.isfinite(main_delta) else float(main_delta),
        "main_ratio": None if not np.isfinite(main_ratio) else float(main_ratio),
        "n_events": int(len(scores_a)),
        "decision": decision,
    }


def _ranking_construction(name: str, scores, truths, grounded: bool, mismatch: set[str]) -> dict:
    if len(scores) < T.l1_min_events:
        return {"name": name, "status": "insufficient_contrast", "n": int(len(scores))}
    conc = pairwise_concordance(scores, truths)
    rho = spearman(scores, truths)
    ok = np.isfinite(conc) and np.isfinite(rho) and conc > T.l1_concordance and rho > T.l1_spearman
    if ok:
        status = "pass"
    elif name in mismatch:
        status = "assumption_mismatch"
    else:
        status = "fail"
    return {
        "name": name,
        "status": status,
        "concordance": None if not np.isfinite(conc) else float(conc),
        "spearman": None if not np.isfinite(rho) else float(rho),
        "n": int(len(scores)),
        "grounded": grounded,
    }


def _temporal_construction(lags, deltas, mismatch: set[str]) -> dict:
    rho = spearman(lags, deltas)
    increases = int(np.sum(np.diff(deltas) > 0)) if len(deltas) >= 2 else 0
    smooth = increases <= max(1, len(deltas) // 5)
    ok = np.isfinite(rho) and rho < T.l1_lag_spearman and smooth
    if ok:
        status = "pass"
    elif "temporal_distance" in mismatch:
        status = "assumption_mismatch"
    else:
        status = "fail"
    return {
        "name": "temporal_distance",
        "status": status,
        "spearman": None if not np.isfinite(rho) else float(rho),
        "smooth": bool(smooth),
        "increases": increases,
    }


def _isolation_construction(iso, conn, mismatch: set[str]) -> dict:
    if len(iso) < 3 or len(conn) < 3:
        return {"name": "exogenous_isolation", "status": "insufficient_contrast", "n_iso": int(len(iso)), "n_conn": int(len(conn))}
    effect = cohens_d(iso, conn)
    p = welch_p(iso, conn)
    ok = np.isfinite(effect) and np.isfinite(p) and effect < T.l1_isolation_effect and p < T.l1_isolation_p
    if ok:
        status = "pass"
    elif "exogenous_isolation" in mismatch:
        status = "assumption_mismatch"
    else:
        status = "fail"
    return {
        "name": "exogenous_isolation",
        "status": status,
        "effect": None if not np.isfinite(effect) else float(effect),
        "p": None if not np.isfinite(p) else float(p),
        "mean_isolated": float(np.mean(iso)),
        "mean_connected": float(np.mean(conn)),
    }


def l1_overall(constructions: list[dict], grounded: bool) -> dict:
    passes = [c for c in constructions if c["status"] == "pass"]
    fails = [c for c in constructions if c["status"] == "fail"]
    # On an observational panel the constructions are assumption checks.
    # They are reported, and they do not confirm the method.
    if not grounded:
        return {
            "status": "assumption_check",
            "n_pass": len(passes),
            "n_fail": len(fails),
            "confirmed": False,
            "constructions": constructions,
        }
    confirmed = len(passes) >= T.l1_min_passes
    return {
        "status": "pass" if confirmed else "fail",
        "n_pass": len(passes),
        "n_fail": len(fails),
        "confirmed": confirmed,
        "constructions": constructions,
    }


def gate_matches_lag(gates: list[list[float]], peak_lag: int) -> dict:
    g = np.asarray(gates, dtype=float) if gates else np.zeros((0, 3))
    if g.ndim != 2 or g.shape[0] < 4 or g.shape[1] < 3:
        return {"pass": False, "reason": "trajectory_too_short", "expect": "coarse_to_fine" if peak_lag >= 8 else "fine_dominant"}
    short, long = g[:, 0], g[:, 2]
    if peak_lag >= 8:
        mid = len(g) // 2
        ok = float(long[:mid].mean()) > float(long[mid:].mean()) and float(short[mid:].mean()) > float(short[:mid].mean())
        expect = "coarse_to_fine"
    else:
        ok = float(short.mean()) > float(long.mean())
        expect = "fine_dominant"
    return {
        "pass": bool(ok),
        "expect": expect,
        "short_mean": float(short.mean()),
        "long_mean": float(long.mean()),
        "short_early": float(short[: len(short) // 2].mean()),
        "short_late": float(short[len(short) // 2 :].mean()),
        "long_early": float(long[: len(long) // 2].mean()),
        "long_late": float(long[len(long) // 2 :].mean()),
    }


def precision_ok(deltas: list[float], true_lag: int) -> dict:
    d = np.asarray(deltas, dtype=float)
    lags = np.arange(1, len(d) + 1)
    if true_lag < 1 or true_lag > len(d) or not np.isfinite(d).any():
        return {"pass": False, "reason": "missing"}
    i = true_lag - 1
    finite = d[np.isfinite(d)]
    height_ok = bool(d[i] >= np.max(finite) / RFF_PEAK_SLACK)
    local = True
    if i > 0 and np.isfinite(d[i - 1]):
        local = local and d[i] >= d[i - 1]
    if i < len(d) - 1 and np.isfinite(d[i + 1]):
        local = local and d[i] >= d[i + 1]
    return {
        "pass": bool(height_ok and local),
        "argmax_lag": int(lags[int(np.nanargmax(d))]),
        "true_lag": int(true_lag),
        "height_ok": height_ok,
        "local_peak": bool(local),
        "deltas": [None if not np.isfinite(v) else float(v) for v in d],
    }


def _significant(ratio: float) -> bool:
    return bool(np.isfinite(ratio) and ratio > T.l2_ratio)


def layer2(panel, model, outcomes, eval_members, cfg, rng, main_event) -> dict:
    source = int(main_event["entity"])
    impulse_t = int(main_event["t"])
    eval_day = impulse_t + panel.peak_lag
    level = reference_dim0(panel, source, "reference_mean", rng)
    scan = []
    ratios = {}
    for h in range(1, 16):
        t0 = eval_day - h
        est = estimate_at(
            panel, model, outcomes[source], source, t0, h, level,
            eval_members, cfg, rng,
            roll_mediator=(int(panel.driver_of[panel.mediator]) == source),
        )
        if est is None:
            scan.append(float("nan"))
            continue
        scan.append(est.ratio)
        ratios[h] = est.ratio
    # A longer gate trajectory from the true intervention, independent of
    # the one-step outcome lag. Ten generated steps when the series allows.
    gate_est = estimate_at(
        panel, model, outcomes[source], source, impulse_t, max(panel.peak_lag, 10),
        level, eval_members, cfg, rng, with_path=True,
        roll_mediator=(int(panel.driver_of[panel.mediator]) == source),
    )
    gates = gate_est.gates if gate_est is not None else []
    ginfo = gate_matches_lag(gates, panel.peak_lag)
    prec = precision_ok(scan, panel.peak_lag)
    placebo_lags = [h for h in (20, 15, 5, 3) if h != panel.peak_lag]
    placebo_ratios = {}
    for h in placebo_lags:
        t0 = eval_day - h
        est = estimate_at(
            panel, model, outcomes[source], source, t0, h, level, eval_members, cfg, rng,
            roll_mediator=(int(panel.driver_of[panel.mediator]) == source),
        )
        placebo_ratios[h] = None if est is None else est.ratio
    peak_ratio = ratios.get(panel.peak_lag)
    evaluated = [h for h, value in placebo_ratios.items() if value is not None]
    others_quiet = len(evaluated) >= 3 and all(not _significant(placebo_ratios[h]) for h in evaluated)
    placebo_pass = _significant(peak_ratio if peak_ratio is not None else float("nan")) and others_quiet
    passed = bool(placebo_pass and prec["pass"] and ginfo["pass"])
    return {
        "status": "pass" if passed else "fail",
        "placebo_pass": bool(placebo_pass),
        "placebo_ratios": {str(k): v for k, v in placebo_ratios.items()},
        "peak_ratio": peak_ratio,
        "precision": prec,
        "gate": ginfo,
        "scan_ratios": [None if not np.isfinite(v) else float(v) for v in scan],
    }


def layer3(model: ConditionalDDPM, panel: Panel, cfg: Config) -> dict:
    rng = np.random.default_rng(cfg.seed + 11)
    pairs = []
    for e in range(panel.block.shape[1]):
        for t in range(max(panel.train_end, 16), len(panel.block)):
            if np.isfinite(panel.block[t, e]).all():
                pairs.append((e, t))
    if len(pairs) > 400:
        take = rng.choice(len(pairs), 400, replace=False)
        pairs = [pairs[i] for i in take]
    if len(pairs) < 20:
        return {"status": "fail", "reason": "too_few_validation_points", "n": len(pairs)}
    cond = np.asarray([condition_at(panel, panel.block, t, e) for e, t in pairs])
    obs = np.asarray([panel.block[t, e] for e, t in pairs])
    # One stochastic sample for Wasserstein and HSIC, and a batch of
    # conditional samples for the randomized PIT.
    gen = model.ancestral(cond, np.random.default_rng(cfg.seed + 3), eta=1.0)[0]
    n_pit = 24
    cond_rep = np.repeat(cond, n_pit, axis=0)
    pit_samples = model.ancestral(cond_rep, np.random.default_rng(cfg.seed + 4), eta=1.0)[0]
    pit_samples = pit_samples.reshape(len(pairs), n_pit, panel.block.shape[-1])
    pvals = []
    for d in range(obs.shape[1]):
        pits = []
        for i in range(len(pairs)):
            pits.append(randomized_pit(obs[i, d : d + 1], pit_samples[i, :, d : d + 1], rng))
        pvals.append(ks_uniform_p(np.asarray(pits)))
    w = wasserstein_nd(obs, gen)
    scale = float(np.std(obs))
    w_ok = bool(np.isfinite(w) and w < T.l3_wasserstein_std * max(scale, 1e-6))
    ratio = hsic_ratio(cond, obs, gen, seed=cfg.seed)
    hsic_ok = bool(np.isfinite(ratio) and T.l3_hsic_low <= ratio <= T.l3_hsic_high)
    pit_ok = bool(pvals) and all(np.isfinite(p) and p > T.l3_ks_p for p in pvals)
    passed = pit_ok and w_ok and hsic_ok
    return {
        "status": "pass" if passed else "fail",
        "pit_p": [None if not np.isfinite(p) else float(p) for p in pvals],
        "pit_pass": pit_ok,
        "wasserstein": float(w),
        "wasserstein_limit": T.l3_wasserstein_std * max(scale, 1e-6),
        "wasserstein_pass": w_ok,
        "hsic_ratio": None if not np.isfinite(ratio) else float(ratio),
        "hsic_pass": hsic_ok,
        "n": len(pairs),
        "caveat": "Observational calibration is necessary and not sufficient for counterfactual accuracy.",
    }


def _dr_ate(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    w = w.astype(int)
    if len(y) < 30 or len(np.unique(w)) < 2:
        return float("nan"), float("nan")
    if (w == 1).sum() < 8 or (w == 0).sum() < 8:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    mid = len(y) // 2
    psi = np.zeros(len(y))
    for train, test in ((perm[:mid], perm[mid:]), (perm[mid:], perm[:mid])):
        if len(np.unique(w[train])) < 2:
            return float("nan"), float("nan")
        clf = LogisticRegression(max_iter=400)
        clf.fit(x[train], w[train])
        prob = np.clip(clf.predict_proba(x[test])[:, 1], 0.05, 0.95)
        mu = np.zeros(len(test))
        mu1 = np.zeros(len(test))
        mu0 = np.zeros(len(test))
        for label, store in ((1, mu1), (0, mu0)):
            sel = w[train] == label
            if sel.sum() < 4:
                store[:] = y[train][sel].mean() if sel.sum() else y[train].mean()
                continue
            reg = Ridge(alpha=1.0)
            reg.fit(x[train][sel], y[train][sel])
            store[:] = reg.predict(x[test])
        ww = w[test]
        yy = y[test]
        psi[test] = (mu1 - mu0) + ww / prob * (yy - mu1) - (1 - ww) / (1 - prob) * (yy - mu0)
    ate = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(len(psi)))
    if se < 1e-8:
        p = 0.0 if abs(ate) > 0 else 1.0
    else:
        from scipy.stats import norm
        p = float(2 * norm.sf(abs(ate) / se))
    return ate, p


def layer4(panel, main: Estimate, mask_delta: float, uplift_ate: float, uplift_p: float, gate_pass: bool) -> dict:
    rollout_sig = _significant(main.ratio)
    effect = float(np.mean(main.y_obs) - np.mean(main.y_cf))
    uplift_sig = bool(np.isfinite(uplift_p) and uplift_p < 0.05 and np.isfinite(uplift_ate))
    if not np.isfinite(uplift_ate):
        uplift = {"name": "uplift_cate", "agree": False, "status": "not_evaluable"}
    elif uplift_sig and rollout_sig:
        uplift = {"name": "uplift_cate", "agree": bool(np.sign(uplift_ate) == np.sign(effect) or abs(effect) < 1e-8), "ate": float(uplift_ate), "p": float(uplift_p), "outcome_shift": effect}
    elif (not uplift_sig) and (not rollout_sig):
        uplift = {"name": "uplift_cate", "agree": True, "ate": float(uplift_ate), "p": float(uplift_p), "outcome_shift": effect, "note": "both null"}
    else:
        uplift = {"name": "uplift_cate", "agree": False, "ate": float(uplift_ate), "p": None if not np.isfinite(uplift_p) else float(uplift_p), "outcome_shift": effect}
    if not np.isfinite(mask_delta):
        mask = {"name": "mask_gap", "agree": False, "status": "not_evaluable"}
    else:
        gap = float(main.delta - mask_delta)
        mask = {"name": "mask_gap", "agree": bool(gap > 0), "gap": gap, "delta_mask": float(mask_delta)}
    # Business-window alignment. z is the eval-day cohort mean against
    # the training-window daily means of the same members.
    business_z = float("nan")
    # Filled by the caller through main; recomputed here if eval index is known.
    proxies = [uplift, mask, {"name": "business_window", "agree": False}, {"name": "gate", "agree": bool(gate_pass)}]
    return {"proxies_partial": proxies, "rollout_significant": rollout_sig, "effect_shift": effect}


def _placebo_time(panel: Panel, eval_t: int, eval_members: np.ndarray) -> int:
    """A non-event day whose cohort mean is inside the training bulk."""
    daily = panel.outcome[:, eval_members].mean(axis=1)
    mu = float(np.nanmean(daily[: panel.train_end]))
    sd = float(np.nanstd(daily[: panel.train_end]))
    sd = sd if sd > 1e-6 else 1.0
    z = (daily - mu) / sd
    occupied: set[int] = set()
    for key in ("dose", "overlap", "isolated", "connected"):
        for event in panel.queries.get(key, []):
            start = int(event["t"])
            for dt in range(0, panel.peak_lag + 15):
                occupied.add(start + dt)
    candidates = [
        t for t in range(16, len(daily))
        if t not in occupied and abs(t - eval_t) >= 15 and np.isfinite(z[t])
    ]
    if not candidates:
        return max(16, min(len(daily) - 1, eval_t - 20))
    calm = [t for t in candidates if abs(z[t]) < 1.0]
    pool = calm or candidates
    return min(pool, key=lambda t: (abs(float(z[t])), abs(t - eval_t)))


def business_alignment(panel: Panel, main: Estimate, eval_members: np.ndarray) -> dict:
    train_daily = panel.outcome[: panel.train_end][:, eval_members].mean(axis=1)
    mu, sd = float(np.mean(train_daily)), float(np.std(train_daily))
    sd = sd if sd > 1e-6 else 1.0
    z_now = float((panel.outcome[main.eval_t, eval_members].mean() - mu) / sd)
    placebo_t = _placebo_time(panel, main.eval_t, eval_members)
    z_p = float((panel.outcome[placebo_t, eval_members].mean() - mu) / sd)
    anomalous = z_now > 1.5 and abs(z_p) < 1.0
    agree = anomalous == _significant(main.ratio)
    return {"name": "business_window", "agree": bool(agree), "z_eval": z_now, "z_placebo": z_p, "anomalous": bool(anomalous)}


def layer5(main: Estimate, baseline_deltas: dict[str, float], kernel_deltas: list[float], arch_deltas: dict[str, float], evalue: float) -> dict:
    cv_b = coefficient_of_variation(list(baseline_deltas.values()))
    cv_k = coefficient_of_variation(kernel_deltas)
    cv_a = coefficient_of_variation(list(arch_deltas.values()))
    b_ok = bool(np.isfinite(cv_b) and cv_b < T.l5_cv_baseline)
    k_ok = bool(np.isfinite(cv_k) and cv_k < T.l5_cv_kernel)
    a_ok = bool(np.isfinite(cv_a) and cv_a < T.l5_cv_architecture)
    e_ok = bool(np.isfinite(evalue) and evalue > T.l5_evalue)
    fragile = bool(np.isfinite(evalue) and evalue < T.l5_evalue_fragile)
    return {
        "status": "pass" if (b_ok and k_ok and a_ok and e_ok) else "fail",
        "baseline_cv": cv_b,
        "baseline_pass": b_ok,
        "baseline_deltas": baseline_deltas,
        "kernel_cv": cv_k,
        "kernel_pass": k_ok,
        "kernel_deltas": kernel_deltas,
        "architecture_cv": cv_a,
        "architecture_pass": a_ok,
        "architecture_deltas": arch_deltas,
        "evalue": None if not np.isfinite(evalue) else float(evalue),
        "evalue_pass": e_ok,
        "fragile": fragile,
    }


def path_shape(path_ratio: list[float]) -> str:
    p = np.asarray(path_ratio, dtype=float)
    p = p[np.isfinite(p)]
    if len(p) < 3 or np.max(p) <= 0:
        return "平坦"
    peak_at = int(np.argmax(p))
    early = float(np.mean(p[: max(len(p) // 3, 1)]))
    if peak_at >= int(len(p) * 0.66) and early < 0.45 * float(np.max(p)):
        return "后期跳变"
    if 0 < peak_at < len(p) - 1 and p[-1] < 0.7 * float(np.max(p)):
        return "峰值后下降"
    if np.all(np.diff(p) >= -0.05 * float(np.max(p))):
        return "单调上升"
    return "不规则"


def credibility(l1, l2, l3, l4_pass, l5_pass, grounded: bool, evalue: float) -> dict:
    flags = {
        "L1": bool(l1.get("confirmed")),
        "L2": l2.get("status") == "pass",
        "L3": l3.get("status") == "pass",
        "L4": bool(l4_pass),
        "L5": bool(l5_pass),
    }
    order = ["L3", "L2", "L1", "L4", "L5"]
    failed = [name for name in order if not flags[name]]
    bottleneck = failed[0] if failed else "none"
    if not flags["L3"]:
        level = "低"
        reason = "观测校准没有通过。生成模型在观测数据上就不可用，反事实分数不能进入决策。"
    elif not flags["L2"]:
        level = "探索性"
        reason = "时间定位没有通过。分数可以描述一个分布差异，但不能把差异钉在声明的那一天。"
    elif flags["L1"] and flags["L4"] and flags["L5"] and grounded and np.isfinite(evalue) and evalue >= 3:
        level = "高"
        reason = "五层都通过，而且 E-value 不低于 3。"
    elif grounded and flags["L1"] and flags["L4"] and sum(flags.values()) >= 4:
        level = "中等偏高"
        reason = "相对排序和代理一致通过。最弱的一项决定这个结论还能承担多大的决策权重。"
    elif sum(flags[k] for k in ("L2", "L3", "L4", "L5")) >= 3:
        level = "中等"
        reason = "多数证伪层通过，但不是每一层都通过。"
    else:
        level = "低"
        reason = "通过的证伪层不够，结论不进入决策。"
    if not grounded and level in {"高", "中等偏高", "中等"}:
        level = "探索性"
        reason += " 这张面板没有可构造的反事实 ground truth，L1 只作为假设检查，不能把可信度抬到决策等级。"
    return {"level": level, "reason": reason, "bottleneck": bottleneck, "flags": flags}


def _uplift_table(panel: Panel, source: int, train_members: np.ndarray) -> tuple[float, float]:
    rows_x, rows_y, rows_w = [], [], []
    series = panel.block[:, source, 0]
    mu = np.nanmean(series[: panel.train_end])
    sd = np.nanstd(series[: panel.train_end])
    sd = sd if sd > 1e-6 else 1.0
    for t in range(panel.peak_lag + 2, panel.train_end):
        z = (series[t - panel.peak_lag] - mu) / sd
        w = 1 if z > 1.0 else 0
        y = float(np.nanmean(panel.outcome[t, train_members]))
        x = _entity_x(panel, t, panel.target)
        if np.isfinite(y) and np.isfinite(x).all():
            rows_x.append(x)
            rows_y.append(y)
            rows_w.append(w)
    if len(rows_y) < 30:
        return float("nan"), float("nan")
    return _dr_ate(np.asarray(rows_x), np.asarray(rows_y), np.asarray(rows_w))


def _exposure_evalue(panel: Panel, source: int, train_members: np.ndarray) -> float:
    series = panel.block[:, source, 0]
    mu = np.nanmean(series[: panel.train_end])
    sd = np.nanstd(series[: panel.train_end])
    sd = sd if sd > 1e-6 else 1.0
    exposed, calm = [], []
    for t in range(panel.peak_lag + 2, panel.train_end):
        z = (series[t - panel.peak_lag] - mu) / sd
        y = float(np.nanmean(panel.outcome[t, train_members]))
        if not np.isfinite(y):
            continue
        if z > 1.0:
            exposed.append(y)
        elif abs(z) < 0.5:
            calm.append(y)
    if len(exposed) < 5 or len(calm) < 5:
        return float("nan")
    return evalue_from_smd(cohens_d(exposed, calm))


def _kernel_sensitivity(y_obs, y_cf, cfg: Config) -> list[float]:
    sigma = cfg.sigma if cfg.sigma is not None else 1.0
    out = []
    for scale in (0.5, 1.0, 2.0):
        out.append(mmd2_rff(y_obs[:, None], y_cf[:, None], sigma=sigma * scale, n_freq=cfg.rff, seed=cfg.seed))
    return out


def _oracle_dose(panel: Panel, eval_members, cfg: Config) -> dict | None:
    if not panel.grounded or panel.raw_block is None:
        return None
    scores, doses = [], []
    for event in panel.queries["dose"]:
        entity = int(event["entity"])
        idx = panel.reference_index[entity]
        raw_level = float(np.mean(panel.raw_block[idx, entity, 0]))
        y_cf = oracle_outcome(panel, int(event["t"]), entity, raw_level)
        eval_t = int(event["t"]) + panel.peak_lag
        if eval_t >= len(panel.outcome):
            continue
        yo = panel.outcome[eval_t, eval_members][:, None]
        yc = y_cf[eval_t, eval_members][:, None]
        delta = mmd2_rff(yo, yc, sigma=cfg.sigma, n_freq=cfg.rff, seed=cfg.seed)
        scores.append(delta)
        doses.append(float(event["dose"]))
    if len(scores) < 4:
        return None
    return {
        "spearman": spearman(scores, doses),
        "concordance": pairwise_concordance(scores, doses),
        "deltas": [float(s) for s in scores],
        "doses": doses,
    }


def run_panel(panel: Panel, cfg: Config) -> dict:
    rng = np.random.default_rng(cfg.seed)
    train_members_for_scale = np.where(panel.member_of == panel.target)[0]
    y_scale = panel.outcome[: panel.train_end][:, train_members_for_scale]
    y_scale = y_scale[np.isfinite(y_scale)]
    cfg.sigma = median_bandwidth(y_scale.reshape(-1, 1))
    print(
        f"\n=== {panel.name}  grounded={panel.grounded}  T={len(panel.block)}  "
        f"E={panel.block.shape[1]}  sigma={cfg.sigma:.3f} ===",
        flush=True,
    )
    panel.embedding = rng.normal(0.0, 0.5, size=(panel.block.shape[1], 4))
    train_members, eval_members, train_times = split_indices(panel, cfg.seed)
    sources = sorted({int(ev["entity"]) for key in ("dose", "overlap", "isolated", "connected") for ev in panel.queries.get(key, [])})
    if panel.queries.get("main"):
        sources.append(int(panel.queries["main"]["entity"]))
    sources = sorted(set(sources))
    outcomes = {s: fit_outcome(panel, train_times, train_members, s, cfg) for s in sources}
    print(f"stage 1  outcome models={len(outcomes)}  train_rows_target={len(train_members)}", flush=True)
    mvp = mvp_gate(panel, outcomes, eval_members, cfg, rng)
    print(f"stage 1  MVP {mvp}", flush=True)
    base = {
        "panel": panel.name,
        "grounded": panel.grounded,
        "notes": panel.notes,
        "peak_lag": panel.peak_lag,
        "semantic": SEMANTIC_DECLARATION,
        "primary_model": PRIMARY_MODEL,
        "mvp": mvp,
        "target": panel.entity_names[panel.target],
        "mediator": panel.entity_names[panel.mediator],
        "source": (
            panel.entity_names[int(panel.queries["main"]["entity"])]
            if panel.queries.get("main") is not None
            else None
        ),
        "n_entities": len(panel.entity_names),
        "n_time": int(len(panel.block)),
        "n_eval_members": int(len(eval_members)),
        "mmd_sigma": cfg.sigma,
    }
    if mvp["decision"] == "stop":
        base.update({
            "stopped": True,
            "delta_cf": None,
            "delta_cf_ci": None,
            "delta_normalized": None,
            "path_curve": [],
            "gate_trajectory": [],
            "support_pass_rate": None,
            "evaluation": {"stopped_at": "stage_1", "reason": "MVP signal check did not clear the pre-registered bar."},
            "credibility": {"level": "停止", "reason": "分位数回归的反事实替换没有给出稳定信号，按协议不训练 DDPM。", "bottleneck": "MVP", "flags": {}},
        })
        return base

    x, cond = transition_matrices(panel)
    slices = pool_slices(panel.block.shape[-1])
    print(f"stage 2  transitions={len(x)} cond_dim={cond.shape[1]}", flush=True)
    ddpm = ConditionalDDPM(
        x.shape[1], cond.shape[1], slices, hidden=cfg.hidden,
        n_steps=cfg.diff_steps, seed=cfg.seed,
    )
    flow = ConditionalRealNVP(x.shape[1], cond.shape[1], hidden=cfg.hidden, seed=cfg.seed + 5)
    ddpm_loss = ddpm.fit(x, cond, cfg.ddpm_epochs, cfg.batch, cfg.seed)
    flow_loss = flow.fit(x, cond, cfg.flow_epochs, cfg.batch, cfg.seed + 9)
    print(f"stage 2  ddpm loss {ddpm_loss[0]:.4f} -> {ddpm_loss[-1]:.4f}  flow {flow_loss[0]:.4f} -> {flow_loss[-1]:.4f}", flush=True)
    print("stage 2  observational calibration", flush=True)
    cal = layer3(ddpm, panel, cfg)
    main_event = panel.queries.get("main")
    if main_event is None:
        base.update({"stopped": True, "evaluation": {"reason": "no main event"}, "delta_cf": None,
                     "delta_cf_ci": None, "delta_normalized": None, "path_curve": [], "gate_trajectory": [],
                     "support_pass_rate": None, "calibration": cal})
        return base
    source = int(main_event["entity"])
    level = reference_dim0(panel, source, "reference_mean", rng)
    print("stage 3  main rollout", flush=True)
    main = estimate_at(
        panel, ddpm, outcomes[source], source, int(main_event["t"]), panel.peak_lag, level,
        eval_members, cfg, rng, with_path=True, with_boot=True,
        roll_mediator=True,
    )
    if main is None:
        base.update({"stopped": True, "delta_cf": None, "delta_cf_ci": None, "delta_normalized": None,
                     "path_curve": [], "gate_trajectory": [], "support_pass_rate": None,
                     "calibration": cal, "evaluation": {"reason": "main query could not be evaluated"}})
        return base

    print("stage 4  L1 rankings", flush=True)
    dose_s, dose_t, _ = _score_events(panel, ddpm, outcomes, panel.queries.get("dose", []), eval_members, cfg, rng)
    ov_s, ov_t, _ = _score_events(panel, ddpm, outcomes, panel.queries.get("overlap", []), eval_members, cfg, rng)
    iso_s, _, _ = _score_events(panel, ddpm, outcomes, panel.queries.get("isolated", []), eval_members, cfg, rng)
    conn_s, _, _ = _score_events(panel, ddpm, outcomes, panel.queries.get("connected", []), eval_members, cfg, rng)
    print("stage 4  L2 temporal scan", flush=True)
    l2 = layer2(panel, ddpm, outcomes, eval_members, cfg, rng, main_event)
    temporal = _temporal_construction(
        list(range(1, len(l2["scan_ratios"]) + 1)),
        [np.nan if v is None else v for v in l2["scan_ratios"]],
        panel.assumption_mismatch,
    )
    constructions = [
        _ranking_construction("dose_response", dose_s, dose_t, panel.grounded, panel.assumption_mismatch),
        temporal,
        _ranking_construction("overlap", ov_s, ov_t, panel.grounded, panel.assumption_mismatch),
        _isolation_construction(iso_s, conn_s, panel.assumption_mismatch),
    ]
    # Observational panels: a numeric pass is still an assumption check.
    if not panel.grounded:
        for c in constructions:
            if c["status"] == "pass":
                c["status"] = "assumption_consistent"
            elif c["status"] == "fail":
                c["status"] = "assumption_rejected"
    l1 = l1_overall(constructions, panel.grounded)

    print("stage 4  L4 proxies and L5 sensitivity", flush=True)
    mask = estimate_at(
        panel, ddpm, outcomes[source], source, int(main_event["t"]), panel.peak_lag, level,
        eval_members, cfg, rng, roll_mediator=False,
    )
    ate, ate_p = _uplift_table(panel, source, train_members)
    partial = layer4(panel, main, mask.delta if mask is not None else float("nan"), ate, ate_p, l2["gate"]["pass"])
    business = business_alignment(panel, main, eval_members)
    proxies = []
    for item in partial["proxies_partial"]:
        if item["name"] == "business_window":
            proxies.append(business)
        else:
            proxies.append(item)
    n_agree = sum(1 for p in proxies if p.get("agree"))
    n_disagree = sum(1 for p in proxies if not p.get("agree"))
    l4_pass = n_agree >= T.l4_min_agree and n_disagree < T.l4_max_disagree
    l4 = {
        "status": "pass" if l4_pass else ("uncertain" if n_disagree >= T.l4_max_disagree else "fail"),
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "proxies": proxies,
    }

    baselines = {}
    for mode in ("zero", "reference_mean", "reference_sample"):
        lv = reference_dim0(panel, source, mode, rng)
        est = estimate_at(
            panel, ddpm, outcomes[source], source, int(main_event["t"]), panel.peak_lag, lv,
            eval_members, cfg, rng, roll_mediator=True,
        )
        baselines[mode] = None if est is None else est.delta
    kernels = _kernel_sensitivity(main.y_obs, main.y_cf, cfg)
    flow_est = estimate_at(
        panel, flow, outcomes[source], source, int(main_event["t"]), panel.peak_lag, level,
        eval_members, cfg, rng, roll_mediator=True,
    )
    arch = {"ddpm": main.delta, "realnvp": None if flow_est is None else flow_est.delta}
    ev = _exposure_evalue(panel, source, train_members)
    l5 = layer5(main, {k: v for k, v in baselines.items() if v is not None}, kernels, {k: v for k, v in arch.items() if v is not None}, ev)
    cred = credibility(l1, l2, cal, l4_pass, l5["status"] == "pass", panel.grounded, ev if np.isfinite(ev) else float("nan"))
    oracle = _oracle_dose(panel, eval_members, cfg)
    shape = path_shape(main.path_ratio)
    print(
        f"done  delta={main.delta:.4f} ratio={main.ratio:.2f} shape={shape} "
        f"L1={l1['status']} L2={l2['status']} L3={cal['status']} L4={l4['status']} L5={l5['status']} "
        f"credibility={cred['level']}",
        flush=True,
    )
    base.update({
        "stopped": False,
        "delta_cf": main.delta,
        "delta_cf_ci": list(main.ci),
        "delta_base": main.delta_base,
        "delta_normalized": main.ratio,
        "path_curve": main.path,
        "path_ratio": main.path_ratio,
        "path_shape": shape,
        "gate_trajectory": main.gates,
        "support_pass_rate": main.support_pass_rate,
        "support_adjusted_steps": main.support_adjusted,
        "localizations": main.localizations[:6],
        "main_event": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v) for k, v in main_event.items() if k in {"t", "entity", "dose", "tag"}},
        "source": panel.entity_names[source],
        "training": {"ddpm_loss": ddpm_loss, "flow_loss": flow_loss},
        "oracle_dose": oracle,
        "evaluation": {"L1": l1, "L2": l2, "L3": cal, "L4": l4, "L5": l5},
        "credibility": cred,
    })
    return base
