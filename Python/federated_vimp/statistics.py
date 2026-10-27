"""Site-level two-group OOD VIMP.

Covariate group: MMD-LOCO, HSIC-LOCO, domain-classifier permutation VIMP.
Concept group: PO-risk LOCO, R-risk LOCO, split importance on the DR score.

These are conditional-association scores. The concept scores track a
change in E[Y | X] across batches; the covariate scores track a change
in the law of X. Neither is a causal effect of the feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor


@dataclass(frozen=True)
class StatConfig:
    mmd_max_n: int = 180
    hsic_max_n: int = 160
    rf_trees: int = 30
    rf_depth: int = 6
    holdout: float = 0.30
    ridge_alpha: float = 1.0
    clip_e: float = 0.10
    tree_depth: int = 3
    tree_leaf: int = 20
    z_transform: str = "asinh"
    calibration_tol: float = 0.80


def simplex(values: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Project a nonnegative score onto the probability simplex.

    Coordinates outside `mask` stay zero. An all-zero score becomes uniform
    on the observed coordinates so a method that found nothing does not
    pretend to have a top feature.
    """
    out = np.maximum(np.asarray(values, dtype=float), 0.0).copy()
    if mask is not None:
        out = np.where(np.asarray(mask, dtype=bool), out, 0.0)
    total = float(out.sum())
    if total < 1e-12:
        out = np.zeros_like(out)
        if mask is None:
            out[:] = 1.0 / out.size
        else:
            observed = np.asarray(mask, dtype=bool)
            count = int(observed.sum())
            if count == 0:
                raise ValueError("feature mask is empty")
            out[observed] = 1.0 / count
        return out
    return out / total


def dominance(vimp_con: np.ndarray, vimp_cov: np.ndarray, eps: float = 1e-8) -> float:
    """Peak-concentration contrast. Positive leans concept, negative covariate."""
    peak_con = float(np.max(vimp_con))
    peak_cov = float(np.max(vimp_cov))
    return (peak_con - peak_cov) / (peak_con + peak_cov + eps)


def _sqdist(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left2 = np.sum(left * left, axis=1)[:, None]
    right2 = np.sum(right * right, axis=1)[None, :]
    return np.maximum(left2 + right2 - 2.0 * left @ right.T, 0.0)


def _subsample_pair(
    x_ref: np.ndarray,
    x_new: np.ndarray,
    max_n: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(x_ref) > max_n:
        x_ref = x_ref[rng.choice(len(x_ref), size=max_n, replace=False)]
    if len(x_new) > max_n:
        x_new = x_new[rng.choice(len(x_new), size=max_n, replace=False)]
    return x_ref, x_new


def _gamma(x_ref: np.ndarray, x_new: np.ndarray) -> float:
    pool = np.vstack([x_ref, x_new])
    take = min(len(pool), 240)
    if len(pool) > take:
        pool = pool[np.linspace(0, len(pool) - 1, take).astype(int)]
    sq = _sqdist(pool, pool)
    positive = sq[sq > 0]
    if positive.size == 0:
        return 1.0
    return 1.0 / float(np.median(positive))


def rbf_mmd2(x_ref: np.ndarray, x_new: np.ndarray, gamma: float) -> float:
    """Biased RBF MMD^2, matching the in-repo FSDS kernel (exp(-gamma ||x-y||^2))."""
    xx = np.exp(-gamma * _sqdist(x_ref, x_ref))
    yy = np.exp(-gamma * _sqdist(x_new, x_new))
    xy = np.exp(-gamma * _sqdist(x_ref, x_new))
    return float(np.mean(xx) + np.mean(yy) - 2.0 * np.mean(xy))


def scalar_mmd(x_ref: np.ndarray, x_new: np.ndarray, max_n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    left, right = _subsample_pair(x_ref, x_new, max_n, rng)
    return rbf_mmd2(left, right, _gamma(left, right))


def mmd_loco(x_ref: np.ndarray, x_new: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left, right = _subsample_pair(np.asarray(x_ref, dtype=float), np.asarray(x_new, dtype=float), max_n, rng)
    gamma = _gamma(left, right)
    full = rbf_mmd2(left, right, gamma)
    p = left.shape[1]
    drop = np.zeros(p, dtype=float)
    for j in range(p):
        keep = [i for i in range(p) if i != j]
        drop[j] = full - rbf_mmd2(left[:, keep], right[:, keep], gamma)
    return drop


def _center(kernel: np.ndarray) -> np.ndarray:
    row = kernel.mean(axis=1, keepdims=True)
    col = kernel.mean(axis=0, keepdims=True)
    return kernel - row - col + kernel.mean()


def _hsic(x_mat: np.ndarray, batch: np.ndarray, gamma: float) -> float:
    n = x_mat.shape[0]
    k_cent = _center(np.exp(-gamma * _sqdist(x_mat, x_mat)))
    labels = np.asarray(batch, dtype=float).reshape(-1)
    ell = (labels[:, None] == labels[None, :]).astype(float)
    ell_cent = _center(ell)
    return float(np.sum(k_cent * ell_cent) / (n - 1) ** 2)


def hsic_loco(x_ref: np.ndarray, x_new: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    """Drop in dependence between X and the batch label when feature j is removed."""
    rng = np.random.default_rng(seed)
    left, right = _subsample_pair(np.asarray(x_ref, dtype=float), np.asarray(x_new, dtype=float), max_n, rng)
    x_mat = np.vstack([left, right])
    batch = np.concatenate([np.zeros(len(left)), np.ones(len(right))])
    gamma = _gamma(left, right)
    full = _hsic(x_mat, batch, gamma)
    p = x_mat.shape[1]
    out = np.zeros(p, dtype=float)
    for j in range(p):
        keep = [i for i in range(p) if i != j]
        out[j] = full - _hsic(x_mat[:, keep], batch, gamma)
    return out


def domain_permutation_vimp(
    x_ref: np.ndarray,
    x_new: np.ndarray,
    seed: int,
    n_trees: int,
    max_depth: int,
    holdout: float,
) -> np.ndarray:
    """Holdout accuracy drop of a forest that predicts the batch label."""
    rng = np.random.default_rng(seed)
    x_mat = np.vstack([np.asarray(x_ref, dtype=float), np.asarray(x_new, dtype=float)])
    batch = np.concatenate([
        np.zeros(len(x_ref), dtype=int),
        np.ones(len(x_new), dtype=int),
    ])
    n, p = x_mat.shape
    order = rng.permutation(n)
    cut = max(1, int(round(n * (1.0 - holdout))))
    cut = min(cut, n - 1)
    train, test = order[:cut], order[cut:]
    forest = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        min_samples_leaf=5,
        n_jobs=1,
        random_state=seed,
    )
    forest.fit(x_mat[train], batch[train])
    base = float(np.mean(forest.predict(x_mat[test]) == batch[test]))
    vimp = np.zeros(p, dtype=float)
    x_test = x_mat[test]
    y_test = batch[test]
    for j in range(p):
        permuted = x_test.copy()
        permuted[:, j] = rng.permutation(permuted[:, j])
        vimp[j] = base - float(np.mean(forest.predict(permuted) == y_test))
    return vimp


def _standardize_fit(x_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = x_mat.mean(axis=0)
    sd = x_mat.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def _apply_scale(x_mat: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x_mat - mu) / sd


def _ridge_fit(x_mat: np.ndarray, y_vec: np.ndarray, alpha: float, sample_weight: Optional[np.ndarray] = None) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(x_mat, y_vec, sample_weight=sample_weight)
    return model


def _cross_fit_indices(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(n)
    mid = n // 2
    return order[:mid], order[mid:]


def r_risk_signal(x_mat: np.ndarray, y_vec: np.ndarray, w_vec: np.ndarray, alpha: float, seed: int) -> float:
    """Positive part of the R-learner risk reduction against tau = 0.

    An effect size, not a p-value: under a fixed concept drift it stays
    O(1) as n grows. The historical MAD, not n, supplies the scale for z.
    """
    x_mat = np.asarray(x_mat, dtype=float)
    y_vec = np.asarray(y_vec, dtype=float).reshape(-1)
    w_vec = np.asarray(w_vec, dtype=int).reshape(-1)
    n = x_mat.shape[0]
    rng = np.random.default_rng(seed)
    first, second = _cross_fit_indices(n, rng)
    mu = np.zeros(n, dtype=float)
    propensity = np.zeros(n, dtype=float)
    for test_idx in (first, second):
        train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)
        mu_loc, mu_sd = _standardize_fit(x_mat[train_idx])
        x_train = _apply_scale(x_mat[train_idx], mu_loc, mu_sd)
        x_test = _apply_scale(x_mat[test_idx], mu_loc, mu_sd)
        mu[test_idx] = _ridge_fit(x_train, y_vec[train_idx], alpha).predict(x_test)
        propensity[test_idx] = _propensity(x_train, w_vec[train_idx], x_test, seed)
    y_tilde = y_vec - mu
    w_tilde = w_vec - propensity
    tau = _fit_tau(x_mat, y_tilde, w_tilde, alpha, seed)
    risk0 = float(np.mean(y_tilde ** 2))
    risk = float(np.mean((y_tilde - w_tilde * tau) ** 2))
    return max(risk0 - risk, 0.0)


def _propensity(x_train: np.ndarray, w_train: np.ndarray, x_test: np.ndarray, seed: int) -> np.ndarray:
    if len(np.unique(w_train)) < 2:
        return np.full(len(x_test), 0.5)
    model = LogisticRegression(max_iter=250, solver="lbfgs", random_state=seed)
    model.fit(x_train, w_train)
    return np.clip(model.predict_proba(x_test)[:, 1], 0.1, 0.9)


def _fit_tau(
    x_mat: np.ndarray,
    y_tilde: np.ndarray,
    w_tilde: np.ndarray,
    alpha: float,
    seed: int,
) -> np.ndarray:
    del seed
    mask = np.abs(w_tilde) > 1e-3
    if int(mask.sum()) < x_mat.shape[1] + 2:
        return np.zeros(len(y_tilde), dtype=float)
    target = y_tilde[mask] / w_tilde[mask]
    weights = w_tilde[mask] ** 2
    mu, sd = _standardize_fit(x_mat[mask])
    model = _ridge_fit(_apply_scale(x_mat[mask], mu, sd), target, alpha, sample_weight=weights)
    return model.predict(_apply_scale(x_mat, mu, sd))


def _dr_phi(
    x_mat: np.ndarray,
    y_vec: np.ndarray,
    w_vec: np.ndarray,
    alpha: float,
    clip_e: float,
    seed: int,
) -> np.ndarray:
    """Cross-fit AIPW score for E[Y | X, batch=1] - E[Y | X, batch=0]."""
    n = x_mat.shape[0]
    rng = np.random.default_rng(seed)
    first, second = _cross_fit_indices(n, rng)
    mu0 = np.zeros(n, dtype=float)
    mu1 = np.zeros(n, dtype=float)
    propensity = np.zeros(n, dtype=float)
    for test_idx in (first, second):
        train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)
        x_train = x_mat[train_idx]
        y_train = y_vec[train_idx]
        w_train = w_vec[train_idx]
        mu_loc, mu_sd = _standardize_fit(x_train)
        x_train_s = _apply_scale(x_train, mu_loc, mu_sd)
        x_test_s = _apply_scale(x_mat[test_idx], mu_loc, mu_sd)
        for label, store in ((0, mu0), (1, mu1)):
            rows = w_train == label
            if int(rows.sum()) < x_mat.shape[1] + 2:
                store[test_idx] = float(np.mean(y_train))
                continue
            model = _ridge_fit(x_train_s[rows], y_train[rows], alpha)
            store[test_idx] = model.predict(x_test_s)
        propensity[test_idx] = _propensity(x_train_s, w_train, x_test_s, seed)
    propensity = np.clip(propensity, clip_e, 1.0 - clip_e)
    treated = w_vec == 1
    phi = (mu1 - mu0) + np.where(
        treated,
        (y_vec - mu1) / propensity,
        -(y_vec - mu0) / (1.0 - propensity),
    )
    return phi


def _loco_explained(x_mat: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Drop in mean(prediction^2) when feature j is removed from a ridge fit."""
    mu, sd = _standardize_fit(x_mat)
    scaled = _apply_scale(x_mat, mu, sd)
    full = _ridge_fit(scaled, target, alpha).predict(scaled)
    full_score = float(np.mean(full ** 2))
    p = x_mat.shape[1]
    out = np.zeros(p, dtype=float)
    for j in range(p):
        keep = [i for i in range(p) if i != j]
        pred = _ridge_fit(scaled[:, keep], target, alpha).predict(scaled[:, keep])
        out[j] = full_score - float(np.mean(pred ** 2))
    return out


def r_risk_loco(x_mat: np.ndarray, y_vec: np.ndarray, w_vec: np.ndarray, alpha: float, seed: int) -> np.ndarray:
    x_mat = np.asarray(x_mat, dtype=float)
    y_vec = np.asarray(y_vec, dtype=float).reshape(-1)
    w_vec = np.asarray(w_vec, dtype=int).reshape(-1)
    n = x_mat.shape[0]
    rng = np.random.default_rng(seed)
    first, second = _cross_fit_indices(n, rng)
    mu = np.zeros(n, dtype=float)
    propensity = np.full(n, 0.5)
    for test_idx in (first, second):
        train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)
        mu_loc, mu_sd = _standardize_fit(x_mat[train_idx])
        x_train = _apply_scale(x_mat[train_idx], mu_loc, mu_sd)
        x_test = _apply_scale(x_mat[test_idx], mu_loc, mu_sd)
        mu[test_idx] = _ridge_fit(x_train, y_vec[train_idx], alpha).predict(x_test)
        propensity[test_idx] = _propensity(x_train, w_vec[train_idx], x_test, seed)
    y_tilde = y_vec - mu
    w_tilde = w_vec - propensity
    mask = np.abs(w_tilde) > 1e-3
    if int(mask.sum()) < x_mat.shape[1] + 2:
        return np.zeros(x_mat.shape[1], dtype=float)
    target = np.zeros(n, dtype=float)
    target[mask] = y_tilde[mask] / w_tilde[mask]
    # Weight enters by repeating the LOCO regression on the masked rows only.
    return _loco_weighted(x_mat[mask], target[mask], w_tilde[mask] ** 2, alpha)


def _loco_weighted(
    x_mat: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mu, sd = _standardize_fit(x_mat)
    scaled = _apply_scale(x_mat, mu, sd)
    full = _ridge_fit(scaled, target, alpha, sample_weight=weights).predict(scaled)
    full_score = float(np.mean(full ** 2))
    p = x_mat.shape[1]
    out = np.zeros(p, dtype=float)
    for j in range(p):
        keep = [i for i in range(p) if i != j]
        pred = _ridge_fit(scaled[:, keep], target, alpha, sample_weight=weights).predict(scaled[:, keep])
        out[j] = full_score - float(np.mean(pred ** 2))
    return out


def po_risk_loco(
    x_mat: np.ndarray,
    y_vec: np.ndarray,
    w_vec: np.ndarray,
    alpha: float,
    clip_e: float,
    seed: int,
) -> Tuple[np.ndarray, float]:
    phi = _dr_phi(x_mat, y_vec, w_vec, alpha, clip_e, seed)
    return _loco_explained(x_mat, phi, alpha), float(np.mean(phi ** 2))


def dr_split_vimp(
    x_mat: np.ndarray,
    y_vec: np.ndarray,
    w_vec: np.ndarray,
    alpha: float,
    clip_e: float,
    seed: int,
    depth: int,
    leaf: int,
) -> np.ndarray:
    """Split importance of a shallow tree on the DR score.

    Same interface as a GRF split-importance vector: which coordinates the
    tree uses to approximate the batch contrast in the outcome. This is not
    a `grf` forest; the upload field is still one simplex vector.
    """
    phi = _dr_phi(x_mat, y_vec, w_vec, alpha, clip_e, seed + 17)
    tree = DecisionTreeRegressor(
        max_depth=depth,
        min_samples_leaf=leaf,
        random_state=seed,
    )
    tree.fit(x_mat, phi)
    return np.asarray(tree.feature_importances_, dtype=float)


def _scatter(full: np.ndarray, observed: np.ndarray, values: np.ndarray) -> None:
    full[np.asarray(observed)] = values


def group_vectors(
    x_ref: np.ndarray,
    x_new: np.ndarray,
    y_ref: np.ndarray,
    y_new: np.ndarray,
    feature_mask: np.ndarray,
    config: StatConfig,
    seed: int,
) -> dict:
    """Return member scores and the two simplex group vectors on the full index set."""
    mask = np.asarray(feature_mask, dtype=bool)
    observed = np.flatnonzero(mask)
    if observed.size == 0:
        raise ValueError("feature mask is empty")
    x_ref_o = np.asarray(x_ref, dtype=float)[:, observed]
    x_new_o = np.asarray(x_new, dtype=float)[:, observed]
    y_ref = np.asarray(y_ref, dtype=float).reshape(-1)
    y_new = np.asarray(y_new, dtype=float).reshape(-1)
    x_all = np.vstack([x_ref_o, x_new_o])
    y_all = np.concatenate([y_ref, y_new])
    w_all = np.concatenate([np.zeros(len(y_ref), dtype=int), np.ones(len(y_new), dtype=int)])
    p = int(mask.size)

    mmd = mmd_loco(x_ref_o, x_new_o, config.mmd_max_n, seed + 1)
    hsic = hsic_loco(x_ref_o, x_new_o, config.hsic_max_n, seed + 2)
    domain = domain_permutation_vimp(
        x_ref_o, x_new_o, seed + 3, config.rf_trees, config.rf_depth, config.holdout
    )
    po, po_signal = po_risk_loco(x_all, y_all, w_all, config.ridge_alpha, config.clip_e, seed + 4)
    r_loco = r_risk_loco(x_all, y_all, w_all, config.ridge_alpha, seed + 5)
    splits = dr_split_vimp(
        x_all, y_all, w_all, config.ridge_alpha, config.clip_e, seed + 6,
        config.tree_depth, config.tree_leaf,
    )
    members = {
        "mmd": simplex(mmd),
        "hsic": simplex(hsic),
        "domain": simplex(domain),
        "po": simplex(po),
        "r": simplex(r_loco),
        "split": simplex(splits),
    }
    vimp_cov_o = (members["mmd"] + members["hsic"] + members["domain"]) / 3.0
    vimp_con_o = (members["po"] + members["r"] + members["split"]) / 3.0
    vimp_cov = np.zeros(p, dtype=float)
    vimp_con = np.zeros(p, dtype=float)
    _scatter(vimp_cov, observed, vimp_cov_o)
    _scatter(vimp_con, observed, vimp_con_o)
    # Renormalize in case of a numerical drift; missing coordinates stay zero.
    vimp_cov = simplex(vimp_cov, mask)
    vimp_con = simplex(vimp_con, mask)
    mmd_scalar = scalar_mmd(x_ref_o, x_new_o, config.mmd_max_n, seed + 7)
    r_scalar = r_risk_signal(x_all, y_all, w_all, config.ridge_alpha, seed + 8)
    return {
        "vimp_cov": vimp_cov,
        "vimp_con": vimp_con,
        "members": members,
        "mmd": mmd_scalar,
        "r_signal": r_scalar,
        "po_signal": po_signal,
        "observed": observed,
    }


def z_inputs(mmd: float, r_signal: float) -> Tuple[float, float]:
    """Transforms whose placebo MAD z-scores are near N(0, 1) on a stationary null.

    Raw MMD^2 is right-skewed, so a median/MAD z-score fails ``|std-1|<=0.3``
    even when the reference window is stationary. ``log(MMD)`` removes that
    skew. ``asinh`` leaves R-risk on a similar scale without clipping a large
    concept signal to a hard cap.
    """
    return (
        float(np.log(max(float(mmd), 1e-8))),
        float(np.arcsinh(max(float(r_signal), 0.0))),
    )


def transform_stat(value: float, kind: str) -> float:
    if kind == "raw":
        return float(value)
    if kind == "asinh":
        return float(np.arcsinh(max(value, 0.0)))
    raise ValueError("unknown z transform: {0}".format(kind))


def robust_z(current: float, history: Sequence[float]) -> Tuple[float, float, float]:
    hist = np.asarray(list(history), dtype=float)
    med = float(np.median(hist))
    mad = float(np.median(np.abs(hist - med)))
    scale = 1.4826 * mad
    if scale < 1e-8:
        scale = float(np.std(hist, ddof=1)) + 1e-8
    return (float(current) - med) / scale, med, scale


def placebo_std(history: Sequence[float]) -> float:
    hist = np.asarray(list(history), dtype=float)
    scores = np.zeros(len(hist), dtype=float)
    for i in range(len(hist)):
        rest = np.delete(hist, i)
        scores[i], _, _ = robust_z(float(hist[i]), rest)
    if len(scores) < 3:
        return float("nan")
    return float(np.std(scores, ddof=1))


def calibration_ok(std_mmd: float, std_r: float, tol: float) -> bool:
    if not np.isfinite(std_mmd) or not np.isfinite(std_r):
        return False
    return abs(std_mmd - 1.0) <= tol and abs(std_r - 1.0) <= tol
