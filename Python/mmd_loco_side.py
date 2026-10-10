"""Numpy port of the MMD leave-one-covariate-out scores.

The algebra matches the JAX snippet: one RBF bandwidth for the joint sample,
then each coordinate is removed by subtracting its squared differences from the
three pairwise distance matrices. This file is a side study of covariate-shift
power. It is not the FSDS horse-race.
"""

from __future__ import annotations

import numpy as np


def pairwise_sq_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    a2 = np.sum(A * A, axis=1)[:, None]
    b2 = np.sum(B * B, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)


def median_heuristic(X_ref: np.ndarray, X_new: np.ndarray, max_points: int = 1000, seed: int = 0) -> float:
    """gamma in exp(-gamma * ||x-y||^2), with gamma = 1 / median squared distance.

    The median is taken over the upper triangle of the pooled sample. A label
    permutation does not change that pool, so the bandwidth stays fixed inside
    a permutation test.
    """
    Z = np.vstack([np.asarray(X_ref, dtype=np.float32), np.asarray(X_new, dtype=np.float32)])
    if Z.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        Z = Z[rng.choice(Z.shape[0], size=max_points, replace=False)]
    D2 = pairwise_sq_distances(Z, Z)
    med = np.median(D2[np.triu_indices(Z.shape[0], k=1)])
    if not np.isfinite(med) or med <= 1e-12:
        return 1.0
    return float(1.0 / med)


def _subsample_pair(X_ref, X_new, max_n: int, seed: int):
    rng = np.random.default_rng(seed)
    X_ref = np.asarray(X_ref, dtype=np.float32)
    X_new = np.asarray(X_new, dtype=np.float32)
    if X_ref.shape[0] > max_n:
        X_ref = X_ref[rng.choice(X_ref.shape[0], size=max_n, replace=False)]
    if X_new.shape[0] > max_n:
        X_new = X_new[rng.choice(X_new.shape[0], size=max_n, replace=False)]
    return X_ref, X_new


def _kernel_terms(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray):
    """Unbiased MMD^2 and its normalization by the two within-sample kernel means.

    NAMMD here is MMD^2_u / sqrt(mean off-diagonal Kxx * mean off-diagonal Kyy).
    Diagonals of an RBF Gram matrix are one, including after a coordinate drop.
    """
    n = Kxx.shape[0]
    m = Kyy.shape[0]
    off_xx = (Kxx.sum(axis=(0, 1)) - n) / (n * (n - 1))
    off_yy = (Kyy.sum(axis=(0, 1)) - m) / (m * (m - 1))
    cross = Kxy.sum(axis=(0, 1)) / (n * m)
    mmd = off_xx + off_yy - 2.0 * cross
    scale = np.sqrt(np.maximum(off_xx, 1e-12) * np.maximum(off_yy, 1e-12))
    return mmd, mmd / scale


def mmd_and_loco(X_ref: np.ndarray, X_new: np.ndarray, gamma: float):
    """Joint scores and the leave-one-coordinate drops.

    Returns unbiased MMD^2, NAMMD, and both LOCO vectors
    (score on all coordinates minus score with that coordinate removed).
    """
    X_ref = np.asarray(X_ref, dtype=np.float32)
    X_new = np.asarray(X_new, dtype=np.float32)
    gamma = np.float32(gamma)
    Dxx = pairwise_sq_distances(X_ref, X_ref)
    Dyy = pairwise_sq_distances(X_new, X_new)
    Dxy = pairwise_sq_distances(X_ref, X_new)
    Kxx = np.exp(-gamma * Dxx)
    Kyy = np.exp(-gamma * Dyy)
    Kxy = np.exp(-gamma * Dxy)
    mmd, nammd = _kernel_terms(Kxx, Kyy, Kxy)

    dxx = (X_ref[:, None, :] - X_ref[None, :, :]) ** 2
    dyy = (X_new[:, None, :] - X_new[None, :, :]) ** 2
    dxy = (X_ref[:, None, :] - X_new[None, :, :]) ** 2
    Kxx_j = Kxx[:, :, None] * np.exp(gamma * dxx)
    Kyy_j = Kyy[:, :, None] * np.exp(gamma * dyy)
    Kxy_j = Kxy[:, :, None] * np.exp(gamma * dxy)
    mmd_j, nammd_j = _kernel_terms(Kxx_j, Kyy_j, Kxy_j)
    return {
        "mmd": float(mmd),
        "nammd": float(nammd),
        "mmd_loco": np.asarray(mmd - mmd_j, dtype=np.float64),
        "nammd_loco": np.asarray(nammd - nammd_j, dtype=np.float64),
    }


def _aggregate(raw: dict) -> dict:
    loco = raw["mmd_loco"]
    nloco = raw["nammd_loco"]
    return {
        "mmd": raw["mmd"],
        "mmd_maxloco": float(np.max(loco)),
        "mmd_posloco": float(np.sum(np.maximum(loco, 0.0))),
        "nammd": raw["nammd"],
        "nammd_maxloco": float(np.max(nloco)),
        "nammd_posloco": float(np.sum(np.maximum(nloco, 0.0))),
        "loco_hat": int(np.argmax(loco)),
    }


def mmd_loco(X_ref, X_new, gamma=None, max_n: int = 1000, seed: int = 0) -> np.ndarray:
    X_ref, X_new = _subsample_pair(X_ref, X_new, max_n, seed)
    if gamma is None:
        gamma = median_heuristic(X_ref, X_new, seed=seed)
    return mmd_and_loco(X_ref, X_new, gamma)["mmd_loco"]


def nammd_loco(X_ref, X_new, gamma=None, max_n: int = 1000, seed: int = 0) -> np.ndarray:
    X_ref, X_new = _subsample_pair(X_ref, X_new, max_n, seed)
    if gamma is None:
        gamma = median_heuristic(X_ref, X_new, seed=seed)
    return mmd_and_loco(X_ref, X_new, gamma)["nammd_loco"]


def permutation_test(X_ref, X_new, n_perm: int, seed: int, gamma: float | None = None) -> dict:
    """Same bandwidth, same permutations, six scalar statistics.

    p = (1 + #{permuted statistics >= observed}) / (n_perm + 1).
    """
    X_ref = np.asarray(X_ref, dtype=np.float32)
    X_new = np.asarray(X_new, dtype=np.float32)
    if gamma is None:
        gamma = median_heuristic(X_ref, X_new, seed=seed)
    observed = _aggregate(mmd_and_loco(X_ref, X_new, gamma))
    keys = ("mmd", "mmd_maxloco", "mmd_posloco", "nammd", "nammd_maxloco", "nammd_posloco")
    counts = {k: 0 for k in keys}
    Z = np.vstack([X_ref, X_new])
    n = X_ref.shape[0]
    rng = np.random.default_rng(seed)
    for _ in range(n_perm):
        perm = rng.permutation(Z.shape[0])
        perm_stats = _aggregate(mmd_and_loco(Z[perm[:n]], Z[perm[n:]], gamma))
        for k in keys:
            counts[k] += perm_stats[k] >= observed[k]
    pvalues = {k: (1 + counts[k]) / (n_perm + 1) for k in keys}
    return {"gamma": gamma, "observed": observed, "pvalue": pvalues, "loco": None}


def loco_matches_column_delete(X_ref, X_new, gamma: float, atol: float = 1e-5) -> float:
    """Max absolute gap between the distance update and an explicit column drop."""
    raw = mmd_and_loco(X_ref, X_new, gamma)
    gaps = []
    p = X_ref.shape[1]
    for j in range(p):
        keep = [i for i in range(p) if i != j]
        dropped = mmd_and_loco(X_ref[:, keep], X_new[:, keep], gamma)
        gaps.append(abs((raw["mmd"] - dropped["mmd"]) - raw["mmd_loco"][j]))
        gaps.append(abs((raw["nammd"] - dropped["nammd"]) - raw["nammd_loco"][j]))
    return float(np.max(gaps))
