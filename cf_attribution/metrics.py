"""Distributional metrics used by the attribution score and the protocol."""

from __future__ import annotations

import numpy as np
from scipy import stats


def pairwise_concordance(score, truth) -> float:
    """Fraction of pairs with the same strict order. Ties in truth are skipped."""
    s = np.asarray(score, dtype=float)
    t = np.asarray(truth, dtype=float)
    n = len(s)
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dt = np.sign(t[i] - t[j])
            if dt == 0 or not np.isfinite(dt):
                continue
            total += 1
            if np.sign(s[i] - s[j]) == dt:
                agree += 1
    if total == 0:
        return float("nan")
    return float(agree / total)


def spearman(score, truth) -> float:
    s = np.asarray(score, dtype=float)
    t = np.asarray(truth, dtype=float)
    mask = np.isfinite(s) & np.isfinite(t)
    if mask.sum() < 3:
        return float("nan")
    if np.std(s[mask]) < 1e-12 or np.std(t[mask]) < 1e-12:
        return float("nan")
    rho, _ = stats.spearmanr(s[mask], t[mask])
    return float(rho)


def median_bandwidth(x: np.ndarray, y: np.ndarray | None = None, cap: int = 256) -> float:
    rng = np.random.default_rng(0)
    z = np.asarray(x, dtype=float)
    if y is not None:
        z = np.vstack([z, np.asarray(y, dtype=float)])
    if len(z) > cap:
        z = z[rng.choice(len(z), cap, replace=False)]
    d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    vals = d2[np.triu_indices(len(z), k=1)]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 1.0
    med = float(np.median(np.sqrt(vals)))
    return med if med > 1e-6 else 1.0


def mmd2_rff(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float | None = None,
    n_freq: int = 128,
    seed: int = 2026,
) -> float:
    """Biased-free random-Fourier estimate of RBF MMD^2. Always non-negative."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if x.shape[1] != y.shape[1]:
        raise ValueError("MMD inputs must share the feature dimension")
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if sigma is None:
        sigma = median_bandwidth(x, y)
    sigma = float(max(sigma, 1e-6))
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0 / sigma, size=(n_freq, x.shape[1]))
    b = rng.uniform(0.0, 2.0 * np.pi, size=n_freq)

    def phi(z: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0 / n_freq) * np.cos(z @ w.T + b)

    diff = phi(x).mean(axis=0) - phi(y).mean(axis=0)
    return float(np.sum(diff * diff))


def split_noise_floor(y: np.ndarray, seed: int = 0, n_freq: int = 64, sigma: float | None = None) -> float:
    """MMD between two halves of one sample. A floor for 'no signal'."""
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if len(y) < 4:
        return 0.0
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    mid = len(y) // 2
    return mmd2_rff(y[perm[:mid]], y[perm[mid:]], sigma=sigma, n_freq=n_freq, seed=seed)


def hsic(x: np.ndarray, y: np.ndarray, cap: int = 180, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    n = len(x)
    if n != len(y) or n < 5:
        return float("nan")
    if n > cap:
        idx = rng.choice(n, cap, replace=False)
        x, y = x[idx], y[idx]
        n = cap

    def rbf(a: np.ndarray) -> np.ndarray:
        d2 = np.sum((a[:, None, :] - a[None, :, :]) ** 2, axis=-1)
        pos = d2[d2 > 0]
        sig2 = 0.5 * float(np.median(pos)) if len(pos) else 1.0
        sig2 = max(sig2, 1e-8)
        return np.exp(-d2 / (2.0 * sig2))

    k = rbf(x)
    ell = rbf(y)
    h = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(k @ h @ ell @ h) / (n - 1) ** 2)


def hsic_ratio(cond, obs, gen, seed: int = 0) -> float:
    num = hsic(cond, gen, seed=seed)
    den = hsic(cond, obs, seed=seed + 1)
    if not np.isfinite(num) or not np.isfinite(den) or den < 1e-8:
        return float("nan")
    return float(num / den)


def randomized_pit(obs: np.ndarray, samples: np.ndarray, rng: np.random.Generator) -> float:
    """samples: (S, d), obs: (d,). Mean randomized PIT across dimensions."""
    obs = np.asarray(obs, dtype=float).reshape(-1)
    samples = np.asarray(samples, dtype=float)
    pits = []
    for j in range(obs.shape[0]):
        col = samples[:, j]
        less = np.mean(col < obs[j])
        equal = np.mean(col == obs[j])
        pits.append(less + float(rng.random()) * equal)
    return float(np.mean(pits))


def wasserstein_nd(obs: np.ndarray, gen: np.ndarray) -> float:
    obs = np.atleast_2d(np.asarray(obs, dtype=float))
    gen = np.atleast_2d(np.asarray(gen, dtype=float))
    dists = []
    for j in range(obs.shape[1]):
        dists.append(stats.wasserstein_distance(obs[:, j], gen[:, j]))
    return float(np.mean(dists))


def ks_uniform_p(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 8:
        return float("nan")
    return float(stats.kstest(v, "uniform").pvalue)


def evalue_from_smd(d: float) -> float:
    """Approximate E-value from a standardized mean difference.

    Uses the Ding-VanderWeele conversion RR ≈ exp(0.91 |d|) and the
    standard E-value formula for a risk ratio.
    """
    if not np.isfinite(d):
        return float("nan")
    rr = float(np.exp(0.91 * abs(d)))
    if rr < 1.0:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def cohens_d(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled < 1e-12:
        return 0.0 if abs(np.mean(a) - np.mean(b)) < 1e-12 else float("inf")
    return float((np.mean(a) - np.mean(b)) / pooled)


def welch_p(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    res = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided")
    return float(res.pvalue)
