"""FSDS pseudo-outcome risk shares for a scorecard horse-race.

The reward is negative batch MSE. The FSDS weight of an arm is the
current-window residual of the doubly robust pseudo-outcome built from that
arm's scalar score, divided by the sum of those residuals. The index is the
UCB or Thompson index minus eta times a reward-scale times that share.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

ROUNDS = 80
BATCH = 200
REF_BATCHES = 6
REFRESH_EVERY = 8
REFRESH_START = 16
MID = ROUNDS // 2


def beta_on(d, cols, scale=0.8):
    b = np.zeros(d)
    sign = 1.0
    for c in cols:
        b[c] = sign * scale
        sign = -sign
    return b


def alpha_of(t, drastic):
    if drastic:
        return 0.0 if t < MID else 1.0
    start = MID // 2
    if t <= start:
        return 0.0
    return min(1.0, (t - start) / float(ROUNDS - 1 - start))


def blocks(d):
    a = list(range(0, max(d // 3, 1)))
    b = list(range(max(d // 3, 1), max(2 * d // 3, 2)))
    return a, b


def arm_names(family):
    if family == "relocate":
        return ("stale", "fresh", "blend", "zero")
    if family == "flip":
        return ("positive", "negative", "zero", "half")
    if family == "die":
        return ("both", "stable", "dying", "zero")
    if family == "nonlin":
        return ("linear", "linear_plus", "zero", "other_block")
    return ("linear", "zero", "half", "irrelevant")


def stream(Xpool, kind, seed):
    """Return y (T, B), preds (T, K, B), post mask, alpha path, arm names."""
    rng = np.random.default_rng(seed)
    d = Xpool.shape[1]
    cols_a, cols_b = blocks(d)
    bA = beta_on(d, cols_a)
    bB = beta_on(d, cols_b)
    drastic = kind.startswith("drastic")
    family = kind.split("_", 1)[1]
    y = np.zeros((ROUNDS, BATCH))
    preds = np.zeros((ROUNDS, 4, BATCH))
    post = np.zeros(ROUNDS, dtype=bool)
    alpha_path = np.zeros(ROUNDS)
    for t in range(ROUNDS):
        alpha = alpha_of(t, drastic)
        alpha_path[t] = alpha
        post[t] = alpha >= 0.5
        idx = rng.integers(0, len(Xpool), size=BATCH)
        X = Xpool[idx]
        lin_a = X @ bA
        lin_b = X @ bB
        noise = rng.normal(size=BATCH)
        if family == "relocate":
            y[t] = (1.0 - alpha) * lin_a + alpha * lin_b + 0.5 * noise
            preds[t, 0] = lin_a
            preds[t, 1] = lin_b
            preds[t, 2] = 0.5 * (lin_a + lin_b)
            preds[t, 3] = 0.0
        elif family == "flip":
            y[t] = (1.0 - 2.0 * alpha) * lin_a + 0.5 * noise
            preds[t, 0] = lin_a
            preds[t, 1] = -lin_a
            preds[t, 2] = 0.0
            preds[t, 3] = 0.5 * lin_a
        elif family == "die":
            y[t] = lin_a + (1.0 - alpha) * lin_b + 0.5 * noise
            preds[t, 0] = lin_a + lin_b
            preds[t, 1] = lin_a
            preds[t, 2] = lin_b
            preds[t, 3] = 0.0
        elif family == "nonlin":
            c0 = cols_b[0]
            c1 = cols_b[min(1, len(cols_b) - 1)]
            extra = 0.7 * np.sin(X[:, c0]) + 0.4 * X[:, c1] ** 2
            y[t] = lin_a + alpha * extra + 0.5 * noise
            preds[t, 0] = lin_a
            preds[t, 1] = lin_a + extra
            preds[t, 2] = 0.0
            preds[t, 3] = lin_b
        else:
            y[t] = lin_a + (0.4 + 1.6 * alpha) * noise
            preds[t, 0] = lin_a
            preds[t, 1] = 0.0
            preds[t, 2] = 0.5 * lin_a
            preds[t, 3] = lin_b
    return y, preds, post, alpha_path, arm_names(family)


def _lstsq(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or float(np.std(x)) < 1e-8:
        return np.array([float(np.mean(y)) if len(y) else 0.0, 0.0])
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def _apply(coef, x):
    return coef[0] + coef[1] * np.asarray(x, dtype=float)


def model_risk(score0, y0, score1, y1, seed):
    """Held-out residual variance of the current-window pseudo-outcome."""
    score0 = np.asarray(score0, dtype=float).reshape(-1)
    score1 = np.asarray(score1, dtype=float).reshape(-1)
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    y1 = np.asarray(y1, dtype=float).reshape(-1)
    s = np.concatenate([score0, score1])
    y = np.concatenate([y0, y1])
    T = np.concatenate([np.zeros(len(y0)), np.ones(len(y1))])
    rng = np.random.default_rng(seed)
    tr = []
    te = []
    for bit in (0, 1):
        idx = np.where(T == bit)[0]
        rng.shuffle(idx)
        mid = max(len(idx) // 2, 1)
        tr.append(idx[:mid])
        te.append(idx[mid:])
    tr = np.concatenate(tr)
    te = np.concatenate(te)
    te = te[T[te] == 1]
    if len(te) < 8 or min(int((T[tr] == 0).sum()), int((T[tr] == 1).sum())) < 8:
        return 1.0
    mu0 = _lstsq(s[tr][T[tr] == 0], y[tr][T[tr] == 0])
    mu1 = _lstsq(s[tr][T[tr] == 1], y[tr][T[tr] == 1])
    if float(np.std(s[tr])) < 1e-8:
        pi = np.full(len(te), 0.5)
        pi_tr = np.full(len(tr), 0.5)
    else:
        pi_m = LogisticRegression(max_iter=200, solver="lbfgs")
        pi_m.fit(s[tr].reshape(-1, 1), T[tr].astype(int))
        pi = np.clip(pi_m.predict_proba(s[te].reshape(-1, 1))[:, 1], 0.01, 0.99)
        pi_tr = np.clip(pi_m.predict_proba(s[tr].reshape(-1, 1))[:, 1], 0.01, 0.99)

    def phi_of(idx, pi_hat):
        mu0_e = _apply(mu0, s[idx])
        mu1_e = _apply(mu1, s[idx])
        fitted = np.where(T[idx] == 1, mu1_e, mu0_e)
        return (mu1_e - mu0_e) + (y[idx] - fitted) * (T[idx] - pi_hat) / (pi_hat * (1.0 - pi_hat))

    tau = _lstsq(s[tr], phi_of(tr, pi_tr))
    resid = phi_of(te, pi) - _apply(tau, s[te])
    return float(np.mean(resid ** 2))


def shares_at(y, preds, t, seed):
    k = preds.shape[1]
    y0 = y[:REF_BATCHES].reshape(-1)
    y1 = y[t - REF_BATCHES : t].reshape(-1)
    risks = np.empty(k)
    for arm in range(k):
        s0 = preds[:REF_BATCHES, arm].reshape(-1)
        s1 = preds[t - REF_BATCHES : t, arm].reshape(-1)
        risks[arm] = model_risk(s0, y0, s1, y1, seed + t + arm)
    total = float(risks.sum())
    if total <= 1e-12:
        return np.ones(k) / k
    return risks / total


def reward_scale(y, preds):
    """Std of the four arm rewards on the reference window. Tax units."""
    vals = []
    for i in range(REF_BATCHES):
        mse = np.mean((y[i] - preds[i]) ** 2, axis=1)
        vals.extend((-mse).tolist())
    return float(np.std(vals, ddof=1))


def share_path(y, preds, seed):
    path = {}
    for t in range(REFRESH_START, y.shape[0], REFRESH_EVERY):
        path[t] = shares_at(y, preds, t, seed)
    return path, reward_scale(y, preds)


def _mean_bonus(hist, t, kind):
    means = np.empty(len(hist))
    bonuses = np.empty(len(hist))
    logt = np.log(t + 1.0)
    for i, h in enumerate(hist):
        arr = np.asarray(h, dtype=float)
        if kind == "ducb":
            wt = 0.9 ** np.arange(len(arr) - 1, -1, -1)
            wsum = float(wt.sum())
            means[i] = float(np.dot(wt, arr) / wsum)
            bonuses[i] = np.sqrt(2.0 * logt / max(wsum, 1e-8))
        else:
            means[i] = float(arr.mean())
            if kind == "eps":
                bonuses[i] = 0.0
            else:
                bonuses[i] = np.sqrt(2.0 * logt / len(arr))
    return means, bonuses


def _ts_draw(hist, rng):
    draws = np.empty(len(hist))
    for i, h in enumerate(hist):
        arr = np.asarray(h, dtype=float)
        mu = float(arr.mean())
        if len(arr) < 2:
            sd = 1.0
        else:
            sd = float(np.sqrt(max(arr.var(ddof=1), 0.0) / len(arr)))
        draws[i] = float(rng.normal(mu, sd))
    return draws


def replay(y, preds, post, alpha, wpath, sigma, policy, seed):
    """policy: dict with keys rule, eta, mode, eps.

    mode 'scale' sets the penalty to eta * sigma * share.
    mode 'abs' sets the penalty to eta * share.
    """
    k = preds.shape[1]
    rounds = y.shape[0]
    hist = [[] for _ in range(k)]
    w = np.ones(k) / k
    early = 0.0
    ramp = 0.0
    post_ex = 0.0
    hit_early = 0
    hit_ramp = 0
    hit_post = 0
    n_early = 0
    n_ramp = 0
    n_post = 0
    path = np.zeros(rounds)
    rng = np.random.default_rng(seed)
    rule = policy["rule"]
    eta = float(policy["eta"])
    mode = policy["mode"]
    eps = float(policy["eps"])
    lam = (eta * sigma) if mode == "scale" else eta
    for t in range(rounds):
        if t in wpath:
            w = wpath[t]
        mse = np.mean((y[t] - preds[t]) ** 2, axis=1)
        best = int(np.argmin(mse))
        pays = -mse
        if t < k:
            arm = t
        else:
            explore = rule == "eps" and rng.random() < eps
            if explore:
                arm = int(rng.integers(0, k))
            elif rule == "ts":
                index = _ts_draw(hist, rng) - lam * w
                arm = int(np.argmax(index))
            else:
                means, bonuses = _mean_bonus(hist, t, "ducb" if rule == "ducb" else rule)
                index = means + bonuses - lam * w
                arm = int(np.argmax(index))
        hist[arm].append(float(pays[arm]))
        excess = float(mse[arm] - mse[best])
        path[t] = excess
        correct = int(arm == best)
        if alpha[t] <= 0.0:
            early += excess
            hit_early += correct
            n_early += 1
        elif post[t]:
            post_ex += excess
            hit_post += correct
            n_post += 1
        else:
            ramp += excess
            hit_ramp += correct
            n_ramp += 1
    hits = (
        hit_early / n_early if n_early else 0.0,
        hit_ramp / n_ramp if n_ramp else 0.0,
        hit_post / n_post if n_post else 0.0,
    )
    return early, ramp, post_ex, hits, path, lam
