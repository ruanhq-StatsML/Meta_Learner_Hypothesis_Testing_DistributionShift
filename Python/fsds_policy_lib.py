"""Finite-horizon policy horse-race with an FSDS pseudo-outcome share.

Four linear policies compete. The environment reward drifts from one feature
block to the next. The scalar score of a policy is the return it would earn
if the world matched its own coefficient. The pseudo-outcome residual of that
score is the arm-level OOD weight.
"""

from __future__ import annotations

import numpy as np

from fsds_porisk_lib import model_risk

H = 4
EP = 200
GAMMA = 0.9
DECAY = 0.85
LENGTH = 4
STEP = 4


def alpha_single(t):
    if t <= 20:
        return 0.0, 0.0
    a1 = min(1.0, (t - 20) / 59.0)
    return a1, 0.0


def alpha_double(t, rounds):
    if t <= 20:
        a1 = 0.0
    elif t >= 50:
        a1 = 1.0
    else:
        a1 = (t - 20) / 30.0
    if t <= 70:
        a2 = 0.0
    elif t >= min(100, rounds - 1):
        a2 = 1.0
    else:
        a2 = (t - 70) / 30.0
    return a1, a2


def phase_of(a1, a2):
    if a2 >= 0.5:
        return 2
    if a1 >= 0.5:
        return 1
    return 0


def rollout(s0, beta_act, beta_rew, v, noise):
    s = np.array(s0, dtype=float, copy=True)
    ret = np.zeros(len(s0))
    disc = 1.0
    feats = []
    rewards = []
    for h in range(H):
        action = np.sign(s @ beta_act)
        reward = action * (s @ beta_rew) - 0.05 * (action ** 2)
        if noise is not None:
            reward = reward + noise[h]
        ret += disc * reward
        feats.append(action[:, None] * s)
        rewards.append(reward.copy())
        s = DECAY * s + (1.0 - DECAY) * action[:, None] * v
        disc *= GAMMA
    return ret, np.concatenate(feats, axis=0), np.concatenate(rewards, axis=0)


def _betas(d):
    cols = [
        list(range(0, max(d // 3, 1))),
        list(range(max(d // 3, 1), max(2 * d // 3, 2))),
        list(range(max(2 * d // 3, 2), d)) or [d - 1],
    ]
    from fsds_porisk_lib import beta_on

    return [
        beta_on(d, cols[0]),
        beta_on(d, cols[1]),
        beta_on(d, cols[2]),
        np.zeros(d),
    ]


def build_pack(Xpool, seed, kind):
    rng = np.random.default_rng(seed)
    d = Xpool.shape[1]
    betas = _betas(d)
    v = np.ones(d) / np.sqrt(d)
    rounds = 80 if kind == "single" else 120
    alpha_fn = alpha_single if kind == "single" else (lambda t: alpha_double(t, rounds))
    G = np.zeros((rounds, 4, EP))
    score = np.zeros((rounds, 4, EP))
    s0 = np.zeros((rounds, EP, d))
    noise = rng.normal(scale=0.25, size=(rounds, H, EP))
    phase = np.zeros(rounds, dtype=int)
    true_beta = np.zeros((rounds, d))
    for t in range(rounds):
        a1, a2 = alpha_fn(t)
        phase[t] = phase_of(a1, a2)
        beta = (1.0 - a1) * (1.0 - a2) * betas[0] + a1 * (1.0 - a2) * betas[1] + a2 * betas[2]
        true_beta[t] = beta
        idx = rng.integers(0, len(Xpool), size=EP)
        s0[t] = Xpool[idx]
        for k in range(4):
            G[t, k], _, _ = rollout(s0[t], betas[k], beta, v, noise[t])
            score[t, k], _, _ = rollout(s0[t], betas[k], betas[k], v, None)
    return {
        "G": G,
        "score": score,
        "s0": s0,
        "noise": noise,
        "phase": phase,
        "betas": betas,
        "v": v,
        "true_beta": true_beta,
    }


def _window_risk(score, outcome, t, seed, arm):
    ref = slice(t - 2 * LENGTH, t - LENGTH)
    cur = slice(t - LENGTH, t)
    if outcome[ref, arm].size == 0 or outcome[cur, arm].size == 0:
        return 1.0
    try:
        return model_risk(
            score[ref, arm].reshape(-1),
            outcome[ref, arm].reshape(-1),
            score[cur, arm].reshape(-1),
            outcome[cur, arm].reshape(-1),
            seed + 100 * arm + t,
        )
    except Exception:
        return 1.0


def oracle_share_path(pack, seed):
    G = pack["G"]
    score = pack["score"]
    rounds = G.shape[0]
    path = {}
    for t in range(2 * LENGTH, rounds, STEP):
        risks = np.array([_window_risk(score, G, t, seed, k) for k in range(4)])
        total = float(risks.sum())
        path[t] = np.ones(4) / 4 if total <= 1e-12 else risks / total
    vals = []
    for t in range(LENGTH):
        vals.extend(G[t].mean(axis=1).tolist())
    sigma = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
    return path, max(sigma, 1e-6)


def _imagined(pack, t0, t1, chosen):
    feats = []
    rewards = []
    for bt in range(t0, t1):
        k = chosen[bt]
        _, feat, reward = rollout(
            pack["s0"][bt],
            pack["betas"][k],
            pack["true_beta"][bt],
            pack["v"],
            pack["noise"][bt],
        )
        feats.append(feat)
        rewards.append(reward)
    beta_hat, *_ = np.linalg.lstsq(np.concatenate(feats), np.concatenate(rewards), rcond=None)
    out = np.zeros((t1 - t0, 4, EP))
    for j, bt in enumerate(range(t0, t1)):
        for k in range(4):
            ret, _, _ = rollout(pack["s0"][bt], pack["betas"][k], beta_hat, pack["v"], None)
            out[j, k] = ret
    return out


def model_share(pack, t, chosen, seed):
    imagined = _imagined(pack, t - 2 * LENGTH, t, chosen)
    # Layout matches a slice of G: index 0 is round t-2L.
    score = pack["score"][t - 2 * LENGTH : t]
    risks = []
    for k in range(4):
        ref = slice(0, LENGTH)
        cur = slice(LENGTH, 2 * LENGTH)
        try:
            risks.append(
                model_risk(
                    score[ref, k].reshape(-1),
                    imagined[ref, k].reshape(-1),
                    score[cur, k].reshape(-1),
                    imagined[cur, k].reshape(-1),
                    seed + 100 * k + t,
                )
            )
        except Exception:
            risks.append(1.0)
    risks = np.asarray(risks, dtype=float)
    total = float(risks.sum())
    if total <= 1e-12:
        return np.ones(4) / 4
    return risks / total


def _onpolicy_shares(logs, t, seed):
    risks = []
    for arm in range(4):
        y0, s0, y1, s1 = [], [], [], []
        start = t - 2 * LENGTH
        for bt, sc, yy in logs[arm]:
            if start <= bt < start + LENGTH:
                y0.append(yy)
                s0.append(sc)
            elif t - LENGTH <= bt < t:
                y1.append(yy)
                s1.append(sc)
        if not y0 or not y1:
            risks.append(1.0)
            continue
        try:
            risks.append(
                model_risk(
                    np.concatenate(s0),
                    np.concatenate(y0),
                    np.concatenate(s1),
                    np.concatenate(y1),
                    seed + 100 * arm + t,
                )
            )
        except Exception:
            risks.append(1.0)
    risks = np.asarray(risks, dtype=float)
    total = float(risks.sum())
    if total <= 1e-12:
        return np.ones(4) / 4
    return risks / total


def replay_policy(pack, seed, policy, oracle_path=None, sigma=1.0):
    G = pack["G"]
    score = pack["score"]
    rounds = G.shape[0]
    rule = policy["rule"]
    eta = float(policy["eta"])
    mode = policy["mode"]
    eps = float(policy["eps"])
    monitor = policy["monitor"]
    lam = (eta * sigma) if mode == "scale" else eta
    hist = [[] for _ in range(4)]
    logs = [[] for _ in range(4)]
    w = np.ones(4) / 4.0
    totals = np.zeros(3)
    hits = np.zeros(3)
    counts = np.zeros(3)
    path = np.zeros(rounds)
    chosen = []
    rng = np.random.default_rng(seed + 17)
    for t in range(rounds):
        refresh = t >= 2 * LENGTH and t % STEP == 0
        if refresh and monitor == "oracle":
            w = oracle_path[t]
        elif refresh and monitor == "onpolicy":
            w = _onpolicy_shares(logs, t, seed)
        elif refresh and monitor == "model":
            w = model_share(pack, t, chosen, seed)
        if t < 4:
            arm = t
        else:
            explore = rule == "eps" and rng.random() < eps
            if explore:
                arm = int(rng.integers(0, 4))
            elif rule == "ts":
                from fsds_porisk_lib import _ts_draw

                arm = int(np.argmax(_ts_draw(hist, rng) - lam * w))
            else:
                from fsds_porisk_lib import _mean_bonus

                means, bonuses = _mean_bonus(hist, t, "ducb" if rule == "ducb" else rule)
                arm = int(np.argmax(means + bonuses - lam * w))
        chosen.append(arm)
        hist[arm].append(float(G[t, arm].mean()))
        if monitor == "onpolicy":
            logs[arm].append((t, score[t, arm], G[t, arm]))
        best = int(np.argmax(G[t].mean(axis=1)))
        excess = float(G[t].mean(axis=1).max() - G[t, arm].mean())
        ph = int(pack["phase"][t])
        totals[ph] += excess
        hits[ph] += int(arm == best)
        counts[ph] += 1
        path[t] = excess
    hit_rate = [hits[i] / counts[i] if counts[i] else 0.0 for i in range(3)]
    return totals, hit_rate, path, lam
