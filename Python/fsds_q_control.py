"""Short-horizon control under a gradual reward drift.

The previous Q rules were stateless indexes on a menu of scorecards. Here the
decision is an action in {-1, 0, +1} at each of four steps. The observable
state is the three-way sign of two known reference scores and the step
index. A second table also sees which reference policy currently holds the
pseudo-outcome share. That share is the same construction as the scorecard
race: every reference policy is scored on the batch, and the weight is
refreshed on the fixed calendar.

The committed menu (block A, block B or the flipped sign, a blend, and the
zero action) is the baseline. UCB and UCB at eta=2 still pick one policy for
the whole episode. They do not condition on the current signs.
"""

from __future__ import annotations

import numpy as np

from fsds_porisk_lib import (
    BATCH,
    REF_BATCHES,
    REFRESH_EVERY,
    REFRESH_START,
    ROUNDS,
    _mean_bonus,
    alpha_of,
    beta_on,
    blocks,
    model_risk,
)

H = 4
EP = BATCH
DECAY = 0.85
COST = 0.05
NOISE = 0.25
ACTIONS = np.array([-1.0, 0.0, 1.0])
N_SIGN = 9
N_W = 5
N_A = 3


def _parts(d):
    left, mid = blocks(d)
    used = set(left) | set(mid)
    right = [i for i in range(d) if i not in used]
    if not right:
        right = [mid[-1]]
    return left, mid, right


def _score_code(s, coef_a, coef_b):
    """Three-way sign of the two reference scores. Both scores are known."""

    def binarize(raw):
        code = np.ones(raw.shape[0], dtype=int)
        code[raw > COST] = 2
        code[raw < -COST] = 0
        return code

    return binarize(s @ coef_a) * 3 + binarize(s @ coef_b)


def _act_from_score(raw):
    a = np.zeros(raw.shape[0])
    a[raw > COST] = 1.0
    a[raw < -COST] = -1.0
    return a


def _push(s, a, v):
    return DECAY * s + (1.0 - DECAY) * a[:, None] * v


def _rollout(s, beta_act, beta_rew, v, noise):
    """Noise-free return, noisy return, and the action at step 0."""
    clean = np.zeros(s.shape[0])
    noisy = np.zeros(s.shape[0])
    first = None
    for h in range(H):
        a = _act_from_score(s @ beta_act)
        if first is None:
            first = a.copy()
        reward = a * (s @ beta_rew) - COST * (a ** 2)
        clean += reward
        noisy += reward + noise[h]
        s = _push(s, a, v)
    return clean, noisy, first


def _optimal(s, beta, v):
    """Finite-horizon optimum. Dynamics are deterministic, so the action tree is exact."""

    def value_action(state, steps):
        best_v = np.full(state.shape[0], -np.inf)
        best_a = np.zeros(state.shape[0])
        for action in ACTIONS:
            a = np.full(state.shape[0], float(action))
            reward = a * (state @ beta) - COST * (a ** 2)
            if steps == 1:
                cont = 0.0
            else:
                cont, _ = value_action(_push(state, a, v), steps - 1)
            total = reward + cont
            better = total > best_v
            best_v = np.where(better, total, best_v)
            best_a = np.where(better, a, best_a)
        return best_v, best_a

    return value_action(s, H)


def _betas(d, kind):
    left, mid, _right = _parts(d)
    block_a = beta_on(d, left)
    block_b = beta_on(d, mid)
    if kind == "control_flip":
        return [block_a, -block_a, np.zeros(d), 0.5 * block_a], block_a, block_b
    return [block_a, block_b, 0.5 * (block_a + block_b), np.zeros(d)], block_a, block_b


def _beta_at(kind, alpha, block_a, block_b):
    if kind == "control_flip":
        return (1.0 - 2.0 * alpha) * block_a
    return (1.0 - alpha) * block_a + alpha * block_b


def _share_at(score, outcome, t, seed):
    risks = np.empty(score.shape[1])
    for arm in range(score.shape[1]):
        risks[arm] = model_risk(
            score[:REF_BATCHES, arm].reshape(-1),
            outcome[:REF_BATCHES, arm].reshape(-1),
            score[t - REF_BATCHES : t, arm].reshape(-1),
            outcome[t - REF_BATCHES : t, arm].reshape(-1),
            seed + 17 * t + arm,
        )
    total = float(risks.sum())
    if total <= 1e-12:
        return np.ones(score.shape[1]) / score.shape[1]
    return risks / total


def prepare(Xpool, kind, seed):
    rng = np.random.default_rng(seed)
    d = Xpool.shape[1]
    policies, block_a, block_b = _betas(d, kind)
    parts = _parts(d)
    v = np.ones(d) / np.sqrt(d)
    noise = rng.normal(scale=NOISE, size=(ROUNDS, H, EP))
    s0 = np.zeros((ROUNDS, EP, d))
    beta = np.zeros((ROUNDS, d))
    alpha = np.zeros(ROUNDS)
    post = np.zeros(ROUNDS, dtype=bool)
    score = np.zeros((ROUNDS, 4, EP))
    outcome = np.zeros((ROUNDS, 4, EP))
    clean_mean = np.zeros((ROUNDS, 4))
    noisy_mean = np.zeros((ROUNDS, 4))
    opt = np.zeros(ROUNDS)
    opt_a = np.zeros((ROUNDS, EP))
    first_a = np.zeros((ROUNDS, 4, EP))
    for t in range(ROUNDS):
        alpha[t] = alpha_of(t, False)
        post[t] = alpha[t] >= 0.5
        beta[t] = _beta_at(kind, alpha[t], block_a, block_b)
        idx = rng.integers(0, len(Xpool), size=EP)
        s0[t] = Xpool[idx]
        for arm, coef in enumerate(policies):
            clean, noisy, first = _rollout(s0[t], coef, beta[t], v, noise[t])
            first_a[t, arm] = first
            outcome[t, arm] = noisy
            clean_mean[t, arm] = float(clean.mean())
            noisy_mean[t, arm] = float(noisy.mean())
            self_clean, _self_noisy, _ = _rollout(s0[t], coef, coef, v, np.zeros_like(noise[t]))
            score[t, arm] = self_clean
        opt_ret, opt_act = _optimal(s0[t], beta[t], v)
        opt[t] = float(opt_ret.mean())
        opt_a[t] = opt_act
    if np.any(opt + 1e-5 < clean_mean.max(axis=1)):
        raise RuntimeError("optimum fell below a committed policy")
    shares = np.ones((ROUNDS, 4)) / 4.0
    wbin = np.zeros(ROUNDS, dtype=int)
    current = np.ones(4) / 4.0
    for t in range(REFRESH_START, ROUNDS, REFRESH_EVERY):
        current = _share_at(score, outcome, t, seed)
        shares[t:] = current
        wbin[t:] = int(np.argmax(current)) + 1
    ref = noisy_mean[:REF_BATCHES].ravel()
    sigma = float(np.std(ref, ddof=1))
    opt_step = float(np.quantile(np.abs(s0[0] @ block_a), 0.9))
    return {
        "kind": kind,
        "s0": s0,
        "beta": beta,
        "v": v,
        "parts": parts,
        "noise": noise,
        "alpha": alpha,
        "post": post,
        "clean_mean": clean_mean,
        "noisy_mean": noisy_mean,
        "opt": opt,
        "opt_a": opt_a,
        "first_a": first_a,
        "shares": shares,
        "wbin": wbin,
        "sigma": max(sigma, 1e-6),
        "opt_step": max(opt_step, 0.5),
        "policies": policies,
    }


def _bucket(pack, t, excess, hit, boxes):
    if pack["alpha"][t] <= 0.0:
        key = "early"
    elif pack["post"][t]:
        key = "post"
    else:
        key = "ramp"
    boxes[key][0] += excess
    boxes[key][1] += hit
    boxes[key][2] += 1


def _finish(boxes):
    out = {}
    for key, (total, hits, count) in boxes.items():
        out[key] = total
        out[key + "_hit"] = hits / count if count else 0.0
    return out


def _fresh():
    return {key: [0.0, 0.0, 0] for key in ("early", "ramp", "post")}


def run_commit(pack, policy, seed):
    rng = np.random.default_rng(seed)
    hist = [[] for _ in range(4)]
    boxes = _fresh()
    rule = policy["rule"]
    eta = float(policy["eta"])
    eps = float(policy["eps"])
    lam = eta * pack["sigma"] if policy["mode"] == "scale" else eta
    kind = "ducb" if rule == "ducb" else ("eps" if rule == "eps" else "ucb")
    for t in range(ROUNDS):
        if t < 4:
            arm = t
        else:
            explore = rule == "eps" and rng.random() < eps
            if explore:
                arm = int(rng.integers(0, 4))
            else:
                means, bonuses = _mean_bonus(hist, t, kind)
                arm = int(np.argmax(means + bonuses - lam * pack["shares"][t]))
        hist[arm].append(float(pack["noisy_mean"][t, arm]))
        excess = float(pack["opt"][t] - pack["clean_mean"][t, arm])
        hit = float(np.mean(pack["first_a"][t, arm] == pack["opt_a"][t]))
        _bucket(pack, t, excess, hit, boxes)
    return _finish(boxes)


def _q_tables():
    shape = (N_SIGN, H, N_W, N_A)
    return np.zeros(shape), np.zeros(shape), np.zeros(shape)


def _batch_update(table, flat, target, alpha):
    uniq, inv = np.unique(flat, return_inverse=True)
    summed = np.zeros(len(uniq))
    count = np.zeros(len(uniq))
    np.add.at(summed, inv, target)
    np.add.at(count, inv, 1.0)
    view = table.ravel()
    view[uniq] += alpha * (summed / count - view[uniq])
    return uniq


def run_q(pack, policy, seed):
    rng = np.random.default_rng(seed)
    use_share = bool(policy["share"])
    kind = policy["qrule"]
    alpha_q = float(policy["alpha"])
    Q, Q2, counts = _q_tables()
    boxes = _fresh()
    v = pack["v"]
    coef_a = pack["policies"][0]
    coef_b = pack["policies"][1]
    for t in range(ROUNDS):
        beta = pack["beta"][t]
        wbin = int(pack["wbin"][t]) if use_share else 0
        state = pack["s0"][t].copy()
        achieved = np.zeros(state.shape[0])
        first_hit = None
        for h in range(H):
            code = _score_code(state, coef_a, coef_b)
            if kind == "dql":
                value = 0.5 * (Q[code, h, wbin] + Q2[code, h, wbin])
            else:
                value = Q[code, h, wbin]
            if kind in ("qucb", "dql"):
                seen = counts[code, h, wbin]
                bonus = np.sqrt(2.0 * np.log(t + 1.0) / np.maximum(seen, 1.0))
                bonus = np.where(seen > 0.0, bonus, 1e3)
                value = value + bonus
            jitter = rng.uniform(0.0, 1e-6, size=value.shape)
            a_idx = np.argmax(value + jitter, axis=1)
            action = ACTIONS[a_idx]
            if first_hit is None:
                first_hit = action == pack["opt_a"][t]
            reward = action * (state @ beta) - COST * (action ** 2)
            achieved += reward
            observed = reward + pack["noise"][t, h]
            nxt = _push(state, action, v)
            target = observed.copy()
            if h < H - 1:
                code_n = _score_code(nxt, coef_a, coef_b)
                if kind == "dql":
                    if rng.random() < 0.5:
                        online, other = Q, Q2
                    else:
                        online, other = Q2, Q
                    online_next = online[code_n, h + 1, wbin]
                    other_next = other[code_n, h + 1, wbin]
                    star = np.argmax(online_next + rng.uniform(0.0, 1e-8, size=online_next.shape), axis=1)
                    target += other_next[np.arange(state.shape[0]), star]
                    flat = ((code * H + h) * N_W + wbin) * N_A + a_idx
                    touched = _batch_update(online, flat, target, alpha_q)
                    counts.ravel()[touched] += 1.0
                else:
                    nxt_q = Q[code_n, h + 1, wbin]
                    star = np.argmax(nxt_q + rng.uniform(0.0, 1e-8, size=nxt_q.shape), axis=1)
                    target += nxt_q[np.arange(state.shape[0]), star]
                    flat = ((code * H + h) * N_W + wbin) * N_A + a_idx
                    touched = _batch_update(Q, flat, target, alpha_q)
                    counts.ravel()[touched] += 1.0
            else:
                flat = ((code * H + h) * N_W + wbin) * N_A + a_idx
                if kind == "dql":
                    online = Q if rng.random() < 0.5 else Q2
                    touched = _batch_update(online, flat, target, alpha_q)
                else:
                    touched = _batch_update(Q, flat, target, alpha_q)
                counts.ravel()[touched] += 1.0
            state = nxt
        excess = float(pack["opt"][t] - achieved.mean())
        hit = float(first_hit.mean()) if first_hit is not None else 0.0
        _bucket(pack, t, excess, hit, boxes)
    return _finish(boxes)


def run_spec(pack, policy, seed):
    if policy["family"] == "commit":
        return run_commit(pack, policy, seed)
    return run_q(pack, policy, seed)
