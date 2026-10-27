"""Semi-synthetic cohort panels with a known lag kernel.

The simulator is the only place a counterfactual ground truth exists.
Learned models never receive the shocks, the kernel, or the coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Impulse:
    t: int
    entity: int
    dose: float
    tag: str


@dataclass
class Panel:
    name: str
    grounded: bool
    entity_names: list[str]
    block: np.ndarray              # (T, E, d_b) standardized
    outcome: np.ndarray            # (T, M) standardized
    covariates: np.ndarray         # (T, M, d_x) standardized
    member_of: np.ndarray          # (M,) entity index
    overlap: np.ndarray            # (E, E)
    time_index: np.ndarray         # (T,)
    target: int
    mediator: int
    driver_of: np.ndarray          # (E,) driver entity, -1 if none
    peak_lag: int
    split_mode: str                # "member" or "time"
    train_end: int
    reference_index: dict[int, np.ndarray]
    notes: str
    assumption_mismatch: set[str] = field(default_factory=set)
    # Grounded-only bookkeeping. Real panels leave these empty.
    impulses: list[Impulse] = field(default_factory=list)
    raw_block: np.ndarray | None = None
    raw_outcome: np.ndarray | None = None
    shocks: np.ndarray | None = None
    x_mean: np.ndarray | None = None
    kernel: np.ndarray | None = None
    alpha_direct: np.ndarray | None = None
    alpha_med: float = 0.0
    beta: np.ndarray | None = None
    y_noise: np.ndarray | None = None
    block_mean: np.ndarray | None = None
    block_std: np.ndarray | None = None
    y_mean: float = 0.0
    y_std: float = 1.0
    queries: dict = field(default_factory=dict)
    entity_covariates: np.ndarray | None = None


def _kernel(peak: int, length: int = 14) -> np.ndarray:
    ks = np.arange(1, length + 1)
    if peak <= 1:
        w = np.exp(-1.4 * (ks - 1))
    else:
        w = np.exp(-0.5 * ((ks - peak) / 1.35) ** 2)
    return w / w.sum()


def _scale_train(x: np.ndarray, train_end: int, axis: int = 0):
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, train_end)
    mu = np.nanmean(x[tuple(sl)], axis=axis, keepdims=True)
    sd = np.nanstd(x[tuple(sl)], axis=axis, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (x - mu) / sd, np.squeeze(mu, axis=axis), np.squeeze(sd, axis=axis)


def simulate_states(
    shocks: np.ndarray,
    x_mean: np.ndarray,
    impulses: list[Impulse],
    mediator: int,
    dose_source: int,
    intervene: tuple[int, int, float] | None = None,
) -> np.ndarray:
    """Structural block dynamics. intervene = (time, entity, replacement of dim 0)."""
    t_len, n_ent, d_b = shocks.shape
    block = np.zeros_like(shocks)
    block[0] = shocks[0]
    impulse_at = {(imp.t, imp.entity): imp.dose for imp in impulses}
    for t in range(1, t_len):
        hist = block[max(0, t - 7) : t]
        persist = 0.72 * block[t - 1] + 0.08 * hist.mean(axis=0)
        cross = np.zeros((n_ent, d_b))
        if t >= 4:
            cross[mediator, 0] = 0.55 * block[t - 4, dose_source, 0]
        block[t] = persist + cross + 0.12 * x_mean[t] + shocks[t]
        for (et, ee), dose in impulse_at.items():
            if et == t:
                if intervene is not None and intervene[0] == t and intervene[1] == ee:
                    block[t, ee, 0] = intervene[2]
                else:
                    block[t, ee, 0] = block[t, ee, 0] + dose
        if intervene is not None and intervene[0] == t:
            # A calm-day intervention still replaces dim 0 even with no impulse.
            block[t, intervene[1], 0] = intervene[2]
    return block


def outcomes_from_block(
    block: np.ndarray,
    covariates: np.ndarray,
    member_of: np.ndarray,
    target: int,
    mediator: int,
    kernel: np.ndarray,
    alpha_direct: np.ndarray,
    alpha_med: float,
    beta: np.ndarray,
    y_noise: np.ndarray,
) -> np.ndarray:
    t_len, n_mem = y_noise.shape
    y = np.zeros((t_len, n_mem))
    lags = len(kernel)
    members = np.where(member_of == target)[0]
    for t in range(t_len):
        sig = 0.0
        for k in range(1, lags + 1):
            if t - k < 0:
                continue
            sig += kernel[k - 1] * float(alpha_direct @ block[t - k, :, 0])
            sig += kernel[k - 1] * alpha_med * block[t - k, mediator, 0]
        y[t, members] = sig + covariates[t, members] @ beta + y_noise[t, members]
    return y


def make_synthetic_panel(memory: str = "long", seed: int = 2026) -> Panel:
    if memory not in {"long", "short"}:
        raise ValueError(memory)
    rng = np.random.default_rng(seed)
    peak = 10 if memory == "long" else 1
    kernel = _kernel(peak)
    names = [
        "target",
        "mediator",
        "dose_source",
        "ov_1",
        "ov_2",
        "ov_3",
        "ov_4",
        "iso_1",
        "iso_2",
        "iso_3",
        "iso_4",
        "conn_1",
        "conn_2",
        "conn_3",
        "conn_4",
    ]
    n_ent = len(names)
    idx = {n: i for i, n in enumerate(names)}
    target = idx["target"]
    mediator = idx["mediator"]
    dose_source = idx["dose_source"]
    members_per = 32
    d_b = 4
    d_x = 3
    n_mem = n_ent * members_per
    # 5 dose + 4 overlap + 4 isolated + 4 connected = 17 slots.
    slot0 = 36
    gap = 16
    n_slots = 17
    last_slot = slot0 + (n_slots - 1) * gap
    # Eval day is slot + peak, and the kernel looks 14 days around it.
    train_end = last_slot + peak + 16
    t_len = train_end + 48

    alpha = np.zeros(n_ent)
    alpha[dose_source] = 1.6
    overlap_ids = [idx[f"ov_{i}"] for i in range(1, 5)]
    overlap_alpha = np.array([0.45, 0.95, 1.45, 1.95])
    for i, a in zip(overlap_ids, overlap_alpha):
        alpha[i] = float(a)
    iso_ids = [idx[f"iso_{i}"] for i in range(1, 5)]
    conn_ids = [idx[f"conn_{i}"] for i in range(1, 5)]
    for i in conn_ids:
        alpha[i] = 2.0

    overlap = np.eye(n_ent)
    # Plant overlap with the target equal to the direct coefficient, scaled
    # into (0, 1]. The overlap construction's ground truth is this number.
    for i, a in zip(overlap_ids, overlap_alpha):
        overlap[target, i] = overlap[i, target] = float(a / overlap_alpha.max())
    overlap[target, dose_source] = overlap[dose_source, target] = 0.8
    for i in conn_ids:
        overlap[target, i] = overlap[i, target] = 0.9
    for i in iso_ids:
        overlap[target, i] = overlap[i, target] = 0.05
    overlap[target, mediator] = overlap[mediator, target] = 0.7

    doses = np.array([1.2, 2.0, 2.8, 3.6, 4.4])
    impulses: list[Impulse] = []
    slot = 0
    dose_events = []
    for dose in doses:
        t = slot0 + slot * gap
        impulses.append(Impulse(t, dose_source, float(dose), "dose"))
        dose_events.append({"t": t, "entity": dose_source, "dose": float(dose), "tag": "dose"})
        slot += 1
    overlap_events = []
    for ent, alp in zip(overlap_ids, overlap_alpha):
        t = slot0 + slot * gap
        impulses.append(Impulse(t, ent, 3.2, "overlap"))
        overlap_events.append({
            "t": t,
            "entity": int(ent),
            "dose": 3.2,
            "tag": "overlap",
            "overlap": float(overlap[target, ent]),
            "alpha": float(alp),
        })
        slot += 1
    iso_events = []
    for ent in iso_ids:
        t = slot0 + slot * gap
        impulses.append(Impulse(t, ent, 3.6, "isolated"))
        iso_events.append({"t": t, "entity": int(ent), "dose": 3.6, "tag": "isolated"})
        slot += 1
    conn_events = []
    for ent in conn_ids:
        t = slot0 + slot * gap
        impulses.append(Impulse(t, ent, 3.6, "connected"))
        conn_events.append({"t": t, "entity": int(ent), "dose": 3.6, "tag": "connected"})
        slot += 1

    shocks = rng.normal(0.0, 0.12, size=(t_len, n_ent, d_b))
    # Dim 0 carries the intervention. Other dims are idiosyncratic.
    x_member = rng.normal(0.0, 1.0, size=(t_len, n_mem, d_x))
    member_of = np.repeat(np.arange(n_ent), members_per)
    # Entity-level covariate drive, shared by its members then noised.
    x_ent = rng.normal(0.0, 1.0, size=(t_len, n_ent, d_b))
    for e in range(n_ent):
        x_member[:, member_of == e, 0] += 0.3 * x_ent[:, e, 0][:, None]
    x_mean = x_ent
    beta = np.array([0.15, -0.05, 0.04])
    y_noise = rng.normal(0.0, 0.28, size=(t_len, n_mem))

    raw_block = simulate_states(shocks, x_mean, impulses, mediator, dose_source, None)
    raw_y = outcomes_from_block(
        raw_block, x_member, member_of, target, mediator, kernel,
        alpha, 0.9, beta, y_noise,
    )
    block, b_mu, b_sd = _scale_train(raw_block, train_end, axis=0)
    # Scale each block dimension with the training window, already done
    # jointly across entities because axis 0 is time. b_mu has shape (E, d).
    y_train_members = member_of == target
    y_slice = raw_y[:train_end][:, y_train_members]
    y_mu = float(np.mean(y_slice))
    y_sd = float(np.std(y_slice))
    if y_sd < 1e-6:
        y_sd = 1.0
    outcome = (raw_y - y_mu) / y_sd
    covariates, _, _ = _scale_train(x_member, train_end, axis=0)

    driver = -np.ones(n_ent, dtype=int)
    driver[mediator] = dose_source

    # Calm days: no impulse within the kernel on that entity.
    ref: dict[int, np.ndarray] = {}
    impulse_times = {}
    for imp in impulses:
        impulse_times.setdefault(imp.entity, []).append(imp.t)
    for e in range(n_ent):
        bad = np.zeros(train_end, dtype=bool)
        for t0 in impulse_times.get(e, []):
            lo = max(0, t0 - 2)
            hi = min(train_end, t0 + 15)
            bad[lo:hi] = True
        if e == mediator:
            for t0 in impulse_times.get(dose_source, []):
                lo = max(0, t0 - 2)
                hi = min(train_end, t0 + 15)
                bad[lo:hi] = True
        idx_ok = np.where(~bad)[0]
        idx_ok = idx_ok[idx_ok >= 20]
        ref[e] = idx_ok

    mismatch = {"temporal_distance"} if memory == "long" else set()
    queries = {
        "dose": dose_events,
        "overlap": overlap_events,
        "isolated": iso_events,
        "connected": conn_events,
        "main": max(dose_events, key=lambda r: r["dose"]),
        "dose_source": dose_source,
    }
    return Panel(
        name=f"synthetic_{memory}",
        grounded=True,
        entity_names=names,
        block=block.astype(np.float64),
        outcome=outcome.astype(np.float64),
        covariates=covariates.astype(np.float64),
        member_of=member_of.astype(int),
        overlap=overlap,
        time_index=np.arange(t_len),
        target=target,
        mediator=mediator,
        driver_of=driver,
        peak_lag=peak,
        split_mode="member",
        train_end=int(train_end),
        reference_index=ref,
        notes=(
            f"Semi-synthetic cohort panel, kernel peaks at lag {peak}. "
            "Direct coefficients and an isolated-versus-connected contrast are planted. "
            "A mediator cohort is driven by the dose source at lag 4."
        ),
        assumption_mismatch=mismatch,
        impulses=impulses,
        raw_block=raw_block,
        raw_outcome=raw_y,
        shocks=shocks,
        x_mean=x_mean,
        kernel=kernel,
        alpha_direct=alpha,
        alpha_med=0.9,
        beta=beta,
        y_noise=y_noise,
        block_mean=np.asarray(b_mu),
        block_std=np.asarray(b_sd),
        y_mean=y_mu,
        y_std=y_sd,
        queries=queries,
    )


def oracle_outcome(panel: Panel, t_intervene: int, entity: int, value_raw: float) -> np.ndarray:
    """Factual shocks, one dim-0 replacement. Returned on the standardized Y scale."""
    if panel.shocks is None or panel.raw_block is None:
        raise RuntimeError("oracle_outcome is only defined for synthetic panels")
    block = simulate_states(
        panel.shocks,
        panel.x_mean,
        panel.impulses,
        panel.mediator,
        int(panel.queries["dose_source"]),
        intervene=(t_intervene, entity, value_raw),
    )
    raw_y = outcomes_from_block(
        block,
        # covariates argument is the original member design; rebuild from scaled
        # by inverting is unnecessary: outcomes_from_block expects the same
        # covariates the factual sim used. Store them on the panel via a closure
        # — we persist them as `_x_raw` below.
        panel._x_raw,  # type: ignore[attr-defined]
        panel.member_of,
        panel.target,
        panel.mediator,
        panel.kernel,
        panel.alpha_direct,
        panel.alpha_med,
        panel.beta,
        panel.y_noise,
    )
    return (raw_y - panel.y_mean) / panel.y_std


def attach_raw_covariates(panel: Panel, x_raw: np.ndarray) -> None:
    panel._x_raw = x_raw  # type: ignore[attr-defined]


def reference_value_raw(panel: Panel, entity: int) -> float:
    """Mean of raw dim 0 on calm training days. The primary baseline."""
    idx = panel.reference_index[entity]
    if panel.raw_block is None:
        raise RuntimeError("raw block missing")
    return float(np.mean(panel.raw_block[idx, entity, 0]))


def make_synthetic_panel_with_oracle_cache(memory: str = "long", seed: int = 2026) -> Panel:
    # Rebuild covariates in the same function so the oracle can invert the
    # intervention. make_synthetic_panel does not keep raw X. Recompute here
    # by duplicating the call path inside a local copy.
    return _make(memory, seed)


def _make(memory: str, seed: int) -> Panel:
    """Same panel as make_synthetic_panel, with raw covariates retained."""
    rng = np.random.default_rng(seed)
    peak = 10 if memory == "long" else 1
    kernel = _kernel(peak)
    names = [
        "target", "mediator", "dose_source",
        "ov_1", "ov_2", "ov_3", "ov_4",
        "iso_1", "iso_2", "iso_3", "iso_4",
        "conn_1", "conn_2", "conn_3", "conn_4",
    ]
    n_ent = len(names)
    idx = {n: i for i, n in enumerate(names)}
    target = idx["target"]
    mediator = idx["mediator"]
    dose_source = idx["dose_source"]
    members_per = 32
    d_b = 4
    d_x = 3
    n_mem = n_ent * members_per
    slot0, gap, n_slots = 36, 16, 17
    last_slot = slot0 + (n_slots - 1) * gap
    train_end = last_slot + peak + 16
    t_len = train_end + 48

    alpha = np.zeros(n_ent)
    alpha[dose_source] = 1.6
    overlap_ids = [idx[f"ov_{i}"] for i in range(1, 5)]
    overlap_alpha = np.array([0.45, 0.95, 1.45, 1.95])
    for i, a in zip(overlap_ids, overlap_alpha):
        alpha[i] = float(a)
    iso_ids = [idx[f"iso_{i}"] for i in range(1, 5)]
    conn_ids = [idx[f"conn_{i}"] for i in range(1, 5)]
    for i in conn_ids:
        alpha[i] = 2.0

    overlap = np.eye(n_ent)
    for i, a in zip(overlap_ids, overlap_alpha):
        overlap[target, i] = overlap[i, target] = float(a / overlap_alpha.max())
    overlap[target, dose_source] = overlap[dose_source, target] = 0.8
    for i in conn_ids:
        overlap[target, i] = overlap[i, target] = 0.9
    for i in iso_ids:
        overlap[target, i] = overlap[i, target] = 0.05
    overlap[target, mediator] = overlap[mediator, target] = 0.7

    impulses: list[Impulse] = []
    dose_events, overlap_events, iso_events, conn_events = [], [], [], []
    slot = 0
    for dose in (1.5, 3.0, 4.5, 6.0, 7.5):
        t = slot0 + slot * gap
        impulses.append(Impulse(t, dose_source, float(dose), "dose"))
        dose_events.append({"t": t, "entity": dose_source, "dose": float(dose), "tag": "dose"})
        slot += 1
    for ent, alp in zip(overlap_ids, overlap_alpha):
        t = slot0 + slot * gap
        impulses.append(Impulse(t, int(ent), 3.2, "overlap"))
        overlap_events.append({
            "t": t, "entity": int(ent), "dose": 3.2, "tag": "overlap",
            "overlap": float(overlap[target, ent]), "alpha": float(alp),
        })
        slot += 1
    for ent in iso_ids:
        t = slot0 + slot * gap
        impulses.append(Impulse(t, int(ent), 3.6, "isolated"))
        iso_events.append({"t": t, "entity": int(ent), "dose": 3.6, "tag": "isolated"})
        slot += 1
    for ent in conn_ids:
        t = slot0 + slot * gap
        impulses.append(Impulse(t, int(ent), 3.6, "connected"))
        conn_events.append({"t": t, "entity": int(ent), "dose": 3.6, "tag": "connected"})
        slot += 1

    shocks = rng.normal(0.0, 0.12, size=(t_len, n_ent, d_b))
    x_member = rng.normal(0.0, 1.0, size=(t_len, n_mem, d_x))
    member_of = np.repeat(np.arange(n_ent), members_per)
    x_ent = rng.normal(0.0, 1.0, size=(t_len, n_ent, d_b))
    for e in range(n_ent):
        x_member[:, member_of == e, 0] += 0.3 * x_ent[:, e, 0][:, None]
    beta = np.array([0.15, -0.05, 0.04])
    y_noise = rng.normal(0.0, 0.28, size=(t_len, n_mem))
    raw_block = simulate_states(shocks, x_ent, impulses, mediator, dose_source, None)
    raw_y = outcomes_from_block(
        raw_block, x_member, member_of, target, mediator, kernel, alpha, 0.9, beta, y_noise,
    )
    block, b_mu, b_sd = _scale_train(raw_block, train_end, axis=0)
    y_mu = float(np.mean(raw_y[:train_end][:, member_of == target]))
    y_sd = float(np.std(raw_y[:train_end][:, member_of == target]))
    y_sd = y_sd if y_sd > 1e-6 else 1.0
    outcome = (raw_y - y_mu) / y_sd
    covariates, _, _ = _scale_train(x_member, train_end, axis=0)
    driver = -np.ones(n_ent, dtype=int)
    driver[mediator] = dose_source
    ref: dict[int, np.ndarray] = {}
    by_ent: dict[int, list[int]] = {}
    for imp in impulses:
        by_ent.setdefault(imp.entity, []).append(imp.t)
    for e in range(n_ent):
        bad = np.zeros(train_end, dtype=bool)
        watched = list(by_ent.get(e, []))
        if e == mediator:
            watched += by_ent.get(dose_source, [])
        for t0 in watched:
            bad[max(0, t0 - 2) : min(train_end, t0 + 15)] = True
        ok = np.where(~bad)[0]
        ref[e] = ok[ok >= 20]
    panel = Panel(
        name=f"synthetic_{memory}",
        grounded=True,
        entity_names=names,
        block=block.astype(np.float64),
        outcome=outcome.astype(np.float64),
        covariates=covariates.astype(np.float64),
        member_of=member_of.astype(int),
        overlap=overlap,
        time_index=np.arange(t_len),
        target=target,
        mediator=mediator,
        driver_of=driver,
        peak_lag=peak,
        split_mode="member",
        train_end=int(train_end),
        reference_index=ref,
        notes=(
            f"Semi-synthetic cohort panel with a lag-{peak} kernel, a mediator "
            "driven by the dose source, planted overlap, and an isolated group."
        ),
        assumption_mismatch={"temporal_distance"} if memory == "long" else set(),
        impulses=impulses,
        raw_block=raw_block,
        raw_outcome=raw_y,
        shocks=shocks,
        x_mean=x_ent,
        kernel=kernel,
        alpha_direct=alpha,
        alpha_med=0.9,
        beta=beta,
        y_noise=y_noise,
        block_mean=np.asarray(b_mu, dtype=float),
        block_std=np.asarray(b_sd, dtype=float),
        y_mean=y_mu,
        y_std=y_sd,
        queries={
            "dose": dose_events,
            "overlap": overlap_events,
            "isolated": iso_events,
            "connected": conn_events,
            "main": max(dose_events, key=lambda r: r["dose"]),
            "dose_source": dose_source,
        },
    )
    ent_x = np.zeros((t_len, n_ent, d_x), dtype=np.float64)
    for e in range(n_ent):
        ent_x[:, e] = covariates[:, member_of == e].mean(axis=1)
    panel.entity_covariates = ent_x
    panel._x_raw = x_member  # type: ignore[attr-defined]
    return panel


# The panel used by the pipeline. Rebinds the earlier helper of the same name.
make_synthetic_panel = _make
