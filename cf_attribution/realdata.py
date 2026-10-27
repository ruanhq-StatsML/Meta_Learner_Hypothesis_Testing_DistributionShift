"""Real panels: Beijing multi-site air quality, and US state COVID incidence.

Both are observational. Relative-ranking numbers on these panels are
assumption checks. They are not a counterfactual ground truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cf_attribution.dgp import Panel

BEIJING_COORDS = {
    "Aotizhongxin": (39.982, 116.397),
    "Changping": (40.217, 116.230),
    "Dingling": (40.292, 116.220),
    "Dongsi": (39.929, 116.417),
    "Guanyuan": (39.929, 116.339),
    "Gucheng": (39.914, 116.184),
    "Huairou": (40.328, 116.628),
    "Nongzhanguan": (39.937, 116.461),
    "Shunyi": (40.127, 116.655),
    "Tiantan": (39.886, 116.407),
    "Wanliu": (39.987, 116.287),
    "Wanshouxigong": (39.878, 116.352),
}

# Census 2020 populations, used only to put incidence on a per-100k scale.
POPULATION = {
    "Alabama": 5024279, "Alaska": 733391, "Arizona": 7151502, "Arkansas": 3011524,
    "California": 39538223, "Colorado": 5773714, "Connecticut": 3605944,
    "Delaware": 989948, "District of Columbia": 689545, "Florida": 21538187,
    "Georgia": 10711908, "Hawaii": 1455271, "Idaho": 1839106, "Illinois": 12812508,
    "Indiana": 6785528, "Iowa": 3190369, "Kansas": 2937880, "Kentucky": 4505836,
    "Louisiana": 4657757, "Maine": 1362359, "Maryland": 6177224,
    "Massachusetts": 7029917, "Michigan": 10077331, "Minnesota": 5706494,
    "Mississippi": 2961279, "Missouri": 6154913, "Montana": 1084225,
    "Nebraska": 1961504, "Nevada": 3104614, "New Hampshire": 1377529,
    "New Jersey": 9288994, "New Mexico": 2117522, "New York": 20201249,
    "North Carolina": 10439388, "North Dakota": 779094, "Ohio": 11799448,
    "Oklahoma": 3959353, "Oregon": 4237256, "Pennsylvania": 13002700,
    "Rhode Island": 1097379, "South Carolina": 5118425, "South Dakota": 886667,
    "Tennessee": 6910840, "Texas": 29145505, "Utah": 3271616, "Vermont": 643077,
    "Virginia": 8631393, "Washington": 7705281, "West Virginia": 1793716,
    "Wisconsin": 5893718, "Wyoming": 576851,
}

REGIONS = {
    "Northeast": ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont", "New Jersey", "New York", "Pennsylvania"],
    "Midwest": ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin", "Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska", "North Dakota", "South Dakota"],
    "South": ["Delaware", "Florida", "Georgia", "Maryland", "North Carolina", "South Carolina", "Virginia", "West Virginia", "Alabama", "Kentucky", "Mississippi", "Tennessee", "Arkansas", "Louisiana", "Oklahoma", "Texas", "District of Columbia"],
    "West": ["Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming", "Alaska", "California", "Hawaii", "Oregon", "Washington"],
}

REGION_COORDS = {
    "Northeast": (42.7, -73.5),
    "Midwest": (41.8, -89.0),
    "South": (33.5, -86.5),
    "West": (39.5, -116.0),
}


def _candidates() -> list[Path]:
    return [Path("/tmp/panel_data"), Path("data/cache"), Path("/workspace/data/cache")]


def _find(name: str) -> Path | None:
    for root in _candidates():
        path = root / name
        if path.exists():
            return path
    return None


def _haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = np.radians([a[0], b[0], a[1], b[1]])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * 6371.0 * np.arcsin(np.sqrt(h)))


def _overlap_from_coords(names: list[str], coords: dict[str, tuple[float, float]], scale_km: float) -> np.ndarray:
    n = len(names)
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            dist = _haversine(coords[names[i]], coords[names[j]])
            val = float(np.exp(-dist / scale_km))
            out[i, j] = out[j, i] = val
    return out


def _standardize_time(x: np.ndarray, train_end: int) -> np.ndarray:
    mu = np.nanmean(x[:train_end], axis=0, keepdims=True)
    sd = np.nanstd(x[:train_end], axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (x - mu) / sd


def _peaks(z: np.ndarray, start: int, end: int, gap: int, k: int, z_min: float) -> list[int]:
    idx = [t for t in range(start, end) if np.isfinite(z[t]) and z[t] >= z_min]
    idx.sort(key=lambda t: -z[t])
    chosen: list[int] = []
    for t in idx:
        if all(abs(t - c) >= gap for c in chosen):
            chosen.append(t)
        if len(chosen) >= k:
            break
    if len(chosen) < min(k, 4):
        # Relax the height cut, keep the spacing. Still pre-registered:
        # we never look at the attribution score while choosing days.
        idx = [t for t in range(start, end) if np.isfinite(z[t])]
        idx.sort(key=lambda t: -z[t])
        chosen = []
        for t in idx:
            if all(abs(t - c) >= gap for c in chosen):
                chosen.append(t)
            if len(chosen) >= k:
                break
    return sorted(chosen)


def _zscore(series: np.ndarray, train_end: int) -> np.ndarray:
    mu = np.nanmean(series[:train_end])
    sd = np.nanstd(series[:train_end])
    if not np.isfinite(sd) or sd < 1e-6:
        sd = 1.0
    return (series - mu) / sd


def _reference_index(z: np.ndarray, train_end: int) -> np.ndarray:
    idx = np.where((np.arange(train_end) >= 20) & (np.abs(z[:train_end]) < 0.75))[0]
    if len(idx) < 15:
        idx = np.arange(20, train_end)
    return idx


def _queries_from_z(
    block: np.ndarray,
    names: list[str],
    target: int,
    dose_source: int,
    mediator: int,
    overlap: np.ndarray,
    train_end: int,
    peak_lag: int,
) -> dict:
    t_len = len(block)
    end = t_len - peak_lag - 1
    dose_z = _zscore(block[:, dose_source, 0], train_end)
    dose_days = _peaks(dose_z, train_end, end, gap=18, k=5, z_min=1.0)
    dose_events = [
        {"t": int(t), "entity": dose_source, "dose": float(dose_z[t]), "tag": "dose"}
        for t in dose_days
    ]
    remaining = [i for i in range(len(names)) if i not in {target, dose_source, mediator}]
    remaining.sort(key=lambda i: overlap[target, i])
    # Four overlap rungs, spread across the similarity range.
    if len(remaining) >= 4:
        picks = [remaining[int(round(x))] for x in np.linspace(0, len(remaining) - 1, 4)]
        picks = list(dict.fromkeys(picks))
    else:
        picks = remaining
    taken = set(dose_days)
    overlap_events = []
    for ent in picks:
        z = _zscore(block[:, ent, 0], train_end)
        days = _peaks(z, train_end, end, gap=18, k=1, z_min=1.0)
        for t in days:
            if all(abs(t - c) >= 18 for c in taken):
                overlap_events.append({
                    "t": int(t),
                    "entity": int(ent),
                    "dose": float(z[t]),
                    "tag": "overlap",
                    "overlap": float(overlap[target, ent]),
                })
                taken.add(t)
    connected = remaining[-1] if remaining else None
    isolated = remaining[0] if remaining else None
    iso_events, conn_events = [], []
    for ent, bucket, tag in (
        (isolated, iso_events, "isolated"),
        (connected, conn_events, "connected"),
    ):
        if ent is None:
            continue
        z = _zscore(block[:, ent, 0], train_end)
        for t in _peaks(z, train_end, end, gap=18, k=4, z_min=0.8):
            if all(abs(t - c) >= 18 for c in taken):
                bucket.append({"t": int(t), "entity": int(ent), "dose": float(z[t]), "tag": tag})
                taken.add(t)
    main = max(dose_events, key=lambda r: r["dose"]) if dose_events else None
    return {
        "dose": dose_events,
        "overlap": overlap_events,
        "isolated": iso_events,
        "connected": conn_events,
        "main": main,
        "dose_source": dose_source,
    }


def load_beijing(peak_lag: int = 10) -> Panel | None:
    root = None
    for base in _candidates():
        hits = list(base.rglob("PRSA_Data_*Dongsi*.csv"))
        if hits:
            root = hits[0].parent
            break
    if root is None:
        return None
    frames = []
    for name in BEIJING_COORDS:
        path = next(root.glob(f"PRSA_Data_{name}_*.csv"))
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[(raw["time"] >= "2014-03-01") & (raw["time"] < "2017-02-01")].copy()
    raw["date"] = raw["time"].dt.floor("D")
    names = list(BEIJING_COORDS)
    # Daily block: four pollutants. Hours are the target's within-day distribution.
    pollutants = ["PM2.5", "PM10", "NO2", "SO2"]
    daily = (
        raw.groupby(["station", "date"], as_index=False)[pollutants + ["TEMP", "DEWP", "WSPM", "RAIN"]]
        .mean()
    )
    dates = sorted(daily["date"].unique())
    # Balanced days: every station has a finite PM2.5 daily mean.
    keep = []
    for day in dates:
        sub = daily[daily["date"] == day]
        if sub["station"].nunique() == len(names) and sub["PM2.5"].notna().all():
            keep.append(day)
    dates = keep
    t_len = len(dates)
    date_index = {d: i for i, d in enumerate(dates)}
    block = np.full((t_len, len(names), 4), np.nan)
    ent_x = np.zeros((t_len, len(names), 4))
    for s_i, name in enumerate(names):
        sub = daily[daily["station"] == name]
        for _, row in sub.iterrows():
            if row["date"] not in date_index:
                continue
            t = date_index[row["date"]]
            block[t, s_i] = [row[c] if np.isfinite(row[c]) else np.nan for c in pollutants]
            ent_x[t, s_i] = [
                row["TEMP"] if np.isfinite(row["TEMP"]) else 0.0,
                row["DEWP"] if np.isfinite(row["DEWP"]) else 0.0,
                row["WSPM"] if np.isfinite(row["WSPM"]) else 0.0,
                row["RAIN"] if np.isfinite(row["RAIN"]) else 0.0,
            ]
    # Interpolate rare pollutant holes along time, per station.
    for s_i in range(len(names)):
        for d in range(4):
            series = pd.Series(block[:, s_i, d]).interpolate(limit=2).bfill().ffill()
            block[:, s_i, d] = series.to_numpy()
    target = names.index("Dongsi")
    dose_source = names.index("Gucheng")
    overlap = _overlap_from_coords(names, BEIJING_COORDS, scale_km=18.0)
    # Mediator: remaining station whose overlap with Dongsi is closest to the median.
    rest = [i for i in range(len(names)) if i not in {target, dose_source}]
    med_target = float(np.median(overlap[target, rest]))
    mediator = min(rest, key=lambda i: abs(overlap[target, i] - med_target))
    # 8 three-hour bins of Dongsi PM2.5 are the cohort distribution.
    hours = raw[raw["station"] == "Dongsi"].copy()
    hours["bin"] = hours["hour"] // 3
    bin_mean = hours.groupby(["date", "bin"], as_index=False)[["PM2.5", "TEMP", "DEWP", "WSPM"]].mean()
    n_bins = 8
    outcome_raw = np.full((t_len, n_bins), np.nan)
    cov = np.zeros((t_len, n_bins, 4))
    for _, row in bin_mean.iterrows():
        if row["date"] not in date_index:
            continue
        t = date_index[row["date"]]
        b = int(row["bin"])
        outcome_raw[t, b] = row["PM2.5"]
        cov[t, b] = [
            row["TEMP"] if np.isfinite(row["TEMP"]) else 0.0,
            row["DEWP"] if np.isfinite(row["DEWP"]) else 0.0,
            row["WSPM"] if np.isfinite(row["WSPM"]) else 0.0,
            np.sin(2 * np.pi * b / n_bins),
        ]
    for b in range(n_bins):
        outcome_raw[:, b] = pd.Series(outcome_raw[:, b]).interpolate(limit=2).bfill().ffill().to_numpy()
        for d in range(3):
            cov[:, b, d] = pd.Series(cov[:, b, d]).interpolate(limit=2).bfill().ffill().to_numpy()
    train_end = int(t_len * 0.65)
    block_s = _standardize_time(block, train_end)
    ent_s = _standardize_time(ent_x, train_end)
    y_mu = float(np.nanmean(outcome_raw[:train_end]))
    y_sd = float(np.nanstd(outcome_raw[:train_end]))
    y_sd = y_sd if y_sd > 1e-6 else 1.0
    outcome = (outcome_raw - y_mu) / y_sd
    cov_s = _standardize_time(cov.reshape(t_len, -1), train_end).reshape(t_len, n_bins, 4)
    driver = -np.ones(len(names), dtype=int)
    driver[mediator] = dose_source
    ref = {}
    for e in range(len(names)):
        ref[e] = _reference_index(_zscore(block_s[:, e, 0], train_end), train_end)
    queries = _queries_from_z(block_s, names, target, dose_source, mediator, overlap, train_end, peak_lag)
    return Panel(
        name="beijing_air",
        grounded=False,
        entity_names=names,
        block=block_s.astype(np.float64),
        outcome=outcome.astype(np.float64),
        covariates=cov_s.astype(np.float64),
        member_of=np.full(n_bins, target, dtype=int),
        overlap=overlap,
        time_index=np.arange(t_len),
        target=target,
        mediator=mediator,
        driver_of=driver,
        peak_lag=peak_lag,
        split_mode="time",
        train_end=train_end,
        reference_index=ref,
        notes=(
            "Beijing multi-site hourly PM2.5, March 2014–January 2017, collapsed to days. "
            f"Target Dongsi, dose source Gucheng, mediator {names[mediator]}. "
            "Within-day 3-hour bins are the cohort distribution. "
            "The primary query lag is 10 days, the same lag as the business question; "
            "regional pollution usually moves faster, and the protocol is allowed to say so."
        ),
        assumption_mismatch=set(),
        queries=queries,
        entity_covariates=ent_s.astype(np.float64),
        y_mean=y_mu,
        y_std=y_sd,
    )


def load_covid(peak_lag: int = 10) -> Panel | None:
    path = _find("us-states.csv")
    if path is None:
        nested = _find("covid/us-states.csv")
        path = nested
    if path is None:
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["state"].isin(POPULATION)].copy()
    df = df[(df["date"] >= "2020-03-15") & (df["date"] <= "2021-06-01")]
    df = df.sort_values(["state", "date"])
    df["new_cases"] = df.groupby("state")["cases"].diff().clip(lower=0)
    df["new_deaths"] = df.groupby("state")["deaths"].diff().clip(lower=0)
    df["inc"] = np.log1p(df["new_cases"] / df["state"].map(POPULATION) * 1e5)
    df["death_rate"] = np.log1p(df["new_deaths"] / df["state"].map(POPULATION) * 1e5)
    names = list(REGIONS)
    # State members of every region, so a query can use the target's states.
    members = []
    member_region = []
    for r_i, region in enumerate(names):
        for state in REGIONS[region]:
            members.append(state)
            member_region.append(r_i)
    dates = sorted(df["date"].unique())
    # Drop the first day of diffs (NaN).
    dates = [d for d in dates if df.loc[df["date"] == d, "inc"].notna().sum() > 40]
    t_len = len(dates)
    date_index = {d: i for i, d in enumerate(dates)}
    outcome_raw = np.zeros((t_len, len(members)))
    for m_i, state in enumerate(members):
        sub = df[df["state"] == state]
        for _, row in sub.iterrows():
            if row["date"] in date_index and np.isfinite(row["inc"]):
                outcome_raw[date_index[row["date"]], m_i] = row["inc"]
    block = np.zeros((t_len, len(names), 4))
    for r_i, region in enumerate(names):
        cols = [i for i, r in enumerate(member_region) if r == r_i]
        series = outcome_raw[:, cols]
        block[:, r_i, 0] = series.mean(axis=1)
        block[:, r_i, 1] = series.std(axis=1)
        # Deaths: region mean.
        death = np.zeros(t_len)
        states = REGIONS[region]
        for state in states:
            sub = df[df["state"] == state]
            for _, row in sub.iterrows():
                if row["date"] in date_index and np.isfinite(row["death_rate"]):
                    death[date_index[row["date"]]] += row["death_rate"] / len(states)
        block[:, r_i, 2] = death
        block[:, r_i, 3] = pd.Series(block[:, r_i, 0]).diff().fillna(0).to_numpy()
    dow = np.array([d.dayofweek for d in dates], dtype=float)
    trend = np.linspace(0, 1, t_len)
    ent_x = np.zeros((t_len, len(names), 3))
    ent_x[:, :, 0] = np.sin(2 * np.pi * dow / 7.0)[:, None]
    ent_x[:, :, 1] = np.cos(2 * np.pi * dow / 7.0)[:, None]
    ent_x[:, :, 2] = trend[:, None]
    cov = np.zeros((t_len, len(members), 3))
    for m in range(len(members)):
        cov[:, m] = ent_x[:, member_region[m]]
    target = names.index("South")
    dose_source = names.index("Northeast")
    mediator = names.index("Midwest")
    overlap = _overlap_from_coords(names, REGION_COORDS, scale_km=900.0)
    train_end = int(t_len * 0.60)
    block_s = _standardize_time(block, train_end)
    # Outcome scale is fit on the target region's members only.
    target_cols = [i for i, r in enumerate(member_region) if r == target]
    y_mu = float(np.mean(outcome_raw[:train_end][:, target_cols]))
    y_sd = float(np.std(outcome_raw[:train_end][:, target_cols]))
    y_sd = y_sd if y_sd > 1e-6 else 1.0
    outcome = (outcome_raw - y_mu) / y_sd
    cov_s = _standardize_time(cov.reshape(t_len, -1), train_end).reshape(cov.shape)
    ent_s = _standardize_time(ent_x, train_end)
    driver = -np.ones(len(names), dtype=int)
    driver[mediator] = dose_source
    ref = {e: _reference_index(_zscore(block_s[:, e, 0], train_end), train_end) for e in range(len(names))}
    queries = _queries_from_z(block_s, names, target, dose_source, mediator, overlap, train_end, peak_lag)
    # Keep only the target region's states as the evaluated cohort. Other
    # states remain available through the region block, which is already built.
    member_of = np.array([target if r == target else -1 for r in member_region], dtype=int)
    # Drop non-target members so the outcome matrix matches member_of == target.
    keep = np.array(target_cols)
    return Panel(
        name="us_covid_states",
        grounded=False,
        entity_names=names,
        block=block_s.astype(np.float64),
        outcome=outcome[:, keep].astype(np.float64),
        covariates=cov_s[:, keep].astype(np.float64),
        member_of=np.full(len(keep), target, dtype=int),
        overlap=overlap,
        time_index=np.arange(t_len),
        target=target,
        mediator=mediator,
        driver_of=driver,
        peak_lag=peak_lag,
        split_mode="time",
        train_end=train_end,
        reference_index=ref,
        notes=(
            "NYT US state COVID panel, 15 Mar 2020–1 Jun 2021. "
            "Cohorts are Census regions; the South's states are the outcome "
            "distribution; the Northeast's log incidence is the source block. "
            "The primary lag is 10 days. Incidence is log1p of new cases per 100k."
        ),
        assumption_mismatch=set(),
        queries=queries,
        entity_covariates=ent_s.astype(np.float64),
        y_mean=y_mu,
        y_std=y_sd,
    )
