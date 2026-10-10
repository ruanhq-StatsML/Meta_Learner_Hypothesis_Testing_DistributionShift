"""Run the FSDS pseudo-outcome horse-race and write figures plus LaTeX tables.

Scorecard experiment: fixed linear scores, reward = negative batch MSE,
batch size 200, pools with at least 10,000 rows or synthetic draws of that
size. The penalty is eta * (reference reward sd) * PO-risk share, with
eta in {0.5, 1, 2}, plus a fixed penalty of 5 for scale comparison.

Policy experiment: four linear policies, horizon 4, 200 episodes a round.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from fsds_policy_lib import build_pack, oracle_share_path, replay_policy
from fsds_porisk_lib import replay, share_path, stream

OUT = os.path.join(os.path.dirname(__file__), "results", "fsds_porisk_poc")
SEEDS = (20261018, 20261019, 20261020, 20261021)
FAMILIES = ("relocate", "flip", "die", "nonlin", "noise")
KINDS = [f"{path}_{fam}" for fam in FAMILIES for path in ("gradual", "drastic")]

SCORE_POLICIES = [
    {"name": "UCB", "rule": "ucb", "eta": 0.0, "mode": "abs", "eps": 0.0},
    {"name": "DUCB", "rule": "ducb", "eta": 0.0, "mode": "abs", "eps": 0.0},
    {"name": "TS", "rule": "ts", "eta": 0.0, "mode": "abs", "eps": 0.0},
    {"name": "EG05", "rule": "eps", "eta": 0.0, "mode": "abs", "eps": 0.05},
    {"name": "EG10", "rule": "eps", "eta": 0.0, "mode": "abs", "eps": 0.10},
    {"name": "EG15", "rule": "eps", "eta": 0.0, "mode": "abs", "eps": 0.15},
    {"name": "UCB-H0.5", "rule": "ucb", "eta": 0.5, "mode": "scale", "eps": 0.0},
    {"name": "UCB-H1", "rule": "ucb", "eta": 1.0, "mode": "scale", "eps": 0.0},
    {"name": "UCB-H2", "rule": "ucb", "eta": 2.0, "mode": "scale", "eps": 0.0},
    {"name": "TS-H0.5", "rule": "ts", "eta": 0.5, "mode": "scale", "eps": 0.0},
    {"name": "TS-H1", "rule": "ts", "eta": 1.0, "mode": "scale", "eps": 0.0},
    {"name": "TS-H2", "rule": "ts", "eta": 2.0, "mode": "scale", "eps": 0.0},
    {"name": "UCB-L5", "rule": "ucb", "eta": 5.0, "mode": "abs", "eps": 0.0},
    {"name": "TS-L5", "rule": "ts", "eta": 5.0, "mode": "abs", "eps": 0.0},
    {"name": "EG10-H1", "rule": "eps", "eta": 1.0, "mode": "scale", "eps": 0.10},
    {"name": "DUCB-H1", "rule": "ducb", "eta": 1.0, "mode": "scale", "eps": 0.0},
    {"name": "DUCB-H2", "rule": "ducb", "eta": 2.0, "mode": "scale", "eps": 0.0},
]

POLICY_POLICIES = [
    {"name": "UCB", "rule": "ucb", "eta": 0.0, "mode": "abs", "eps": 0.0, "monitor": "oracle"},
    {"name": "DUCB", "rule": "ducb", "eta": 0.0, "mode": "abs", "eps": 0.0, "monitor": "oracle"},
    {"name": "TS", "rule": "ts", "eta": 0.0, "mode": "abs", "eps": 0.0, "monitor": "oracle"},
    {"name": "EG05", "rule": "eps", "eta": 0.0, "mode": "abs", "eps": 0.05, "monitor": "oracle"},
    {"name": "EG10", "rule": "eps", "eta": 0.0, "mode": "abs", "eps": 0.10, "monitor": "oracle"},
    {"name": "EG15", "rule": "eps", "eta": 0.0, "mode": "abs", "eps": 0.15, "monitor": "oracle"},
    {"name": "UCB-H1", "rule": "ucb", "eta": 1.0, "mode": "scale", "eps": 0.0, "monitor": "oracle"},
    {"name": "UCB-H2", "rule": "ucb", "eta": 2.0, "mode": "scale", "eps": 0.0, "monitor": "oracle"},
    {"name": "TS-H1", "rule": "ts", "eta": 1.0, "mode": "scale", "eps": 0.0, "monitor": "oracle"},
    {"name": "TS-H2", "rule": "ts", "eta": 2.0, "mode": "scale", "eps": 0.0, "monitor": "oracle"},
    {"name": "EG10-H1", "rule": "eps", "eta": 1.0, "mode": "scale", "eps": 0.10, "monitor": "oracle"},
    {"name": "UCB-H1-on", "rule": "ucb", "eta": 1.0, "mode": "scale", "eps": 0.0, "monitor": "onpolicy"},
    {"name": "TS-H1-on", "rule": "ts", "eta": 1.0, "mode": "scale", "eps": 0.0, "monitor": "onpolicy"},
    {"name": "UCB-H1-fit", "rule": "ucb", "eta": 1.0, "mode": "scale", "eps": 0.0, "monitor": "model"},
    {"name": "TS-H1-fit", "rule": "ts", "eta": 1.0, "mode": "scale", "eps": 0.0, "monitor": "model"},
]

CURVE_POLICIES = ("UCB", "DUCB", "TS", "EG10", "UCB-H1", "TS-H1", "UCB-H2", "EG10-H1")
COLORS = {
    "UCB": "#4C78A8",
    "DUCB": "#72B7B2",
    "TS": "#F58518",
    "EG05": "#E45756",
    "EG10": "#E45756",
    "EG15": "#FF9D98",
    "UCB-H0.5": "#B5E2B0",
    "UCB-H1": "#54A24B",
    "UCB-H2": "#1F7A1F",
    "TS-H0.5": "#D7B5D8",
    "TS-H1": "#B279A2",
    "TS-H2": "#6B3F69",
    "UCB-L5": "#9D755D",
    "TS-L5": "#BAB0AC",
    "EG10-H1": "#F2CF5B",
    "UCB-H1-on": "#88D27A",
    "TS-H1-on": "#C49AC4",
    "UCB-H1-fit": "#2E5A27",
    "TS-H1-fit": "#4C2A4C",
}


POOLS = {}


def _init_worker(smoke=False):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    global POOLS
    POOLS = load_pools(smoke=smoke)


def _standardize(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd


def load_pools(smoke=False):
    rng = np.random.default_rng(0)
    pools = {
        "gaussian30": _standardize(rng.normal(size=(20000, 30))),
        "gaussian12": _standardize(rng.normal(size=(20000, 12))),
        "student_t5": _standardize(rng.standard_t(5, size=(20000, 30))),
    }
    d = 30
    cov = 0.6 ** np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
    pools["ar1_gaussian"] = _standardize(rng.multivariate_normal(np.zeros(d), cov, size=20000))
    if smoke:
        return {k: v for k, v in pools.items() if k == "gaussian30"}
    from sklearn.datasets import fetch_california_housing, fetch_covtype

    cal = fetch_california_housing(data_home="/tmp/skdata").data
    pools["california"] = _standardize(cal)
    covtype = fetch_covtype(data_home="/tmp/skdata").data
    take = rng.choice(len(covtype), size=30000, replace=False)
    pools["covertype"] = _standardize(covtype[take])
    return pools


def score_job(payload):
    name, kind, seed = payload
    X = POOLS[name]
    y, preds, post, alpha, arms = stream(X, kind, seed)
    wpath, sigma = share_path(y, preds, seed)
    detail = []
    curves = []
    shares = []
    for t, w in wpath.items():
        shares.append((t, int(post[t]), w))
    for i, policy in enumerate(SCORE_POLICIES):
        early, ramp, post_ex, hits, path, lam = replay(
            y, preds, post, alpha, wpath, sigma, policy, seed + 1000 + i
        )
        detail.append(
            {
                "dataset": name,
                "scenario": kind,
                "seed": seed,
                "policy": policy["name"],
                "early": early,
                "ramp": ramp,
                "post": post_ex,
                "hit_early": hits[0],
                "hit_ramp": hits[1],
                "hit_post": hits[2],
                "lam": lam,
                "sigma": sigma,
            }
        )
        if policy["name"] in CURVE_POLICIES:
            cum = np.cumsum(path)
            for t in range(len(path)):
                curves.append((policy["name"], t, float(path[t]), float(cum[t]), int(post[t])))
    return detail, curves, shares, arms


def policy_job(payload):
    name, kind, seed = payload
    X = POOLS[name]
    pack = build_pack(X, seed, kind)
    path_w, sigma = oracle_share_path(pack, seed)
    detail = []
    curves = []
    shares = []
    phase = pack["phase"]
    for t, w in path_w.items():
        shares.append((t, int(phase[t]), w))
    for i, policy in enumerate(POLICY_POLICIES):
        if policy["monitor"] == "oracle":
            totals, hits, path, lam = replay_policy(
                pack, seed + 3000 + i, policy, oracle_path=path_w, sigma=sigma
            )
        else:
            totals, hits, path, lam = replay_policy(pack, seed + 3000 + i, policy, sigma=sigma)
        detail.append(
            {
                "dataset": name,
                "scenario": kind,
                "seed": seed,
                "policy": policy["name"],
                "monitor": policy["monitor"],
                "phase0": float(totals[0]),
                "phase1": float(totals[1]),
                "phase2": float(totals[2]),
                "hit0": float(hits[0]),
                "hit1": float(hits[1]),
                "hit2": float(hits[2]),
                "lam": lam,
                "sigma": sigma,
                "n0": int(np.sum(phase == 0)),
                "n1": int(np.sum(phase == 1)),
                "n2": int(np.sum(phase == 2)),
            }
        )
        if policy["name"] in ("UCB", "DUCB", "TS", "EG10", "UCB-H1", "TS-H1", "UCB-H1-on", "UCB-H1-fit"):
            cum = np.cumsum(path)
            for t in range(len(path)):
                curves.append((policy["name"], t, float(path[t]), float(cum[t]), int(phase[t])))
    return detail, curves, shares


def _write_score_csv(details, curves, share_rows, path):
    import csv

    with open(os.path.join(path, "scorecard_detail.csv"), "w", newline="") as f:
        fields = [
            "dataset", "scenario", "seed", "policy",
            "early", "ramp", "post", "hit_early", "hit_ramp", "hit_post", "lam", "sigma",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in details:
            w.writerow(row)
    with open(os.path.join(path, "scorecard_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "scenario", "seed", "policy", "t", "excess", "cum", "post"])
        for ds, kind, seed, rows in curves:
            for policy, t, excess, cum, post in rows:
                w.writerow([ds, kind, seed, policy, t, excess, cum, post])
    with open(os.path.join(path, "scorecard_shares.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "scenario", "seed", "t", "post", "arm0", "arm1", "arm2", "arm3", "name0", "name1", "name2", "name3"])
        for ds, kind, seed, rows, arms in share_rows:
            for t, post, weights in rows:
                w.writerow([ds, kind, seed, t, post, *weights.tolist(), *arms])


def _write_policy_csv(details, curves, share_rows, path):
    import csv

    fields = [
        "dataset", "scenario", "seed", "policy", "monitor",
        "phase0", "phase1", "phase2", "hit0", "hit1", "hit2",
        "lam", "sigma", "n0", "n1", "n2",
    ]
    with open(os.path.join(path, "policy_detail.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in details:
            w.writerow(row)
    with open(os.path.join(path, "policy_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "scenario", "seed", "policy", "t", "excess", "cum", "phase"])
        for ds, kind, seed, rows in curves:
            for policy, t, excess, cum, phase in rows:
                w.writerow([ds, kind, seed, policy, t, excess, cum, phase])
    with open(os.path.join(path, "policy_shares.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "scenario", "seed", "t", "phase", "block0", "block1", "block2", "zero"])
        for ds, kind, seed, rows in share_rows:
            for t, phase, weights in rows:
                w.writerow([ds, kind, seed, t, phase, *weights.tolist()])


def _summarize_score(details):
    """Nested means: dataset, scenario, policy -> mean and sd over seeds."""
    bucket = {}
    for row in details:
        key = (row["dataset"], row["scenario"], row["policy"])
        bucket.setdefault(key, []).append(row)
    out = {}
    for key, rows in bucket.items():
        def col(name, rows=rows):
            arr = np.array([r[name] for r in rows], dtype=float)
            return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)

        out[key] = {
            "early": col("early"),
            "ramp": col("ramp"),
            "post": col("post"),
            "hit_early": col("hit_early"),
            "hit_ramp": col("hit_ramp"),
            "hit_post": col("hit_post"),
            "lam": col("lam"),
            "sigma": col("sigma"),
        }
    return out


def run(smoke=False, workers=4, part="all"):
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    pools = load_pools(smoke=smoke)
    seeds = SEEDS[:1] if smoke else SEEDS
    kinds = ["gradual_relocate", "gradual_flip"] if smoke else KINDS
    print(f"pools={list(pools)} kinds={len(kinds)} seeds={list(seeds)} part={part}", flush=True)
    if part in ("all", "score"):
        jobs = [(name, kind, seed) for name in pools for kind in kinds for seed in seeds]
        details, curves, share_rows = [], [], []
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(smoke,)) as ex:
            for i, (detail, curve, shares, arms) in enumerate(ex.map(score_job, jobs, chunksize=1), start=1):
                name = detail[0]["dataset"]
                kind = detail[0]["scenario"]
                seed = detail[0]["seed"]
                details.extend(detail)
                curves.append((name, kind, seed, curve))
                share_rows.append((name, kind, seed, shares, arms))
                if i % 8 == 0 or i == len(jobs):
                    print(f"  scorecard {i}/{len(jobs)} {name} {kind}", flush=True)
        _write_score_csv(details, curves, share_rows, OUT)
    if part in ("all", "policy"):
        pjobs = [(name, kind, seed) for name in pools for kind in ("single", "double") for seed in seeds]
        pdetails, pcurves, pshares = [], [], []
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(smoke,)) as ex:
            for i, (detail, curve, shares) in enumerate(ex.map(policy_job, pjobs, chunksize=1), start=1):
                pdetails.extend(detail)
                pcurves.append((detail[0]["dataset"], detail[0]["scenario"], detail[0]["seed"], curve))
                pshares.append((detail[0]["dataset"], detail[0]["scenario"], detail[0]["seed"], shares))
                print(f"  policy {i}/{len(pjobs)} {detail[0]['dataset']} {detail[0]['scenario']}", flush=True)
        _write_policy_csv(pdetails, pcurves, pshares, OUT)
    print("csv written", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--part", choices=("all", "score", "policy"), default="all")
    args = parser.parse_args()
    run(smoke=args.smoke, workers=args.workers, part=args.part)
