"""Online Q-learning variants on the same scorecard horse-race.

The environment, batch size, and FSDS share are unchanged. Only the index
changes. Gradual relocation is the comparison that the tax already wins.
Gradual sign flip and noise growth are the two checks: the tax should stay
off on a flip, and a frozen Q should not be required when the best card
never moves.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from fsds_porisk_lib import replay, share_path, stream
from run_fsds_porisk_poc import SEEDS, _init_worker, load_pools

OUT = os.path.join(os.path.dirname(__file__), "results", "fsds_ql")
KINDS = ("gradual_relocate", "gradual_flip", "gradual_noise")

POLICIES = [
    {"name": "UCB", "rule": "ucb", "eta": 0.0, "mode": "abs", "eps": 0.0},
    {"name": "DUCB", "rule": "ducb", "eta": 0.0, "mode": "abs", "eps": 0.0},
    {"name": "UCB-H2", "rule": "ucb", "eta": 2.0, "mode": "scale", "eps": 0.0},
    {"name": "TS", "rule": "ts", "eta": 0.0, "mode": "abs", "eps": 0.0},
    {"name": "EG10", "rule": "eps", "eta": 0.0, "mode": "abs", "eps": 0.10},
    {"name": "QL05", "rule": "ql", "eta": 0.0, "mode": "abs", "eps": 0.0, "alpha": 0.05},
    {"name": "QL20", "rule": "ql", "eta": 0.0, "mode": "abs", "eps": 0.0, "alpha": 0.20},
    {"name": "QL20-H2", "rule": "ql", "eta": 2.0, "mode": "scale", "eps": 0.0, "alpha": 0.20},
    {"name": "QL-UCB", "rule": "qlucb", "eta": 0.0, "mode": "abs", "eps": 0.0, "alpha": 0.20},
    {"name": "QL-UCB-H2", "rule": "qlucb", "eta": 2.0, "mode": "scale", "eps": 0.0, "alpha": 0.20},
    {"name": "DQL", "rule": "dql", "eta": 0.0, "mode": "abs", "eps": 0.0, "alpha": 0.20},
    {"name": "DQL-H2", "rule": "dql", "eta": 2.0, "mode": "scale", "eps": 0.0, "alpha": 0.20},
    {"name": "SQL", "rule": "sql", "eta": 0.0, "mode": "abs", "eps": 0.0, "alpha": 0.20},
    {"name": "BOLT", "rule": "bolt", "eta": 0.0, "mode": "abs", "eps": 0.0, "alpha": 0.20},
    {"name": "QWIN", "rule": "qwin", "eta": 0.0, "mode": "abs", "eps": 0.0, "window": 8, "bonus": False},
    {"name": "QWIN-UCB-H2", "rule": "qwin", "eta": 2.0, "mode": "scale", "eps": 0.0, "window": 8, "bonus": True},
]


def _job(payload):
    name, kind, seed = payload
    from run_fsds_porisk_poc import POOLS
    X = POOLS[name]
    y, preds, post, alpha, _arms = stream(X, kind, seed)
    wpath, sigma = share_path(y, preds, seed)
    rows = []
    for i, policy in enumerate(POLICIES):
        early, ramp, post_ex, hits, _path, lam = replay(
            y, preds, post, alpha, wpath, sigma, policy, seed + 1000 + i
        )
        rows.append({
            "dataset": name,
            "scenario": kind,
            "seed": seed,
            "policy": policy["name"],
            "early": early,
            "ramp": ramp,
            "post": post_ex,
            "hit_post": hits[2],
            "lam": lam,
            "sigma": sigma,
        })
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, policy), sub in df.groupby(["scenario", "policy"], sort=False):
        pool_means = []
        for _dataset, pool in sub.groupby("dataset"):
            pool_means.append(float(pool["post"].mean()))
        arr = np.asarray(pool_means, dtype=float)
        rows.append({
            "scenario": scenario,
            "policy": policy,
            "median_post": float(np.median(arr)),
            "mean_post": float(arr.mean()),
            "mean_hit": float(sub["hit_post"].mean()),
            "mean_early": float(sub["early"].mean()),
        })
    return pd.DataFrame(rows)


def pool_table(df: pd.DataFrame, scenario: str, policies: list[str]) -> pd.DataFrame:
    sub = df[df["scenario"] == scenario]
    rows = []
    for dataset, pool in sub.groupby("dataset"):
        rec = {"dataset": dataset}
        for name in policies:
            sl = pool[pool["policy"] == name]["post"]
            rec[name] = float(sl.mean())
            rec[name + "_sd"] = float(sl.std(ddof=1)) if len(sl) > 1 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


PRETTY = {
    "ar1_gaussian": "AR(1)",
    "california": "California",
    "covertype": "covertype",
    "gaussian12": "Gaussian-12",
    "gaussian30": "Gaussian-30",
    "student_t5": r"Student-$t_5$",
}


def _cell(summary: pd.DataFrame, scenario: str, policy: str, col: str) -> float:
    block = summary[(summary["scenario"] == scenario) & (summary["policy"] == policy)]
    return float(block[col].iloc[0])


def _gaps(detail: pd.DataFrame, scenario: str, winner: str, loser: str):
    sub = detail[detail["scenario"] == scenario]
    rows = []
    for dataset, group in sub.groupby("dataset"):
        left = group[group["policy"] == winner]["post"]
        right = group[group["policy"] == loser]["post"]
        gap = float(right.mean() - left.mean())
        msd = max(float(left.std(ddof=1)), float(right.std(ddof=1)))
        rows.append({
            "dataset": dataset,
            "gap": gap,
            "msd": msd,
            "clear": gap > msd,
            "winner_mean": float(left.mean()),
            "winner_sd": float(left.std(ddof=1)),
            "loser_mean": float(right.mean()),
            "loser_sd": float(right.std(ddof=1)),
        })
    return rows


def _n_clear(detail: pd.DataFrame, scenario: str, winner: str, loser: str) -> int:
    return sum(int(row["clear"]) for row in _gaps(detail, scenario, winner, loser))


def _inside(detail: pd.DataFrame, scenario: str, winner: str, loser: str):
    return [row for row in _gaps(detail, scenario, winner, loser) if not row["clear"]]


def _n_equal(detail: pd.DataFrame, scenario: str, left: str, right: str):
    sub = detail[detail["scenario"] == scenario]
    wide = sub.pivot_table(index=["dataset", "seed"], columns="policy", values="post")
    match = np.abs(wide[left] - wide[right]) < 1e-8
    full = []
    for dataset, flag in match.groupby(level=0):
        if bool(flag.all()):
            full.append(str(dataset))
    return int(match.sum()), int(len(match)), full


def _pretty_list(names: list[str]) -> str:
    labels = [PRETTY.get(name, name) for name in names]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + ", and " + labels[-1]


def _plot(summary: pd.DataFrame, path: str):
    order = [p["name"] for p in POLICIES]
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.6), sharex=True)
    titles = {
        "gradual_relocate": "Gradual relocation",
        "gradual_flip": "Gradual sign flip",
    }
    for ax, scenario in zip(axes, ("gradual_relocate", "gradual_flip")):
        sub = summary[summary["scenario"] == scenario].copy()
        sub["policy"] = pd.Categorical(sub["policy"], order, ordered=True)
        sub = sub.sort_values("policy")
        colors = ["#1F7A1F" if name == "UCB-H2" else "#4C78A8" for name in sub["policy"]]
        ax.bar(np.arange(len(sub)), sub["median_post"].to_numpy(), color=colors)
        ax.set_ylabel("median post-shift excess")
        ax.set_title(titles[scenario])
        ax.grid(True, axis="y", alpha=0.3)
    axes[-1].set_xticks(np.arange(len(order)))
    axes[-1].set_xticklabels(order, rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _pool_block(caption: str, pools: pd.DataFrame, focus: list[str]) -> list[str]:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(focus) + "}",
        r"\toprule",
        " & ".join(["pool"] + [s.replace("_", r"\_") for s in focus]) + r" \\",
        r"\midrule",
    ]
    for _, row in pools.iterrows():
        cells = [PRETTY.get(str(row["dataset"]), str(row["dataset"]).replace("_", r"\_"))]
        for name in focus:
            cells.append(f"{row[name]:.1f} ({row[name + '_sd']:.1f})")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return lines


def _tex(summary: pd.DataFrame, pools: pd.DataFrame, flip: pd.DataFrame) -> str:
    show = [
        "UCB", "DUCB", "UCB-H2", "TS", "QL20", "QL20-H2",
        "QL-UCB", "QL-UCB-H2", "DQL", "SQL", "BOLT", "QWIN-UCB-H2",
    ]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Median, across six pools, of the four-seed mean post-shift excess.}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(show) + "}",
        r"\toprule",
        " & ".join(["scenario"] + [s.replace("_", r"\_") for s in show]) + r" \\",
        r"\midrule",
    ]
    for scenario in summary["scenario"].drop_duplicates():
        cells = [scenario.replace("_", r"\_")]
        block = summary[summary["scenario"] == scenario]
        for name in show:
            val = float(block[block["policy"] == name]["median_post"].iloc[0])
            cells.append(f"{val:.1f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    focus = ["UCB", "UCB-H2", "QL20", "QL-UCB", "QL-UCB-H2", "QWIN-UCB-H2"]
    lines += _pool_block(
        "Gradual relocation. Mean post-shift excess over four seeds, "
        "seed standard deviation in parentheses.",
        pools,
        focus,
    )
    lines += _pool_block(
        "Gradual sign flip. Mean post-shift excess over four seeds, "
        "seed standard deviation in parentheses.",
        flip,
        ["UCB", "DUCB", "UCB-H2", "QL-UCB", "SQL"],
    )
    return "\n".join(lines)


def _note(summary: pd.DataFrame, detail: pd.DataFrame) -> str:
    def med(scenario, policy):
        return _cell(summary, scenario, policy, "median_post")

    def hit(scenario, policy):
        return _cell(summary, scenario, policy, "mean_hit")

    def early(scenario, policy):
        return _cell(summary, scenario, policy, "mean_early")

    def clear(scenario, winner, loser):
        n = _n_clear(detail, scenario, winner, loser)
        return n

    eq_n, eq_d, eq_pools = _n_equal(detail, "gradual_relocate", "QL-UCB-H2", "UCB-H2")
    win_n, win_d, win_pools = _n_equal(detail, "gradual_relocate", "QWIN-UCB-H2", "UCB-H2")
    cal = _inside(detail, "gradual_flip", "QL-UCB", "DUCB")
    cal_bits = []
    for row in cal:
        cal_bits.append(
            f"{PRETTY.get(row['dataset'], row['dataset'])}, "
            f"where the bonus rule is {row['winner_mean']:.1f} "
            f"(sd {row['winner_sd']:.1f}) and discounted UCB is "
            f"{row['loser_mean']:.1f} (sd {row['loser_sd']:.1f})"
        )
    cal_text = "; ".join(cal_bits) if cal_bits else "none"
    edged = [row for row in _gaps(detail, "gradual_relocate", "QL-UCB-H2", "UCB-H2") if row["clear"]]
    edge_bits = []
    for row in edged:
        edge_bits.append(
            f"{PRETTY.get(row['dataset'], row['dataset'])} "
            f"({row['winner_mean']:.1f}, sd {row['winner_sd']:.1f}, against "
            f"{row['loser_mean']:.1f}, sd {row['loser_sd']:.1f})"
        )
    edge_text = "; ".join(edge_bits) if edge_bits else "no pool"
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs,graphicx,amsmath}}
\title{{Online Q-learning on the same gradual scorecard race}}
\author{{}}
\date{{}}
\begin{{document}}
\maketitle

\section{{What can differ from UCB}}

The arms, the batch of 200, and the pseudo-outcome share are the relocation MVP.
The reward is still negative batch mean squared error. In a stateless bandit,
Sarsa and Q-learning select the same arm, and a step size \(1/n\) is the sample
mean already inside UCB. The updates below are the ones that can actually move
the index: a constant step, two tables, a speedy extrapolation, a softmax, and
a window of the last eight rewards. Constant-step Q, double Q, the bonus,
and the window are also run with the FSDS tax at \(\eta=2\). Speedy Q and
Boltzmann are reported without that tax. The first four rounds still pull each
arm once. A table is
initialized at the first reward of that arm. A zero start would be optimistic,
because the rewards are negative.

Constant \(\alpha\) forgets. Greedy Q never refreshes an arm it has stopped
pulling, so a value frozen before the ramp cannot follow the fresh block.
A bonus, a softmax, or a sliding window is what lets the other arms move.
The tax is subtracted from every arm's index, not only from the arm just pulled.

\section{{Relocation stays with the tax}}

UCB at \(\eta=2\) remains the reference. Its median post-shift excess is
{med('gradual_relocate','UCB-H2'):.1f}, against {med('gradual_relocate','UCB'):.1f}
for plain UCB and {med('gradual_relocate','DUCB'):.1f} for discounted UCB.
Greedy Q at \(\alpha=0.2\) is {med('gradual_relocate','QL20'):.1f}. That gap
against UCB at \(\eta=2\) clears the larger seed standard deviation on
{clear('gradual_relocate','UCB-H2','QL20')}/6 pools. The same tax on the frozen
value gives {med('gradual_relocate','QL20-H2'):.1f}. Q plus a UCB bonus, without
the tax, is {med('gradual_relocate','QL-UCB'):.1f}, next to discounted UCB, and
it clears discounted UCB on {clear('gradual_relocate','QL-UCB','DUCB')}/6 pools.
The bonus with the tax is {med('gradual_relocate','QL-UCB-H2'):.1f}. Speedy Q is
the lowest untaxed rule at {med('gradual_relocate','SQL'):.1f}, and UCB at
\(\eta=2\) still clears it on {clear('gradual_relocate','UCB-H2','SQL')}/6 pools.
Double Q updates one table at random and finishes at
{med('gradual_relocate','DQL'):.1f}. Boltzmann pays during the stationary window
(early excess {early('gradual_relocate','BOLT'):.1f}, against
{early('gradual_relocate','UCB'):.1f} for UCB) and finishes at
{med('gradual_relocate','BOLT'):.1f}.

On {_pretty_list(eq_pools)}, all four seeds of Q-plus-bonus at \(\eta=2\) match
UCB at \(\eta=2\). The match is {eq_n} of {eq_d} seed-pool pairs, and the
eight-step window with the same bonus and tax matches on {win_n} of {win_d}
({_pretty_list(win_pools)}). Once the share is sharp, the tax is choosing the arm.
The taxed bonus rule undercuts UCB at \(\eta=2\) by more than the seed standard
deviation on {edge_text}. The absolute gap is under one excess MSE, and the
median across pools stays with UCB at \(\eta=2\)
({med('gradual_relocate','UCB-H2'):.1f} against
{med('gradual_relocate','QL-UCB-H2'):.1f}).

\section{{The sign flip is the one Q gain}}

On a gradual sign flip the tax is the wrong instrument. UCB sits at
{med('gradual_flip','UCB'):.1f} and UCB at \(\eta=2\) at
{med('gradual_flip','UCB-H2'):.1f}. Discounted UCB is
{med('gradual_flip','DUCB'):.1f}. Q plus a UCB bonus, with the tax off, is
{med('gradual_flip','QL-UCB'):.1f}. That gap clears the larger seed standard
deviation against discounted UCB on {clear('gradual_flip','QL-UCB','DUCB')}/6
pools and against plain UCB on {clear('gradual_flip','QL-UCB','UCB')}/6.
The pool that stays inside the seed noise is {cal_text}. Speedy Q is
{med('gradual_flip','SQL'):.1f} and clears discounted UCB on
{clear('gradual_flip','SQL','DUCB')}/6 pools, so the median alone overstates it.
The post-shift hit rate of Q plus the bonus is {hit('gradual_flip','QL-UCB'):.2f}.
The rule still misses the live card on most batches after the ramp. This is a
side result. It does not replace the relocation index.

\section{{Noise growth}}

When the best card never changes, forgetting a still-valid history costs excess.
Thompson sampling is {med('gradual_noise','TS'):.1f} and clears UCB on
{clear('gradual_noise','TS','UCB')}/6 pools. UCB is
{med('gradual_noise','UCB'):.1f} and UCB at \(\eta=2\) is
{med('gradual_noise','UCB-H2'):.1f}. Greedy Q at \(\alpha=0.05\) is
{med('gradual_noise','QL05'):.1f}, speedy Q is
{med('gradual_noise','SQL'):.1f}, and Q plus a bonus is
{med('gradual_noise','QL-UCB'):.1f}. UCB at \(\eta=2\) clears that bonus rule
on {clear('gradual_noise','UCB-H2','QL-UCB')}/6 pools.

The relocation index stays UCB minus \(\eta\sigma w_k\) at \(\eta=2\).
The sign-flip gain of Q plus a bonus is a side record, and on that ramp the tax
stays off.

\input{{ql_table.tex}}

\begin{{figure}}[ht]
\centering
\includegraphics[width=\textwidth]{{ql_bars.png}}
\caption{{Median post-shift excess across the six pools.
UCB at \(\eta=2\) is marked in green. The top panel is gradual relocation.
The bottom panel is the gradual sign flip, where that tax raises the excess.}}
\end{{figure}}
\end{{document}}
"""


def write_outputs(df: pd.DataFrame):
    os.makedirs(OUT, exist_ok=True)
    summary = summarize(df)
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    focus = ["UCB", "UCB-H2", "QL20", "QL-UCB", "QL-UCB-H2", "QWIN-UCB-H2"]
    pools_df = pool_table(df, "gradual_relocate", focus)
    pools_df.to_csv(os.path.join(OUT, "relocate_pools.csv"), index=False)
    flip_focus = ["UCB", "DUCB", "UCB-H2", "QL-UCB", "SQL"]
    has_flip = (df["scenario"] == "gradual_flip").any()
    flip_df = pool_table(df, "gradual_flip", flip_focus) if has_flip else pools_df.iloc[0:0]
    if has_flip:
        flip_df.to_csv(os.path.join(OUT, "flip_pools.csv"), index=False)
    if (df["scenario"] == "gradual_relocate").any():
        _plot(summary, os.path.join(OUT, "ql_bars.png"))
    with open(os.path.join(OUT, "ql_table.tex"), "w", encoding="utf-8") as handle:
        handle.write(_tex(summary, pools_df, flip_df))
    if set(KINDS).issubset(set(df["scenario"])):
        note = _note(summary, df)
    else:
        note = (
            "Smoke run only. The full note is written when relocation, "
            "sign flip, and noise growth are all present.\n"
        )
    with open(os.path.join(OUT, "ql_note.tex"), "w", encoding="utf-8") as handle:
        handle.write(note)
    print(summary.to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pools = load_pools(smoke=args.smoke)
    names = list(pools.keys())
    seeds = SEEDS[:1] if args.smoke else SEEDS
    kinds = ("gradual_relocate",) if args.smoke else KINDS
    tasks = [(name, kind, seed) for name in names for kind in kinds for seed in seeds]
    rows = []
    if args.workers <= 1:
        os.environ["FSDS_SMOKE"] = "1" if args.smoke else "0"
        _init_worker(args.smoke)
        for task in tasks:
            rows.extend(_job(task))
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.smoke,),
        ) as pool:
            for part in pool.map(_job, tasks):
                rows.extend(part)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "detail.csv"), index=False)
    write_outputs(df)


if __name__ == "__main__":
    main()
