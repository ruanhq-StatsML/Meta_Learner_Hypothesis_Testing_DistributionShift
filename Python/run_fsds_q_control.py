"""Tabular Q on a short-horizon control task with a gradual reward drift.

The committed menu is the baseline from the scorecard race: one policy for
the whole episode, screened by the pseudo-outcome share. The Q rules see the
sign pattern of the current state. QS also sees which reference policy holds
the share.
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

from fsds_q_control import prepare, run_spec
from run_fsds_porisk_poc import SEEDS, _init_worker, load_pools

OUT = os.path.join(os.path.dirname(__file__), "results", "fsds_q_control")
KINDS = ("control_relocate", "control_flip")

POLICIES = [
    {"name": "UCB", "family": "commit", "rule": "ucb", "eta": 0.0, "mode": "abs", "eps": 0.0},
    {"name": "DUCB", "family": "commit", "rule": "ducb", "eta": 0.0, "mode": "abs", "eps": 0.0},
    {"name": "UCB-H2", "family": "commit", "rule": "ucb", "eta": 2.0, "mode": "scale", "eps": 0.0},
    {"name": "EG10", "family": "commit", "rule": "eps", "eta": 0.0, "mode": "abs", "eps": 0.10},
    {"name": "Q", "family": "q", "qrule": "q", "share": False, "alpha": 0.20},
    {"name": "QUCB", "family": "q", "qrule": "qucb", "share": False, "alpha": 0.20},
    {"name": "DQ", "family": "q", "qrule": "dql", "share": False, "alpha": 0.20},
    {"name": "QS", "family": "q", "qrule": "q", "share": True, "alpha": 0.20},
    {"name": "QS-UCB", "family": "q", "qrule": "qucb", "share": True, "alpha": 0.20},
    {"name": "DQS", "family": "q", "qrule": "dql", "share": True, "alpha": 0.20},
]

PRETTY = {
    "ar1_gaussian": "AR(1)",
    "california": "California",
    "covertype": "covertype",
    "gaussian12": "Gaussian-12",
    "gaussian30": "Gaussian-30",
    "student_t5": r"Student-$t_5$",
}


def _job(payload):
    name, kind, seed = payload
    from run_fsds_porisk_poc import POOLS

    pack = prepare(POOLS[name], kind, seed)
    rows = []
    for i, policy in enumerate(POLICIES):
        stats = run_spec(pack, policy, seed + 1000 + i)
        rows.append({
            "dataset": name,
            "scenario": kind,
            "seed": seed,
            "policy": policy["name"],
            "early": stats["early"],
            "ramp": stats["ramp"],
            "post": stats["post"],
            "hit_post": stats["post_hit"],
            "hit_early": stats["early_hit"],
            "sigma": pack["sigma"],
        })
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, policy), sub in df.groupby(["scenario", "policy"], sort=False):
        pool_means = [float(pool["post"].mean()) for _name, pool in sub.groupby("dataset")]
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


def _gaps(detail, scenario, winner, loser):
    sub = detail[detail["scenario"] == scenario]
    rows = []
    for dataset, group in sub.groupby("dataset"):
        left = group[group["policy"] == winner]["post"]
        right = group[group["policy"] == loser]["post"]
        gap = float(right.mean() - left.mean())
        msd = max(float(left.std(ddof=1)), float(right.std(ddof=1)))
        rows.append({"dataset": dataset, "gap": gap, "msd": msd, "clear": gap > msd})
    return rows


def _n_clear(detail, scenario, winner, loser) -> int:
    return sum(int(row["clear"]) for row in _gaps(detail, scenario, winner, loser))


def _plot(summary: pd.DataFrame, path: str):
    order = [p["name"] for p in POLICIES]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.8), sharex=True)
    titles = {
        "control_relocate": "Control, gradual relocation",
        "control_flip": "Control, gradual sign flip",
    }
    for ax, scenario in zip(axes, KINDS):
        sub = summary[summary["scenario"] == scenario].copy()
        sub["policy"] = pd.Categorical(sub["policy"], order, ordered=True)
        sub = sub.sort_values("policy")
        colors = ["#1F7A1F" if name == "UCB-H2" else "#4C78A8" for name in sub["policy"]]
        ax.bar(np.arange(len(sub)), sub["median_post"].to_numpy(), color=colors)
        ax.set_ylabel("median post excess")
        ax.set_title(titles[scenario])
        ax.grid(True, axis="y", alpha=0.3)
    axes[-1].set_xticks(np.arange(len(order)))
    axes[-1].set_xticklabels(order, rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _cell(summary, scenario, policy, col):
    block = summary[(summary["scenario"] == scenario) & (summary["policy"] == policy)]
    return float(block[col].iloc[0])


def _pool_tex(caption, pools, focus):
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
        cells = [PRETTY.get(str(row["dataset"]), str(row["dataset"]))]
        for name in focus:
            cells.append(f"{row[name]:.1f} ({row[name + '_sd']:.1f})")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return lines


def _tex(summary, relocate, flip):
    show = ["UCB", "DUCB", "UCB-H2", "EG10", "Q", "QUCB", "DQ", "QS", "QS-UCB", "DQS"]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Median, across six pools, of the four-seed mean post-shift excess return.}",
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
    focus = ["UCB", "DUCB", "UCB-H2", "QUCB", "DQ", "QS-UCB"]
    lines += _pool_tex(
        "Control relocation. Sum of per-round mean excess return after the ramp "
        "crosses one half. Seed standard deviation in parentheses.",
        relocate,
        focus,
    )
    lines += _pool_tex(
        "Control sign flip. Same excess. Seed standard deviation in parentheses.",
        flip,
        focus,
    )
    return "\n".join(lines)


def _note(summary, detail):
    def med(scenario, policy):
        return _cell(summary, scenario, policy, "median_post")

    def hit(scenario, policy):
        return _cell(summary, scenario, policy, "mean_hit")

    def early(scenario, policy):
        return _cell(summary, scenario, policy, "mean_early")

    def clear(scenario, winner, loser):
        return _n_clear(detail, scenario, winner, loser)

    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs,graphicx,amsmath}}
\title{{Online Q on a short-horizon control task\\
\large The share is a state, not a penalty on a menu}}
\author{{}}
\date{{}}
\begin{{document}}
\maketitle

\section{{The scenario}}

The scorecard race picks one frozen score for a batch of 200 rows. A
constant-step update of that menu is a stateless bandit, and a step size
\(1/n\) is the sample mean already inside UCB. This note changes the decision.

Each round is an episode of four steps on 200 trajectories drawn from the
pool. The action is \(-1\), \(0\), or \(+1\). The state moves by
\(s\leftarrow 0.85 s + 0.15 a v\), with \(v\) the unit vector, and the reward is
\(a(s^\top\beta_t)-0.05 a^2\) plus \(N(0,0.25^2)\) noise. There is no discount.
The coefficient \(\beta_t\) follows the same gradual ramp as the relocation MVP:
it stays on the first block through round 20, then moves linearly onto the
second block. A second run flips the sign of the first block on that same
ramp. The next round is not announced, and there is no jump.

The observable state for Q is the three-way sign of two known reference
scores, \(s^\top\beta_A\) and \(s^\top\beta_B\), and the step index. Nine
cells times four steps. On a sign flip the second score is the negative of
the first, so the cell does not by itself say which sign is live. A second
set of tables also conditions on which of the four reference policies
currently holds the pseudo-outcome share. The share uses the fixed calendar
from the scorecard race: the first six rounds against the latest six, refreshed
every eight rounds from round 16. Until that refresh the share bin is empty.
Reference policies are scored on every batch, as the scorecards were.

The baseline commits to one reference policy for the whole episode and picks
it with UCB, discounted UCB, \(\varepsilon=0.10\), or UCB minus \(\eta\sigma w_k\)
at \(\eta=2\). Q is updated once per visited state-action at the end of the
step, with \(\alpha=0.2\), on the average temporal-difference error of the
particles in that cell. Tables start at zero. Greedy Q with no bonus keeps
the first action that breaks the tie. The visit-bonus rules add
\(\sqrt{{2\log(t+1)/n}}\) at decision time only, with a large bonus on a
cell that has not been visited, and the bonus is not part of the backup.
Double Q keeps two tables, acts on their average plus that same bonus, and
bootstraps from the table that was not updated. Ties are broken by a fresh
jitter, so the first batch spreads its particles across tied actions.

Excess is the noise-free return of the finite-horizon optimum minus the
noise-free return of the actions taken, summed over rounds after the ramp
crosses one half. The optimum is the exact action tree; the dynamics have no
process noise.

\section{{Relocation}}

UCB at \(\eta=2\) has median post-shift excess
{med('control_relocate','UCB-H2'):.1f}, against
{med('control_relocate','UCB'):.1f} for plain UCB and
{med('control_relocate','DUCB'):.1f} for discounted UCB.
\(\varepsilon=0.10\) is {med('control_relocate','EG10'):.1f}.
Greedy Q on the sign state is {med('control_relocate','Q'):.1f}.
The same table with the share bin is {med('control_relocate','QS'):.1f}.
Q with a visit bonus is {med('control_relocate','QUCB'):.1f} without the
share and {med('control_relocate','QS-UCB'):.1f} with it. Double Q is
{med('control_relocate','DQ'):.1f} on signs and
{med('control_relocate','DQS'):.1f} with the share bin.

The visit-bonus table on the two scores, QUCB, is lower than plain UCB by
more than the larger seed standard deviation on
{clear('control_relocate','QUCB','UCB')}/6 pools, and lower than UCB at
\(\eta=2\) on {clear('control_relocate','QUCB','UCB-H2')}/6. UCB at
\(\eta=2\) is lower than QUCB on
{clear('control_relocate','UCB-H2','QUCB')}/6. Adding the share bin, QS, is
lower than UCB at \(\eta=2\) on {clear('control_relocate','QS','UCB-H2')}/6
and higher on {clear('control_relocate','UCB-H2','QS')}/6. Greedy Q with no
bonus is lower than UCB at \(\eta=2\) on
{clear('control_relocate','Q','UCB-H2')}/6. Early excess, before the ramp,
is {early('control_relocate','UCB'):.1f} for UCB,
{early('control_relocate','UCB-H2'):.1f} for the tax,
{early('control_relocate','Q'):.1f} for sign-only Q, and
{early('control_relocate','QS'):.1f} for the share-bin table.
The post-shift hit rate on the first action is
{hit('control_relocate','QUCB'):.2f} for visit-bonus Q and
{hit('control_relocate','UCB-H2'):.2f} for UCB at \(\eta=2\).

\section{{Sign flip}}

On the sign flip, UCB is {med('control_flip','UCB'):.1f}, discounted UCB is
{med('control_flip','DUCB'):.1f}, and UCB at \(\eta=2\) is
{med('control_flip','UCB-H2'):.1f}. Visit-bonus Q on the two scores is
{med('control_flip','QUCB'):.1f}, and double Q on the same state is
{med('control_flip','DQ'):.1f}. Greedy Q with no bonus is
{med('control_flip','Q'):.1f}. The share-bin table is
{med('control_flip','QS'):.1f}, its visit-bonus version is
{med('control_flip','QS-UCB'):.1f}, and double Q with the share bin is
{med('control_flip','DQS'):.1f}.

QUCB is lower than discounted UCB by more than the seed standard deviation on
{clear('control_flip','QUCB','DUCB')}/6 pools, and lower than UCB at
\(\eta=2\) on {clear('control_flip','QUCB','UCB-H2')}/6. It is lower than the
share-bin visit-bonus table on {clear('control_flip','QUCB','QS-UCB')}/6.
Greedy Q with no bonus is lower than discounted UCB on
{clear('control_flip','Q','DUCB')}/6. The first-action hit rate is
{hit('control_flip','QUCB'):.2f} for QUCB and
{hit('control_flip','UCB-H2'):.2f} for UCB at \(\eta=2\).

A constant step replaces the action stored in a cell. On a flip the same
score-sign cell needs the opposite action after the ramp, and the visit bonus
is what keeps the other actions available while that replacement happens.
Splitting the cell by the share bin means each piece is visited less often.
The scorecard MVP keeps its index. This note is the record of the control
task, where the decision depends on the current scores.

\input{{control_table.tex}}

\begin{{figure}}[ht]
\centering
\includegraphics[width=\textwidth]{{control_bars.png}}
\caption{{Median post-shift excess return across six pools.
UCB at \(\eta=2\) is marked in green. The top panel relocates the reward
block. The bottom panel flips its sign.}}
\end{{figure}}
\end{{document}}
"""


def write_outputs(df: pd.DataFrame):
    os.makedirs(OUT, exist_ok=True)
    summary = summarize(df)
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    focus = ["UCB", "DUCB", "UCB-H2", "QUCB", "DQ", "QS-UCB"]
    relocate = pool_table(df, "control_relocate", focus)
    flip = pool_table(df, "control_flip", focus)
    relocate.to_csv(os.path.join(OUT, "relocate_pools.csv"), index=False)
    flip.to_csv(os.path.join(OUT, "flip_pools.csv"), index=False)
    if set(KINDS).issubset(set(df["scenario"])):
        _plot(summary, os.path.join(OUT, "control_bars.png"))
        with open(os.path.join(OUT, "control_table.tex"), "w", encoding="utf-8") as handle:
            handle.write(_tex(summary, relocate, flip))
        with open(os.path.join(OUT, "control_note.tex"), "w", encoding="utf-8") as handle:
            handle.write(_note(summary, df))
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
    kinds = ("control_relocate",) if args.smoke else KINDS
    tasks = [(name, kind, seed) for name in names for kind in kinds for seed in seeds]
    rows = []
    if args.workers <= 1:
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
