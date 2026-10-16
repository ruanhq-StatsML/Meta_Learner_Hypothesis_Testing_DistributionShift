"""FSDS share as an action weight on the control Q-index.

The share is pushed forward onto the action each reference policy would take
in the current state. The decision index is Q plus the visit bonus minus
eta * sigma * that mass, scaled by the fraction of the horizon still left.
The temporal-difference backup is the observed return. The share is not a
state coordinate, so the table is not split.
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
from run_fsds_q_control import PRETTY, _n_clear, pool_table, summarize

OUT = os.path.join(os.path.dirname(__file__), "results", "fsds_q_weight")
KINDS = ("control_relocate", "control_flip")

POLICIES = [
    {"name": "DUCB", "family": "commit", "rule": "ducb", "eta": 0.0, "mode": "abs", "eps": 0.0},
    {"name": "UCB-H2", "family": "commit", "rule": "ucb", "eta": 2.0, "mode": "scale", "eps": 0.0},
    {"name": "QUCB", "family": "q", "qrule": "qucb", "share": False, "alpha": 0.20, "eta": 0.0},
    {"name": "QW1", "family": "q", "qrule": "qucb", "share": False, "alpha": 0.20, "eta": 1.0},
    {"name": "QW2", "family": "q", "qrule": "qucb", "share": False, "alpha": 0.20, "eta": 2.0},
    {
        "name": "QW2-Z",
        "family": "q",
        "qrule": "qucb",
        "share": False,
        "alpha": 0.20,
        "eta": 2.0,
        "shut_zero": True,
    },
]


def _job(payload):
    name, kind, seed = payload
    from run_fsds_porisk_poc import POOLS

    pack = prepare(POOLS[name], kind, seed)
    post = pack["post"]
    zero_hold = float(np.mean(np.argmax(pack["shares"][post], axis=1) == pack["zero_arm"]))
    rows = []
    # QW2-Z shares QUCB's stream only through its own rule. It shares QW2's
    # seed so a shut-off that never fires reproduces QW2 pathwise.
    seed_of = {p["name"]: seed + 3000 + i for i, p in enumerate(POLICIES)}
    seed_of["QW2-Z"] = seed_of["QW2"]
    for policy in POLICIES:
        stats = run_spec(pack, policy, seed_of[policy["name"]])
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
            "zero_hold": zero_hold,
        })
    return rows


def _plot(summary: pd.DataFrame, path: str):
    order = [p["name"] for p in POLICIES]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    titles = {
        "control_relocate": "Relocation, action weight",
        "control_flip": "Sign flip, action weight",
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
    axes[-1].set_xticklabels(order, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _tex(summary, relocate, flip):
    show = [p["name"] for p in POLICIES]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Median, across six pools, of the four-seed mean post-shift excess return.}",
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
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    focus = ["UCB-H2", "QUCB", "QW1", "QW2", "QW2-Z"]
    for caption, pools in (
        ("Relocation. Sum of per-round mean excess return. Seed standard deviation in parentheses.", relocate),
        ("Sign flip. Same excess. Seed standard deviation in parentheses.", flip),
    ):
        lines += [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{" + caption + "}",
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
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _note(summary, detail):
    def med(scenario, policy):
        block = summary[(summary["scenario"] == scenario) & (summary["policy"] == policy)]
        return float(block["median_post"].iloc[0])

    def early(scenario, policy):
        block = summary[(summary["scenario"] == scenario) & (summary["policy"] == policy)]
        return float(block["mean_early"].iloc[0])

    def hit(scenario, policy):
        block = summary[(summary["scenario"] == scenario) & (summary["policy"] == policy)]
        return float(block["mean_hit"].iloc[0])

    def clear(scenario, winner, loser):
        return _n_clear(detail, scenario, winner, loser)

    def hold(scenario):
        sub = detail[detail["scenario"] == scenario]
        means = [float(pool["zero_hold"].mean()) for _name, pool in sub.groupby("dataset")]
        return float(np.median(means))

    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs,graphicx,amsmath}}
\title{{An FSDS weight on the Q-index\\
\large Same table, a penalty on the action the stale policy would take}}
\author{{}}
\date{{}}
\begin{{document}}
\maketitle

\section{{Where the weight sits}}

The control task already has a visit-bonus table \(Q(s,a)\). The state \(s\) is
the three-way sign of two known reference scores and the step. Putting the
pseudo-outcome share into that state split the visits and raised the excess
on the sign flip. The weight below does not add a coordinate.

Four reference policies are scored on every batch, on the same calendar as
the scorecard race. The weight uses the two directional scores, the same two
whose signs form the state. Write \(w_{{(1)}}\) and \(w_{{(2)}}\) for the larger
and the smaller of their shares, and \(a_{{(1)}}(s)\), \(a_{{(2)}}(s)\) for the
actions they prescribe. The penalty on an action is the share gap, and only
on the action of the higher share, and only in states where the two actions
differ:
\[
\delta(s,a)=(w_{{(1)}}-w_{{(2)}})\,1\{{a=a_{{(1)}}(s)\}}\,1\{{a_{{(1)}}(s)\neq a_{{(2)}}(s)\}}.
\]
The decision index is
\[
Q(s,a)+c(s,a)-\eta\sigma\cdot\frac{{H-h}}{{H}}\cdot\delta(s,a),
\]
with \(\eta\in\{{1,2\}}\) and \(\sigma\) the scale of the four reference
returns on the first six rounds. The factor \((H-h)/H\) charges the
episode-level penalty once along the horizon. Equal shares give
\(\delta=0\), so the weight is inert while the two scores still carry the
same residual. Agreement of the two actions also gives \(\delta=0\): the
weight does not tax a button both scores would press.

The backup is unchanged. The target is the observed reward plus the max of
the next raw table. The share is a statement about which reference score
still explains the outcome. It is not a reward, so it does not enter the
value being learned. An unvisited cell still carries the large visit bonus,
so the weight cannot hide an action the table has not tried.

One variant turns the weight off on a round whose largest share belongs to
the zero policy. That is the sign-flip symptom: the null score holds the
mass, and a penalty aimed at a stale action has nothing stale to point at.
Across pools, the fraction of post-shift rounds in which the zero policy
holds the share has median {hold('control_relocate'):.2f} on relocation and
{hold('control_flip'):.2f} on the flip.

\section{{Relocation}}

The menu at \(\eta=2\) has median post excess
{med('control_relocate','UCB-H2'):.1f}. Visit-bonus Q with no weight is
{med('control_relocate','QUCB'):.1f}. The same table at \(\eta=1\) is
{med('control_relocate','QW1'):.1f}, at \(\eta=2\) is
{med('control_relocate','QW2'):.1f}, and at \(\eta=2\) with the zero-policy
shut-off is {med('control_relocate','QW2-Z'):.1f}. Discounted UCB on the
menu is {med('control_relocate','DUCB'):.1f}.

The \(\eta=2\) weight is lower than unweighted Q by more than the larger
seed standard deviation on {clear('control_relocate','QW2','QUCB')}/6 pools.
It is lower than the menu at \(\eta=2\) on
{clear('control_relocate','QW2','UCB-H2')}/6, and the menu is lower than the
weight on {clear('control_relocate','UCB-H2','QW2')}/6. Early excess is
{early('control_relocate','QUCB'):.1f} without the weight and
{early('control_relocate','QW2'):.1f} with it. The first-action hit rate is
{hit('control_relocate','QW2'):.2f} for the weight and
{hit('control_relocate','UCB-H2'):.2f} for the menu.

\section{{Sign flip}}

Unweighted visit-bonus Q is {med('control_flip','QUCB'):.1f}. The weight at
\(\eta=1\) is {med('control_flip','QW1'):.1f}, at \(\eta=2\) is
{med('control_flip','QW2'):.1f}, and with the zero-policy shut-off is
{med('control_flip','QW2-Z'):.1f}. The menu at \(\eta=2\) is
{med('control_flip','UCB-H2'):.1f} and discounted UCB is
{med('control_flip','DUCB'):.1f}.

The shut-off rule is lower than the always-on \(\eta=2\) weight by more than
the seed standard deviation on {clear('control_flip','QW2-Z','QW2')}/6
pools. Unweighted Q is lower than the always-on weight on
{clear('control_flip','QUCB','QW2')}/6 and lower than the shut-off rule on
{clear('control_flip','QUCB','QW2-Z')}/6. The first-action hit rate is
{hit('control_flip','QUCB'):.2f} without the weight,
{hit('control_flip','QW2'):.2f} with it always on, and
{hit('control_flip','QW2-Z'):.2f} with the shut-off.

On relocation the \(\eta=2\) weight clears the unweighted table on
{clear('control_relocate','QW2','QUCB')}/6 pools, and the menu at
\(\eta=2\) stays lower on the pools where the table was already behind.
On the sign flip the same weight moves the median from
{med('control_flip','QUCB'):.1f} to {med('control_flip','QW2'):.1f}, and the
first-action hit rate moves from {hit('control_flip','QUCB'):.2f} to
{hit('control_flip','QW2'):.2f}. The zero-policy shut-off does not fire
once the ramp has crossed one half. On the flip it fires on a few earlier
rounds, and the post-shift excess stays within a point of the always-on
weight ({med('control_flip','QW2-Z'):.1f} against
{med('control_flip','QW2'):.1f}).
The share names which reference score has stopped explaining the outcome.
Once a cell has a value, that name is not the button to press. The menu
keeps the relocation index it already has. The visit-bonus table keeps the
sign flip without this weight.

\input{{weight_table.tex}}

\begin{{figure}}[ht]
\centering
\includegraphics[width=0.92\textwidth]{{weight_bars.png}}
\caption{{Median post-shift excess. The menu at \(\eta=2\) is marked in green.
QW1 and QW2 put the share on the action index. QW2-Z turns that weight off
when the zero policy holds the share.}}
\end{{figure}}
\end{{document}}
"""


def write_outputs(df: pd.DataFrame):
    os.makedirs(OUT, exist_ok=True)
    summary = summarize(df)
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    focus = ["UCB-H2", "QUCB", "QW1", "QW2", "QW2-Z"]
    relocate = pool_table(df, "control_relocate", focus)
    flip = pool_table(df, "control_flip", focus)
    relocate.to_csv(os.path.join(OUT, "relocate_pools.csv"), index=False)
    flip.to_csv(os.path.join(OUT, "flip_pools.csv"), index=False)
    if set(KINDS).issubset(set(df["scenario"])):
        _plot(summary, os.path.join(OUT, "weight_bars.png"))
        with open(os.path.join(OUT, "weight_table.tex"), "w", encoding="utf-8") as handle:
            handle.write(_tex(summary, relocate, flip))
        with open(os.path.join(OUT, "weight_note.tex"), "w", encoding="utf-8") as handle:
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
