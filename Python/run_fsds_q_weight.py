"""FSDS share gap as a penalty on the visit-bonus Q-index.

The two directional scores have shares w_(1) >= w_(2). The penalty falls on
the action of the higher share, and only where the two scores prescribe
different actions. The temporal-difference backup is the observed return.
Draft note: results/fsds_q_weight/weight_note.pdf.
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
    # QW2-Z reuses QW2's seed. A shut-off that never fires reproduces that path.
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
        labels = {"control_relocate": "relocation", "control_flip": "sign flip"}
        cells = [labels.get(str(scenario), str(scenario)).replace("_", r"\_")]
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

    def ms(scenario, dataset, policy):
        sl = detail[
            (detail["scenario"] == scenario)
            & (detail["dataset"] == dataset)
            & (detail["policy"] == policy)
        ]["post"]
        return f"{float(sl.mean()):.1f} ({float(sl.std(ddof=1)):.1f})"

    def avg(scenario, policy):
        block = summary[(summary["scenario"] == scenario) & (summary["policy"] == policy)]
        return float(block["mean_post"].iloc[0])

    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs,graphicx,amsmath}}
\title{{An FSDS weight on the Q-index\\
\large The share names a stale score. The action stays in the table.}}
\author{{}}
\date{{}}
\begin{{document}}
\maketitle

\section{{The index}}

The control task keeps a visit-bonus table \(Q(s,a)\). The state \(s\) is the
three-way sign of two known reference scores and the step. An earlier table
also conditioned on which reference policy held the pseudo-outcome share.
That coordinate split the visits and raised the excess on the sign flip.
The weight in this note is a term in the decision index. The state is the
same two signs and the step.

Four reference policies are scored on every batch, on the same calendar as
the scorecard race: the first six rounds against the latest six, refreshed
every eight rounds from round 16. The weight uses the two directional
scores, the same two whose signs form the state. Write \(w_{{(1)}}\) and
\(w_{{(2)}}\) for the larger and the smaller of their shares, and
\(a_{{(1)}}(s)\), \(a_{{(2)}}(s)\) for the actions they prescribe. The
penalty is the share gap. It falls on the action of the higher share, and
only in states where the two actions differ,
\[
\delta(s,a)=(w_{{(1)}}-w_{{(2)}})\,1\{{a=a_{{(1)}}(s)\}}\,1\{{a_{{(1)}}(s)\neq a_{{(2)}}(s)\}}.
\]
The decision index is
\[
Q(s,a)+c(s,a)-\eta\sigma\cdot\frac{{H-h}}{{H}}\cdot\delta(s,a),
\]
with \(\eta\in\{{1,2\}}\) and \(\sigma\) the scale of the four reference
returns on the first six rounds. The factor \((H-h)/H\) places one
episode-level charge along the horizon. Equal shares give \(\delta=0\), so
the weight is inert while the two scores still carry the same residual.
Agreed actions also give \(\delta=0\): both scores would press the same
button, and that button carries no penalty.

The backup is the observed reward plus the max of the next raw table. The
share records which reference score still explains the outcome, and it stays
out of the value being learned. An unvisited cell still carries the large
visit bonus, so the weight cannot hide an action the table has not tried.

One variant sets the weight to zero on a round whose largest share belongs
to the zero policy. That is the sign-flip symptom the shut-off was built
for: the null score holds the mass, and a penalty aimed at a stale direction
has no stale direction left. Across the six pools, the fraction of
post-shift rounds on which the zero policy holds the share has median
{hold('control_relocate'):.2f} on relocation and {hold('control_flip'):.2f}
on the flip.

\section{{Six pools}}

A reported median is the median, across six pools, of the four-seed mean
post-shift excess. A named pool is that four-seed mean, with the seed
standard deviation in parentheses. A gap counts only when it exceeds the
larger of the two seed standard deviations.

On relocation the menu at \(\eta=2\) is {med('control_relocate','UCB-H2'):.1f}.
Visit-bonus Q with no weight is {med('control_relocate','QUCB'):.1f}. The
same table at \(\eta=1\) is {med('control_relocate','QW1'):.1f} and at
\(\eta=2\) is {med('control_relocate','QW2'):.1f}. Discounted UCB on the
menu is {med('control_relocate','DUCB'):.1f}. The \(\eta=2\) weight is lower
than unweighted Q on {clear('control_relocate','QW2','QUCB')}/6 pools, and
lower than the menu on {clear('control_relocate','QW2','UCB-H2')}/6. The
menu is lower than the weight on {clear('control_relocate','UCB-H2','QW2')}/6.
California is {ms('control_relocate','california','UCB-H2')} for the menu and
{ms('control_relocate','california','QW2')} for the weight. Covertype is
{ms('control_relocate','covertype','UCB-H2')} for the menu and
{ms('control_relocate','covertype','QW2')} for the weight. On AR(1),
Gaussian-12, Gaussian-30, and Student-$t_5$ the gap stays inside the larger
seed standard deviation. Early excess, before the ramp, is
{early('control_relocate','QUCB'):.1f} for the unweighted table and
{early('control_relocate','QW2'):.1f} for the weight. The first-action hit
rate is {hit('control_relocate','QW2'):.2f} for the weight and
{hit('control_relocate','UCB-H2'):.2f} for the menu.

On the sign flip, unweighted visit-bonus Q is {med('control_flip','QUCB'):.1f},
with first-action hit rate {hit('control_flip','QUCB'):.2f}. The weight at
\(\eta=1\) is {med('control_flip','QW1'):.1f} and at \(\eta=2\) is
{med('control_flip','QW2'):.1f}, with hit rate {hit('control_flip','QW2'):.2f}.
The menu at \(\eta=2\) is {med('control_flip','UCB-H2'):.1f} and discounted
UCB is {med('control_flip','DUCB'):.1f}. Unweighted Q is lower than the
\(\eta=2\) weight on {clear('control_flip','QUCB','QW2')}/6 pools.
Gaussian-30 is {ms('control_flip','gaussian30','QUCB')} for the unweighted
table and {ms('control_flip','gaussian30','QW2')} for the weight.
Student-$t_5$ is {ms('control_flip','student_t5','QUCB')} for the unweighted
table and {ms('control_flip','student_t5','QW2')} for the weight. AR(1),
California, and Gaussian-12 stay near 3, unweighted table then weight:
{ms('control_flip','ar1_gaussian','QUCB')} and
{ms('control_flip','ar1_gaussian','QW2')},
{ms('control_flip','california','QUCB')} and
{ms('control_flip','california','QW2')},
{ms('control_flip','gaussian12','QUCB')} and
{ms('control_flip','gaussian12','QW2')}. Covertype is
{ms('control_flip','covertype','QUCB')} and
{ms('control_flip','covertype','QW2')}; that gap is smaller than the seed
standard deviation.

The zero-policy shut-off does not fire once the ramp has crossed one half.
On relocation every pool mean matches the always-on weight, so the median
stays at {med('control_relocate','QW2-Z'):.1f}. On the flip the shut-off
fires on a few earlier rounds. The post-shift median stays at
{med('control_flip','QW2'):.1f}. The mean across the six pools moves from
{avg('control_flip','QW2'):.2f} to {avg('control_flip','QW2-Z'):.2f}.
AR(1) moves from {ms('control_flip','ar1_gaussian','QW2')} to
{ms('control_flip','ar1_gaussian','QW2-Z')}. The shut-off is lower than the
always-on weight by more than the seed standard deviation on
{clear('control_flip','QW2-Z','QW2')}/6 pools.

The share names which reference score has stopped explaining the outcome.
Once a cell has a value, that name is not the button. Relocation stays with
the menu at \(\eta=2\). The sign flip stays with visit-bonus Q, and this
weight stays off that table.

\input{{weight_table.tex}}

\begin{{figure}}[ht]
\centering
\includegraphics[width=0.92\textwidth]{{weight_bars.png}}
\caption{{Median post-shift excess across six pools. The menu at \(\eta=2\)
is marked in green. QW1 and QW2 subtract the share gap from the visit-bonus
index. QW2-Z uses the same index and sets the weight to zero on a round
whose largest share is the zero policy.}}
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
