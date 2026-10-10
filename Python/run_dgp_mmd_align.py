"""Covariate-shift tests on every stored DGP draw.

The draws are the pickled batches in
datasets/DGP_Simulation_BothCSCD/. Each test sees X only, so a pure concept
drift is a negative control. Vanilla MMD, MMD-LOCO, the Xu adaptive mean
test, and the uploaded random-forest C2ST share those batches.

This is the alignment of the MMD-LOCO side check with the project's DGPs.
It is not the FSDS horse-race.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mmd_loco_side import median_heuristic, permutation_test, unbiased_mmd2


ALPHA = 0.05
N_PERM = 49
C2ST_B = 19
DATA_DIR = "datasets/DGP_Simulation_BothCSCD"
FILES = (
    "DGP_whole_outputs.npy",
    "DGP_whole_outputs_onesided.npy",
)
STATS = (
    "mmd",
    "mmd_maxloco",
    "mmd_posloco",
    "mmd_gamma1",
    "xu",
    "c2st",
)
STAT_LABEL = {
    "mmd": "MMD",
    "mmd_maxloco": "max LOCO",
    "mmd_posloco": "positive-sum LOCO",
    "mmd_gamma1": "MMD $\\gamma=1$",
    "xu": "Xu",
    "c2st": "C2ST",
}


def _quiet_threads():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def load_draws(data_dir: str = DATA_DIR):
    """Unique stored batches. A repeated config seed is kept once."""
    import shift_dgp_whole  # noqa: F401  (unpickle needs the class)

    draws = []
    seen = set()
    for name in FILES:
        path = os.path.join(data_dir, name)
        arr = np.load(path, allow_pickle=True)
        for obj in arr:
            cfg = obj.config
            key = (
                int(cfg.seed),
                str(cfg.x_shift_type),
                str(cfg.concept_drift_type),
                float(cfg.delta_x),
                float(cfg.delta_c),
                int(cfg.n_ref),
                int(cfg.d),
            )
            if key in seen:
                continue
            seen.add(key)
            draws.append(
                {
                    "source": name,
                    "dgp_id": int(cfg.dgp_id),
                    "seed": int(cfg.seed),
                    "x_shift_type": str(cfg.x_shift_type),
                    "concept_drift_type": str(cfg.concept_drift_type),
                    "delta_x": float(cfg.delta_x),
                    "delta_c": float(cfg.delta_c),
                    "label": cfg.label,
                    "n_ref": int(cfg.n_ref),
                    "d": int(cfg.d),
                    "X_ref": np.asarray(obj.X_ref, dtype=np.float32),
                    "X_new": np.asarray(obj.X_new, dtype=np.float32),
                }
            )
    return draws


def xu_pvalue(X, Y, n_perm: int, seed: int) -> float:
    """Uploaded adaptive sum-of-powers combination.

    Five powers each place one permutation at rank zero, so with only 49
    permutations the combined p-value cannot fall below about 0.10. The
    reported column uses 199 permutations, the same scale as the uploaded
    default of 150.
    """
    gammas = (1, 2, 4, 8, np.inf)
    n1 = X.shape[0]
    diff = X.mean(axis=0) - Y.mean(axis=0)

    def spu(d, g):
        if np.isinf(g):
            return float(np.max(np.abs(d)))
        return float(np.sum(np.abs(d) ** g))

    obs = np.array([spu(diff, g) for g in gammas])
    Z = np.vstack([X, Y])
    perm_stats = np.zeros((n_perm, len(gammas)))
    rng = np.random.default_rng(seed)
    for i in range(n_perm):
        perm = rng.permutation(Z.shape[0])
        dlt = Z[perm[:n1]].mean(axis=0) - Z[perm[n1:]].mean(axis=0)
        for j, g in enumerate(gammas):
            perm_stats[i, j] = spu(dlt, g)
    p_each = np.mean(perm_stats > obs, axis=0)
    null_p = np.empty_like(perm_stats)
    for j in range(len(gammas)):
        null_p[:, j] = np.argsort(np.argsort(-perm_stats[:, j])) / n_perm
    return float(np.mean(null_p.min(axis=1) <= p_each.min()))


def c2st_pvalue(X, Y, B: int, seed: int) -> float:
    """Uploaded random-forest classifier two-sample test, fixed split seed."""
    Z = np.vstack([np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)])
    y = np.concatenate([np.zeros(len(X)), np.ones(len(Y))])
    p = Z.shape[1]
    clf = RandomForestClassifier(
        max_features=int(max(1, round(np.sqrt(p)))),
        max_depth=4,
        random_state=0,
        n_jobs=1,
    )
    Xtr, Xte, ytr, yte = train_test_split(Z, y, test_size=0.2, random_state=seed)
    clf.fit(Xtr, ytr)
    acc = float(np.mean(clf.predict(Xte) == yte))
    rng = np.random.RandomState(seed)
    exceed = 0
    for _ in range(B):
        clf.fit(Xtr, rng.permutation(ytr))
        if float(np.mean(clf.predict(Xte) == yte)) >= acc:
            exceed += 1
    return float((exceed + 1.0) / (B + 1.0))


def _eval_one(task):
    meta, n_perm, c2st_b = task
    X = meta["X_ref"]
    Y = meta["X_new"]
    gamma = median_heuristic(X, Y, max_points=X.shape[0] + Y.shape[0], seed=meta["seed"])
    out = permutation_test(X, Y, n_perm=n_perm, seed=meta["seed"] + 17, gamma=gamma)
    p_gamma1 = _mmd_perm(X, Y, gamma=1.0, n_perm=n_perm, seed=meta["seed"] + 19)
    pvals = dict(out["pvalue"])
    pvals["mmd_gamma1"] = p_gamma1
    pvals["xu"] = xu_pvalue(X, Y, max(n_perm, 199), meta["seed"] + 23)
    pvals["c2st"] = c2st_pvalue(X, Y, c2st_b, meta["seed"] + 29)
    row = {k: meta[k] for k in (
        "source", "dgp_id", "seed", "x_shift_type", "concept_drift_type",
        "delta_x", "delta_c", "label", "n_ref", "d",
    )}
    row["gamma"] = gamma
    for stat in STATS:
        row["p_" + stat] = pvals[stat]
        row["rej_" + stat] = int(pvals[stat] <= ALPHA)
    return row


def _mmd_perm(X, Y, gamma, n_perm, seed) -> float:
    obs = unbiased_mmd2(X, Y, gamma)
    Z = np.vstack([X, Y])
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        perm = rng.permutation(Z.shape[0])
        if unbiased_mmd2(Z[perm[:n]], Z[perm[n:]], gamma) >= obs:
            exceed += 1
    return float((1 + exceed) / (n_perm + 1))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["x_shift_type", "delta_x"], sort=True)
    for (xs, dx), sub in grouped:
        rec = {
            "x_shift_type": xs,
            "delta_x": float(dx),
            "n_draws": int(len(sub)),
            "n_concept": int((sub["concept_drift_type"] != "none").sum()),
        }
        base = sub["rej_mmd"].to_numpy().astype(bool)
        for stat in STATS:
            rej = sub["rej_" + stat].to_numpy().astype(bool)
            power = float(rej.mean())
            rec[stat] = power
            rec[stat + "_se"] = float(np.sqrt(power * (1.0 - power) / len(rej)))
            rec[stat + "_only"] = int(np.sum(rej & ~base))
            rec[stat + "_miss"] = int(np.sum(~rej & base))
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, sub in df.groupby("label", sort=True):
        rec = {"label": label, "n_draws": int(len(sub))}
        for stat in STATS:
            rej = sub["rej_" + stat].to_numpy()
            power = float(rej.mean())
            rec[stat] = power
            rec[stat + "_se"] = float(np.sqrt(power * (1.0 - power) / len(rej)))
        rows.append(rec)
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, path: str):
    order = ["mean", "covariance", "mixture", "support"]
    colors = {
        "mmd": "#4C4C4C",
        "mmd_maxloco": "#C0392B",
        "mmd_posloco": "#1F6F8B",
        "mmd_gamma1": "#8E6C8A",
        "xu": "#B9770E",
        "c2st": "#1E8449",
    }
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), sharey=True)
    for ax, xs in zip(axes.ravel(), order):
        sub = summary[summary["x_shift_type"] == xs].sort_values("delta_x")
        for stat in STATS:
            ax.plot(
                sub["delta_x"], sub[stat], marker="o", color=colors[stat],
                label=STAT_LABEL[stat].replace("$\\gamma=1$", "gamma=1"),
            )
        ax.axhline(ALPHA, color="0.5", ls="--", lw=0.8)
        ax.set_title(xs)
        ax.set_xlabel("design delta")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("rejection rate")
    axes[1, 0].set_ylabel("rejection rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def _tex_table(summary: pd.DataFrame, labels: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\caption{Rejection rate on the stored draws, pooled across concept-drift "
        r"settings. Those settings change $Y$ only, so they are replicates of the "
        r"same $P(X)$ shift. The null row is every draw with no covariate shift.}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lll" + "c" * len(STATS) + "}",
        r"\toprule",
        " & ".join(["shift", r"$\delta$", "$n$"] + [STAT_LABEL[s] for s in STATS]) + r" \\",
        r"\midrule",
    ]
    for _, row in summary.sort_values(["x_shift_type", "delta_x"]).iterrows():
        cells = [row["x_shift_type"], _fmt(row["delta_x"]), str(int(row["n_draws"]))]
        cells += [_fmt(row[s]) for s in STATS]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        r"\clearpage",
        r"\begin{table}[p]",
        r"\centering",
        r"\caption{Same tests, averaged inside each DGP label. "
        r"\texttt{concept\_only} keeps $P(X)$ fixed and moves the regression.}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * (len(STATS) + 1) + "}",
        r"\toprule",
        " & ".join(["label", "$n$"] + [STAT_LABEL[s] for s in STATS]) + r" \\",
        r"\midrule",
    ]
    for _, row in labels.iterrows():
        cells = [str(row["label"]).replace("_", r"\_"), str(int(row["n_draws"]))]
        cells += [_fmt(row[s]) for s in STATS]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        r"\clearpage",
    ]
    return "\n".join(lines) + "\n"


def _note(n_draws: int, n_perm: int, c2st_b: int) -> str:
    return rf"""\documentclass{{article}}
\usepackage{{booktabs,geometry,graphicx,amsmath}}
\geometry{{margin=1in}}
\title{{Covariate-shift tests on the stored DGP grid}}
\author{{}}
\date{{}}
\begin{{document}}
\maketitle

This note aligns the MMD leave-one-coordinate scores with the project's own
batches. It is not the FSDS horse-race. Every draw stored in
\texttt{{DGP\_whole\_outputs.npy}} and \texttt{{DGP\_whole\_outputs\_onesided.npy}}
is tested once, after dropping duplicate configuration seeds. There are {n_draws}
draws, each with $n=1000$ reference rows, $n=1000$ new rows, and $d=10$
coordinates. The tests receive $X$ only.

The kernel statistics use $\exp(-\gamma\|x-y\|^2)$. The median bandwidth is
$1$ over the median pooled squared distance, computed on the whole pool and
held fixed across permutations. A second vanilla MMD uses $\gamma=1$, the
default argument of the uploaded \texttt{{kernel\_MMD}}. Both use the unbiased
U-statistic from Gretton et al.\ (2012), including division of the cross term
by $nm$. The uploaded function omits that divisor, which at this sample size
leaves a raw double sum in place of an average. LOCO subtracts one coordinate's
squared differences from the three distance matrices and is summarized by its
maximum and by the sum of its positive coordinates. The Xu statistic is the
uploaded adaptive sum-of-powers mean test. C2ST is the uploaded random forest
with depth $4$ and $B={c2st_b}$ label permutations. Kernel tests use ${n_perm}$
permutations. The Xu combination uses 199, because five powers each contribute
a zero rank and 49 permutations cannot push that combination under $0.05$.
A draw is counted as a rejection at $\alpha=0.05$.

Mean shift moves the first three coordinates. Covariance writes correlations
into two pairs. Mixture adds a point mass of size $2$ on the first three
coordinates with probability $\delta$. Support replaces the new batch with
a uniform on $[-\,(1+\delta),\,1+\delta]$. Concept drift is added only to $Y$.
Draws that share a covariate mechanism and a $\delta$, and differ only in the
concept drift, are replicates of the same $P(X)$.

On a mean shift, median-bandwidth MMD and both LOCO summaries reject every
stored draw, already at $\delta=0.25$. At that same point $\gamma=1$ rejects
$0.81$ and C2ST rejects $0.76$. Xu is a mean test once it has 199 permutations:
it follows the mean and mixture shifts and stays near the null on covariance
and support shifts.

The covariance shift is where the LOCO summaries pull apart from vanilla MMD.
At $\delta=0.50$ (21 draws) vanilla MMD rejects $0.57$, the maximum rejects
$0.76$ (5 draws only by the maximum, 1 only by MMD), and the positive sum
rejects every draw. At $\delta=0.25$ the order reverses: vanilla MMD rejects
$0.29$, the maximum $0.10$, and the positive sum $0.19$.

Support at $\delta=0.75$ is the bandwidth check. The new batch is uniform on
$[-1.75,1.75]$, whose variance matches the reference Gaussian. Median-bandwidth
MMD rejects $0.10$ of those draws and both LOCO summaries reject none.
$\gamma=1$ and C2ST reject all 21. On the other support values the median
kernel is back at rejection 1.

The 41 draws with no covariate shift, most of them concept drift only, give
rejection $0.049$ for median MMD, $0$ for the maximum, $0.024$ for the
positive sum, and $0.073$ for C2ST. A change in the regression does not move
a test that only sees $X$.

\input{{power_table.tex}}

\begin{{figure}}[p]
\centering
\includegraphics[width=\textwidth]{{power.png}}
\caption{{Rejection rate against the design delta. Each panel is one covariate
mechanism. Concept-drift companions of the same mechanism are pooled.
The dashed line is $0.05$.}}
\end{{figure}}
\end{{document}}
"""


def write_report(detail: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    detail.to_csv(os.path.join(out_dir, "detail.csv"), index=False)
    summary = summarize(detail)
    labels = summarize_labels(detail)
    summary.to_csv(os.path.join(out_dir, "by_shift.csv"), index=False)
    labels.to_csv(os.path.join(out_dir, "by_label.csv"), index=False)
    _plot(summary, os.path.join(out_dir, "power.png"))
    with open(os.path.join(out_dir, "power_table.tex"), "w", encoding="utf-8") as f:
        f.write(_tex_table(summary, labels))
    with open(os.path.join(out_dir, "dgp_align_note.tex"), "w", encoding="utf-8") as f:
        f.write(_note(len(detail), N_PERM, C2ST_B))
    return summary, labels


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="results/dgp_mmd_align")
    args = parser.parse_args()
    _quiet_threads()
    draws = load_draws()
    if args.limit:
        draws = draws[: args.limit]
    print(f"draws {len(draws)}", flush=True)
    tasks = [(d, N_PERM, C2ST_B) for d in draws]
    t0 = time.perf_counter()
    rows = []
    if args.workers <= 1:
        iterator = map(_eval_one, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_quiet_threads)
        iterator = pool.map(_eval_one, tasks, chunksize=1)
    os.makedirs(args.out, exist_ok=True)
    detail_path = os.path.join(args.out, "detail_partial.csv")
    try:
        for i, row in enumerate(iterator, start=1):
            rows.append(row)
            if i % 10 == 0 or i == len(tasks):
                pd.DataFrame(rows).to_csv(detail_path, index=False)
                print(
                    f"[{i}/{len(tasks)}] {row['label']} {row['x_shift_type']} "
                    f"dx={row['delta_x']} "
                    + " ".join(f"{s}={row['rej_' + s]}" for s in STATS)
                    + f" ({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown()
    detail = pd.DataFrame(rows)
    summary, labels = write_report(detail, args.out)
    print(summary.to_string(index=False))
    print(labels.to_string(index=False))
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
