"""Side power check: MMD-LOCO against vanilla MMD on a mean covariate shift.

Not the FSDS horse-race. Both statistics share the RBF kernel, the median
bandwidth, and the permutation draws. Rejection is at alpha = 0.05.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmd_loco_side import loco_matches_column_delete, median_heuristic, permutation_test


ALPHA = 0.05
STATS = (
    "mmd",
    "mmd_maxloco",
    "mmd_posloco",
    "nammd",
    "nammd_maxloco",
    "nammd_posloco",
)
STAT_LABEL = {
    "mmd": "MMD",
    "mmd_maxloco": "max LOCO",
    "mmd_posloco": "positive-sum LOCO",
    "nammd": "NAMMD",
    "nammd_maxloco": "max NAMMD-LOCO",
    "nammd_posloco": "positive-sum NAMMD-LOCO",
}


def draw_pair(n: int, p: int, s: int, delta: float, rho: float, rng: np.random.Generator):
    def sample():
        if rho == 0.0:
            return rng.normal(size=(n, p))
        eps = rng.normal(size=(n, p))
        X = np.empty((n, p), dtype=np.float64)
        X[:, 0] = eps[:, 0]
        scale = np.sqrt(1.0 - rho * rho)
        for j in range(1, p):
            X[:, j] = rho * X[:, j - 1] + scale * eps[:, j]
        return X

    X_ref = sample()
    X_new = sample()
    if s > 0 and delta != 0.0:
        X_new[:, :s] += delta
    return X_ref, X_new


def _one_rep(task):
    n, p, s, delta, rho, n_perm, seed = task
    rng = np.random.default_rng(seed)
    X_ref, X_new = draw_pair(n, p, s, delta, rho, rng)
    gamma = median_heuristic(X_ref, X_new, seed=seed)
    out = permutation_test(X_ref, X_new, n_perm=n_perm, seed=seed + 17, gamma=gamma)
    hit = int(out["observed"]["loco_hat"] < s) if s > 0 else np.nan
    row = {
        "pvalue_" + k: out["pvalue"][k]
        for k in STATS
    }
    row["hit"] = hit
    row["gamma"] = out["gamma"]
    return row


def run_cell(n, p, s, delta, rho, reps, n_perm, seed0, workers):
    tasks = [
        (n, p, s, delta, rho, n_perm, seed0 + 100003 * r)
        for r in range(reps)
    ]
    if workers <= 1:
        rows = [_one_rep(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_quiet_threads) as pool:
            rows = list(pool.map(_one_rep, tasks, chunksize=1))
    return rows


def _quiet_threads():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def summarize(rows, meta):
    out = []
    hits = np.array([r["hit"] for r in rows], dtype=float)
    rej = {}
    pvals_by = {}
    for stat in STATS:
        pvals = np.array([r["pvalue_" + stat] for r in rows], dtype=float)
        pvals_by[stat] = pvals
        rej[stat] = pvals <= ALPHA
    base = rej["mmd"]
    for stat in STATS:
        reject = rej[stat]
        power = float(np.mean(reject))
        se = float(np.sqrt(power * (1.0 - power) / len(reject)))
        out.append(
            {
                **meta,
                "stat": stat,
                "power": power,
                "se": se,
                "mean_pvalue": float(np.mean(pvals_by[stat])),
                "loco_hit": float(np.nanmean(hits)),
                "n_only_this": int(np.sum(reject & ~base)),
                "n_only_mmd": int(np.sum(~reject & base)),
                "n_both": int(np.sum(reject & base)),
            }
        )
    return out


def smoke():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    Y = rng.normal(size=(40, 6))
    Y[:, 0] += 0.8
    gamma = median_heuristic(X, Y)
    gap = loco_matches_column_delete(X, Y, gamma)
    t0 = time.perf_counter()
    out = permutation_test(X, Y, n_perm=29, seed=1, gamma=gamma)
    elapsed = time.perf_counter() - t0
    print(f"identity_gap {gap:.3e}")
    print(f"one_test_sec {elapsed:.3f} gamma {out['gamma']:.4f}")
    for k, v in out["pvalue"].items():
        print(f"  {k:16s} stat {out['observed'][k]: .5f}  p {v:.3f}")
    # float32 Gram matrices: the distance update and an explicit column drop
    # agree well past the scale of the MMD itself (typically 1e-2 to 1e-1).
    if gap > 1e-4:
        raise SystemExit("LOCO distance update does not match column deletion")


def write_outputs(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "power.csv")
    df.to_csv(csv_path, index=False)
    fig_path = os.path.join(out_dir, "power.png")
    _plot(df, fig_path)
    tex_path = os.path.join(out_dir, "power_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(_latex_table(df))
    note_path = os.path.join(out_dir, "mmd_loco_note.tex")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(_note(df))
    return csv_path, fig_path, tex_path, note_path


def _plot(df: pd.DataFrame, path: str):
    show = ["mmd", "mmd_maxloco", "mmd_posloco"]
    colors = {"mmd": "#4C4C4C", "mmd_maxloco": "#C0392B", "mmd_posloco": "#1F6F8B"}
    indep = df[df["rho"] == 0].copy()
    panels = list(indep.groupby(["p", "s"], sort=True).groups.keys())
    ncol = min(3, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow), sharey=True, squeeze=False)
    for ax, (p, s) in zip(axes.ravel(), panels):
        sub = indep[(indep["p"] == p) & (indep["s"] == s)]
        for stat in show:
            sl = sub[sub["stat"] == stat].sort_values("delta")
            ax.plot(sl["delta"], sl["power"], marker="o", color=colors[stat], label=STAT_LABEL[stat])
            ax.fill_between(
                sl["delta"],
                np.clip(sl["power"] - 1.96 * sl["se"], 0, 1),
                np.clip(sl["power"] + 1.96 * sl["se"], 0, 1),
                color=colors[stat],
                alpha=0.12,
            )
        ax.axhline(ALPHA, color="0.5", ls="--", lw=0.8)
        title = f"p={p}, sparse s={s}" if s < p else f"p={p}, dense s={s}"
        ax.set_title(title)
        ax.set_xlabel("mean shift")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
    axes.ravel()[0].set_ylabel("rejection rate")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def _cell_keys(block: pd.DataFrame) -> pd.DataFrame:
    return (
        block.groupby(["p", "s", "delta"], sort=True)
        .size()
        .reset_index()[["p", "s", "delta"]]
    )


def _power_at(block, p, s, delta, stat) -> float:
    row = block[
        (block["p"] == p)
        & (block["s"] == s)
        & (np.isclose(block["delta"], delta))
        & (block["stat"] == stat)
    ]
    return float(row["power"].iloc[0])


def _latex_table(df: pd.DataFrame) -> str:
    indep = df[df["rho"] == 0]
    blocks = [("Independent coordinates", indep)]
    corr = df[df["rho"] != 0]
    if len(corr):
        rho = float(corr["rho"].iloc[0])
        blocks.append((f"AR(1) correlation {rho:.1f}", corr))
    show = list(STATS)
    lines = []
    for title, block in blocks:
        colspec = "ll" + "c" * len(show)
        lines.append(r"\begin{table}[p]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{" + title + r". Rejection rate at $\alpha=0.05$. "
            r"Each row is one covariate-shift cell. The six statistics share "
            r"the kernel, the bandwidth, and the permutations.}"
        )
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(r"\begin{tabular}{" + colspec + "}")
        lines.append(r"\toprule")
        lines.append(
            " & ".join([r"$p$", r"$s$, shift"] + [STAT_LABEL[s] for s in show]) + r" \\"
        )
        lines.append(r"\midrule")
        for _, key in _cell_keys(block).iterrows():
            cells = [str(int(key["p"])), f"{int(key['s'])}, {_fmt(key['delta'])}"]
            for stat in show:
                cells.append(_fmt(_power_at(block, key["p"], key["s"], key["delta"], stat)))
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}}")
        lines.append(r"\end{table}")
        lines.append(r"\clearpage")
    lines.extend(_paired_table(df))
    return "\n".join(lines) + "\n"


def _paired_table(df: pd.DataFrame) -> list[str]:
    """Discordant rejections against vanilla MMD, same Monte Carlo replicates."""
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\caption{Paired comparison with vanilla MMD on the same replicates. "
        r"``Only max'' counts replicates rejected by max LOCO and not by MMD. "
        r"``MMD, not max'' is the reverse. The last two columns are the same split "
        r"for the sum of positive LOCO scores.}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"$p$ & $s$, shift & MMD & max LOCO & only max & MMD, not max & pos.\ sum & only pos.\ & MMD, not pos.\ \\",
        r"\midrule",
    ]
    keys = (
        df.groupby(["rho", "p", "s", "delta"], sort=True)
        .size()
        .reset_index()[["rho", "p", "s", "delta"]]
    )
    for _, key in keys.iterrows():
        sub = df[
            np.isclose(df["rho"], key["rho"])
            & (df["p"] == key["p"])
            & (df["s"] == key["s"])
            & np.isclose(df["delta"], key["delta"])
        ]
        mmd = sub[sub["stat"] == "mmd"].iloc[0]
        mx = sub[sub["stat"] == "mmd_maxloco"].iloc[0]
        ps = sub[sub["stat"] == "mmd_posloco"].iloc[0]
        tag = f"{int(key['s'])}, {_fmt(key['delta'])}"
        if key["rho"] != 0:
            tag += rf", $\rho={key['rho']:.1f}$"
        lines.append(
            f"{int(key['p'])} & {tag} & {_fmt(mmd['power'])} & {_fmt(mx['power'])} & "
            f"{int(mx['n_only_this'])} & {int(mx['n_only_mmd'])} & {_fmt(ps['power'])} & "
            f"{int(ps['n_only_this'])} & {int(ps['n_only_mmd'])} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        r"\clearpage",
    ])
    return lines


def _note(df: pd.DataFrame) -> str:
    n = int(df["n"].iloc[0])
    reps = int(df["reps"].iloc[0])
    n_perm = int(df["n_perm"].iloc[0])
    return rf"""\documentclass{{article}}
\usepackage{{booktabs,geometry,graphicx,amsmath}}
\geometry{{margin=1in}}
\title{{A side check: MMD-LOCO against vanilla MMD\\on a covariate shift}}
\author{{}}
\date{{}}
\begin{{document}}
\maketitle

This note is not the FSDS horse-race. It only asks whether the leave-one-coordinate
MMD scores, turned into a single permutation statistic, detect a mean shift in $X$
more often than the joint MMD on the same kernel.

The kernel is $\exp(-\gamma\|x-y\|^2)$ with $\gamma$ equal to one over the median
pooled squared distance. $\gamma$ is computed once on the observed pool and held
fixed. Dropping coordinate $j$ subtracts that coordinate's squared differences from
the three distance matrices, which is the joint kernel on the remaining coordinates
at the same $\gamma$. The vector of interest is $\mathrm{{MMD}}^2-\mathrm{{MMD}}^2_{{-j}}$.
Two scalars are tested: the maximum coordinate, and the sum of the positive coordinates.
The baseline is the unbiased $\mathrm{{MMD}}^2$ itself. A normalized version,
$\mathrm{{NAMMD}}^2=\mathrm{{MMD}}^2_u/\sqrt{{\bar k_{{xx}}\bar k_{{yy}}}}$, is carried along
because the same drop applies to it. Each test uses the same label permutations.
The $p$-value is $(1+\#\{{T_b\ge T\}}) / (B+1)$.

Each batch has $n={n}$ reference rows and $n$ new rows. Coordinates are standard
normal, independent in the main grid, with an AR(1) check beside it. The new batch
adds a constant to the first $s$ coordinates. $B={n_perm}$ permutations and
${reps}$ Monte Carlo replicates. The dashed line in the figure is $\alpha=0.05$.
ECMMD is omitted: it needs a response and is a conditional discrepancy, not a test of $P(X)$.

\input{{power_table.tex}}

\begin{{figure}}[p]
\centering
\includegraphics[width=\textwidth]{{power.png}}
\caption{{Independent-coordinate grid. Curves are rejection rates. Bands are
pointwise $1.96$ Monte Carlo standard errors. The horizontal line is $0.05$.}}
\end{{figure}}

\end{{document}}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "scan", "full"), default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default="results/mmd_loco_power")
    args = parser.parse_args()
    if args.mode == "smoke":
        smoke()
        return

    if args.mode == "scan":
        grid = [
            dict(n=120, p=30, s=1, delta=d, rho=0.0, reps=12, n_perm=49)
            for d in (0.0, 0.5, 1.0, 1.5)
        ] + [
            dict(n=120, p=30, s=30, delta=d, rho=0.0, reps=12, n_perm=49)
            for d in (0.0, 0.3, 0.6)
        ]
    else:
        grid = _full_grid()

    records = []
    t0 = time.perf_counter()
    for i, cell in enumerate(grid, start=1):
        c0 = time.perf_counter()
        rows = run_cell(
            cell["n"], cell["p"], cell["s"], cell["delta"], cell["rho"],
            cell["reps"], cell["n_perm"], seed0=20261018 + 1000 * i, workers=args.workers,
        )
        meta = {k: cell[k] for k in ("n", "p", "s", "delta", "rho", "reps", "n_perm")}
        records.extend(summarize(rows, meta))
        spent = time.perf_counter() - c0
        brief = ", ".join(
            f"{r['stat']}={r['power']:.2f}"
            for r in records[-len(STATS):]
            if r["stat"] in ("mmd", "mmd_maxloco", "mmd_posloco")
        )
        print(
            f"[{i}/{len(grid)}] p={cell['p']} s={cell['s']} d={cell['delta']} "
            f"rho={cell['rho']} {brief} ({spent:.1f}s)",
            flush=True,
        )
    df = pd.DataFrame(records)
    paths = write_outputs(df, args.out)
    print(f"wrote {paths} in {time.perf_counter() - t0:.1f}s")


def _full_grid():
    """Deltas sit on the transition seen in a 12-replicate scan at p=30.

    Sparse cells need a larger shift before the joint MMD moves. Dense cells
    move earlier, so their grid is compressed toward zero.
    """
    n, reps, n_perm = 120, 80, 199
    specs = []
    plan = {
        20: {
            1: (0.0, 0.20, 0.35, 0.50, 0.80),
            4: (0.0, 0.20, 0.35, 0.50, 0.75),
            20: (0.0, 0.10, 0.20, 0.35, 0.50),
        },
        50: {
            1: (0.0, 0.40, 0.60, 0.85, 1.15),
            4: (0.0, 0.30, 0.50, 0.75, 1.00),
            50: (0.0, 0.10, 0.18, 0.30, 0.45),
        },
    }
    for p, by_s in plan.items():
        for s, deltas in by_s.items():
            for delta in deltas:
                specs.append(dict(n=n, p=p, s=s, delta=delta, rho=0.0, reps=reps, n_perm=n_perm))
    for s, deltas in ((1, (0.0, 0.35, 0.60)), (20, (0.0, 0.20, 0.40))):
        for delta in deltas:
            specs.append(dict(n=n, p=20, s=s, delta=delta, rho=0.5, reps=reps, n_perm=n_perm))
    return specs


if __name__ == "__main__":
    main()
