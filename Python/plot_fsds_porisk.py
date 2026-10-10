"""Figures and LaTeX tables for the FSDS pseudo-outcome horse-race."""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results", "fsds_porisk_poc")
FIG = os.path.join(OUT, "figures")

DS_ORDER = [
    "gaussian30",
    "gaussian12",
    "student_t5",
    "ar1_gaussian",
    "california",
    "covertype",
]
DS_TEX = {
    "gaussian30": r"Gaussian $d=30$",
    "gaussian12": r"Gaussian $d=12$",
    "student_t5": r"Student-$t_5$",
    "ar1_gaussian": r"AR(1) Gaussian",
    "california": "California housing",
    "covertype": "Covertype",
}
DS_SHORT = {
    "gaussian30": "Gauss30",
    "gaussian12": "Gauss12",
    "student_t5": "t5",
    "ar1_gaussian": "AR1",
    "california": "Calif.",
    "covertype": "Cover",
}
FAMILIES = ("relocate", "flip", "die", "nonlin", "noise")
SCENARIOS = [f"{path}_{fam}" for fam in FAMILIES for path in ("gradual", "drastic")]
SCEN_TEX = {
    "gradual_relocate": "Gradual relocation",
    "drastic_relocate": "Drastic relocation",
    "gradual_flip": "Gradual sign flip",
    "drastic_flip": "Drastic sign flip",
    "gradual_die": "Gradual dying block",
    "drastic_die": "Drastic dying block",
    "gradual_nonlin": "Gradual nonlinear add-on",
    "drastic_nonlin": "Drastic nonlinear add-on",
    "gradual_noise": "Gradual noise growth",
    "drastic_noise": "Drastic noise growth",
    "single": "Single block move",
    "double": "Two block moves",
}

SCORE_POLICIES = [
    "UCB", "DUCB", "TS", "EG05", "EG10", "EG15",
    "UCB-H0.5", "UCB-H1", "UCB-H2",
    "TS-H0.5", "TS-H1", "TS-H2",
    "UCB-L5", "TS-L5", "EG10-H1",
]
POL_TEX = {
    "UCB": "UCB",
    "DUCB": "dUCB",
    "TS": "TS",
    "EG05": r"$\varepsilon$.05",
    "EG10": r"$\varepsilon$.10",
    "EG15": r"$\varepsilon$.15",
    "UCB-H0.5": r"UCB$_{\eta=.5}$",
    "UCB-H1": r"UCB$_{\eta=1}$",
    "UCB-H2": r"UCB$_{\eta=2}$",
    "TS-H0.5": r"TS$_{\eta=.5}$",
    "TS-H1": r"TS$_{\eta=1}$",
    "TS-H2": r"TS$_{\eta=2}$",
    "UCB-L5": r"UCB$_{\lambda=5}$",
    "TS-L5": r"TS$_{\lambda=5}$",
    "EG10-H1": r"$\varepsilon$.10$_{\eta=1}$",
    "UCB-H1-on": r"UCB$_{\eta=1}$, on-policy",
    "TS-H1-on": r"TS$_{\eta=1}$, on-policy",
    "UCB-H1-fit": r"UCB$_{\eta=1}$, fitted",
    "TS-H1-fit": r"TS$_{\eta=1}$, fitted",
}
CURVE_POLICIES = ("UCB", "DUCB", "TS", "EG10", "UCB-H1", "TS-H1", "UCB-H2", "EG10-H1")
COLORS = {
    "UCB": "#4C78A8",
    "DUCB": "#72B7B2",
    "TS": "#F58518",
    "EG10": "#E45756",
    "UCB-H1": "#54A24B",
    "TS-H1": "#B279A2",
    "UCB-H2": "#1F7A1F",
    "EG10-H1": "#F2CF5B",
    "UCB-H1-on": "#88D27A",
    "UCB-H1-fit": "#2E5A27",
}
ARM_COLORS = ("#4C78A8", "#54A24B", "#F58518", "#E45756")


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row, key):
    return float(row[key])


def summarize_score(rows):
    bucket = defaultdict(list)
    for row in rows:
        bucket[(row["dataset"], row["scenario"], row["policy"])].append(row)
    out = {}
    for key, group in bucket.items():
        rec = {}
        for name in ("early", "ramp", "post", "hit_early", "hit_ramp", "hit_post", "lam", "sigma"):
            arr = np.array([_f(r, name) for r in group])
            rec[name] = (float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0))
        out[key] = rec
    return out


def summarize_policy(rows):
    bucket = defaultdict(list)
    for row in rows:
        bucket[(row["dataset"], row["scenario"], row["policy"])].append(row)
    out = {}
    for key, group in bucket.items():
        rec = {}
        for name in ("phase0", "phase1", "phase2", "hit0", "hit1", "hit2", "lam", "sigma"):
            arr = np.array([_f(r, name) for r in group])
            rec[name] = (float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0))
        rec["n2"] = int(float(group[0]["n2"]))
        out[key] = rec
    return out


def _cell(pair, digits=2):
    mean, sd = pair
    return f"{mean:.{digits}f} ({sd:.{digits}f})"


def _tex_table(caption, label, col_heads, row_heads, cells, note):
    spec = "l" + "c" * len(col_heads)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join([""] + col_heads) + r" \\",
        r"\midrule",
    ]
    for head, row in zip(row_heads, cells):
        lines.append(head + " & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\parbox{{\textwidth}}{{\scriptsize {note}}}", r"\end{table}", ""]
    return "\n".join(lines)


def _datasets(summary):
    present = {k[0] for k in summary}
    return [d for d in DS_ORDER if d in present]


def write_score_tables(summary, share_rows):
    datasets = _datasets(summary)
    chunks = []
    # Headline: median across datasets of the seed-mean post excess.
    head_policies = ["UCB", "DUCB", "TS", "EG10", "UCB-H1", "UCB-H2", "TS-H1", "TS-H2", "EG10-H1", "UCB-L5"]
    head_cells = []
    winners = []
    for scen in SCENARIOS:
        row = []
        scores = {}
        for policy in SCORE_POLICIES:
            vals = [summary[(ds, scen, policy)]["post"][0] for ds in datasets]
            scores[policy] = float(np.median(vals))
        best = min(scores, key=scores.get)
        winners.append((scen, best, scores[best], scores["UCB"], scores["EG10"], scores["DUCB"], scores["TS"]))
        for policy in head_policies:
            row.append(f"{scores[policy]:.2f}")
        row.append(POL_TEX[best])
        head_cells.append(row)
    chunks.append(_tex_table(
        "Median across covariate pools of the mean post-shift excess MSE. "
        "Each pool contributes the average over four seeds. Lower is better. "
        "The last column is the lowest median among all fifteen rules, not only the columns shown.",
        "tab:headline",
        [POL_TEX[p] for p in head_policies] + ["best"],
        [SCEN_TEX[s] for s in SCENARIOS],
        head_cells,
        r"$\eta$ multiplies the reference-window standard deviation of arm rewards. "
        r"$\lambda=5$ is an absolute penalty on the share, not scaled.",
    ))

    metrics = [
        ("post", "Post-shift excess MSE", "tab:post", "Sum of per-round gaps to the best scorecard once the mixing weight reaches $1/2$."),
        ("early", "Excess MSE before the shift starts", "tab:early", "Rounds with mixing weight exactly zero. The share is still refreshed from round 16, so a tax can already be active on a gradual path's later reference window."),
        ("hit_post", "Post-shift hit rate", "tab:hit", "Fraction of post-shift rounds on the single best scorecard. A blend can have low excess and a low hit rate."),
    ]
    for key, title, label, note in metrics:
        for scen in SCENARIOS:
            cells = []
            for policy in SCORE_POLICIES:
                cells.append([_cell(summary[(ds, scen, policy)][key], 2 if key != "hit_post" else 2) for ds in datasets])
            chunks.append(_tex_table(
                f"{title}: {SCEN_TEX[scen]}. Mean over four seeds, seed standard deviation in parentheses.",
                f"{label}:{scen}",
                [DS_TEX[d] for d in datasets],
                [POL_TEX[p] for p in SCORE_POLICIES],
                cells,
                note,
            ))
        # ramp only meaningful for gradual
        if key == "post":
            for scen in SCENARIOS:
                if scen.startswith("drastic"):
                    continue
                cells = []
                for policy in SCORE_POLICIES:
                    cells.append([_cell(summary[(ds, scen, policy)]["ramp"]) for ds in datasets])
                chunks.append(_tex_table(
                    f"Ramp excess MSE: {SCEN_TEX[scen]}. Rounds with mixing weight in $(0,1/2)$.",
                    f"tab:ramp:{scen}",
                    [DS_TEX[d] for d in datasets],
                    [POL_TEX[p] for p in SCORE_POLICIES],
                    cells,
                    "The ramp is where a historical mean is already stale and the fresh arm is not yet dominant.",
                ))

    # shares
    bucket = defaultdict(list)
    arm_names = {}
    for row in share_rows:
        if int(row["post"]) != 1:
            continue
        key = (row["dataset"], row["scenario"], row["seed"])
        bucket[key].append([float(row[f"arm{i}"]) for i in range(4)])
        arm_names[(row["dataset"], row["scenario"])] = [row[f"name{i}"] for i in range(4)]
    for scen in SCENARIOS:
        names = arm_names[(datasets[0], scen)]
        cells = []
        for ds in datasets:
            seed_means = []
            seeds = {k[2] for k in bucket if k[0] == ds and k[1] == scen}
            for seed in seeds:
                arr = np.array(bucket[(ds, scen, seed)])
                seed_means.append(arr.mean(axis=0))
            mat = np.array(seed_means)
            cells.append([f"{mat[:, i].mean():.3f} ({mat[:, i].std(ddof=1):.3f})" for i in range(4)])
        chunks.append(_tex_table(
            f"Post-shift PO-risk shares: {SCEN_TEX[scen]}. "
            f"Arms are {', '.join(names)}. Mean of the refresh-time shares on post-shift rounds, then mean and seed sd across four seeds.",
            f"tab:share:{scen}",
            [n.replace("_", r"\_") for n in names],
            [DS_TEX[d] for d in datasets],
            cells,
            "The share is the residual pseudo-outcome risk of that score, divided by the sum across the four scores. It does not depend on which arm was served.",
        ))

    # scaled lambda actually used
    cells = []
    for ds in datasets:
        # sigma is constant across policies; average over scenarios and seeds via the summary of UCB
        sigs = [summary[(ds, scen, "UCB")]["sigma"][0] for scen in SCENARIOS]
        lams = [summary[(ds, scen, "UCB-H1")]["lam"][0] for scen in SCENARIOS]
        cells.append([f"{np.mean(sigs):.2f}", f"{np.mean(lams):.2f}", f"{np.min(lams):.2f}", f"{np.max(lams):.2f}"])
    chunks.append(_tex_table(
        r"Reward scale $\sigma$ on the reference window, and the resulting penalty multiplier $\eta\sigma$ at $\eta=1$.",
        "tab:sigma",
        [r"mean $\sigma$", r"mean $\eta\sigma$", r"min $\eta\sigma$", r"max $\eta\sigma$"],
        [DS_TEX[d] for d in datasets],
        cells,
        r"$\sigma$ is the standard deviation of the four arm rewards on the first six batches. The index subtracts $\eta\sigma w_k$.",
    ))
    path = os.path.join(OUT, "scorecard_tables.tex")
    with open(path, "w") as f:
        f.write("\n".join(chunks))
    return winners, datasets


def write_policy_tables(summary):
    datasets = _datasets(summary)
    policies = [
        "UCB", "DUCB", "TS", "EG05", "EG10", "EG15",
        "UCB-H1", "UCB-H2", "TS-H1", "TS-H2", "EG10-H1",
        "UCB-H1-on", "TS-H1-on", "UCB-H1-fit", "TS-H1-fit",
    ]
    chunks = []
    for scen in ("single", "double"):
        phases = [("phase0", "hit0", "Phase 0, before the first block has majority weight")]
        phases.append(("phase1", "hit1", "Phase 1, first moved block has majority weight"))
        if scen == "double":
            phases.append(("phase2", "hit2", "Phase 2, second moved block has majority weight"))
        for key, hit_key, title in phases:
            cells = []
            hit_cells = []
            for policy in policies:
                cells.append([_cell(summary[(ds, scen, policy)][key]) for ds in datasets])
                hit_cells.append([_cell(summary[(ds, scen, policy)][hit_key]) for ds in datasets])
            chunks.append(_tex_table(
                f"Policy horse-race excess return: {SCEN_TEX[scen]}, {title}. "
                "Sum of per-round gaps to the best policy. Mean over four seeds.",
                f"tab:pol:{scen}:{key}",
                [DS_TEX[d] for d in datasets],
                [POL_TEX[p] for p in policies],
                cells,
                "Oracle rows score every policy on every round. On-policy rows only retain the served policy. Fitted rows imagine every policy from a least-squares reward fit on the served transitions.",
            ))
            chunks.append(_tex_table(
                f"Policy horse-race hit rate: {SCEN_TEX[scen]}, {title}.",
                f"tab:polhit:{scen}:{hit_key}",
                [DS_TEX[d] for d in datasets],
                [POL_TEX[p] for p in policies],
                hit_cells,
                "Hit rate is the fraction of rounds whose served policy has the highest mean return on that round.",
            ))
    path = os.path.join(OUT, "policy_tables.tex")
    with open(path, "w") as f:
        f.write("\n".join(chunks))


def _style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
    })


def plot_score_curves(curve_rows, datasets):
    _style()
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in curve_rows:
        grouped[row["scenario"]][row["dataset"]][row["policy"]].append(
            (int(row["seed"]), int(row["t"]), float(row["cum"]))
        )
    for scen in SCENARIOS:
        if scen not in grouped:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2), sharex=True)
        first_post = None
        for ax, ds in zip(axes.ravel(), datasets):
            by_policy = grouped[scen][ds]
            for policy in CURVE_POLICIES:
                seed_map = defaultdict(list)
                for seed, t, cum in by_policy[policy]:
                    seed_map[seed].append((t, cum))
                mats = []
                for seed, pairs in seed_map.items():
                    pairs.sort()
                    mats.append([c for _, c in pairs])
                mat = np.array(mats)
                mean = mat.mean(axis=0)
                sd = mat.std(axis=0, ddof=1) if len(mat) > 1 else np.zeros_like(mean)
                tt = np.arange(len(mean))
                ax.plot(tt, mean, color=COLORS[policy], lw=1.6, label=policy)
                ax.fill_between(tt, mean - sd, mean + sd, color=COLORS[policy], alpha=0.12)
            # post start from any policy's rows is not stored; use the scenario rule
            cut = 40 if scen.startswith("drastic") else 50
            ax.axvline(cut, color="0.45", ls="--", lw=0.8)
            ax.set_title(DS_SHORT[ds])
            ax.set_xlabel("round")
            ax.set_ylabel("cumulative excess MSE")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(SCEN_TEX[scen] + "  ·  cumulative excess, mean ± seed sd", y=1.08, fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"cum_{scen}.png"), bbox_inches="tight")
        plt.close(fig)


def plot_shares(share_rows, datasets):
    _style()
    for scen in SCENARIOS:
        rows = [r for r in share_rows if r["scenario"] == scen]
        if not rows:
            continue
        names = [rows[0][f"name{i}"] for i in range(4)]
        fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), sharex=True, sharey=True)
        for ax, ds in zip(axes.ravel(), datasets):
            sub = [r for r in rows if r["dataset"] == ds]
            ts = sorted({int(r["t"]) for r in sub})
            for i, name in enumerate(names):
                means, sds = [], []
                for t in ts:
                    vals = [float(r[f"arm{i}"]) for r in sub if int(r["t"]) == t]
                    means.append(np.mean(vals))
                    sds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                means = np.array(means)
                sds = np.array(sds)
                ax.plot(ts, means, color=ARM_COLORS[i], lw=1.7, marker="o", ms=3.5, label=name)
                ax.fill_between(ts, means - sds, means + sds, color=ARM_COLORS[i], alpha=0.12)
            cut = 40 if scen.startswith("drastic") else 50
            ax.axvline(cut, color="0.45", ls="--", lw=0.8)
            ax.set_ylim(0, 1)
            ax.set_title(DS_SHORT[ds])
            ax.set_xlabel("round")
            ax.set_ylabel("PO-risk share")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(SCEN_TEX[scen] + "  ·  where the screening mass sits", y=1.08, fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"shares_{scen}.png"), bbox_inches="tight")
        plt.close(fig)


def plot_bars(summary, datasets):
    _style()
    for scen in SCENARIOS:
        med = []
        for policy in SCORE_POLICIES:
            vals = [summary[(ds, scen, policy)]["post"][0] for ds in datasets]
            med.append(float(np.median(vals)))
        order = np.argsort(med)
        fig, ax = plt.subplots(figsize=(8.2, 5.4))
        y = np.arange(len(order))
        ax.barh(y, [med[i] for i in order], color="#4C78A8")
        ax.set_yticks(y)
        ax.set_yticklabels([SCORE_POLICIES[i] for i in order])
        ax.set_xlabel("median post-shift excess MSE across pools")
        ax.set_title(SCEN_TEX[scen])
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"bar_{scen}.png"), bbox_inches="tight")
        plt.close(fig)


def plot_eta(summary, datasets):
    _style()
    etas = [0.0, 0.5, 1.0, 2.0]
    ucb_names = ["UCB", "UCB-H0.5", "UCB-H1", "UCB-H2"]
    ts_names = ["TS", "TS-H0.5", "TS-H1", "TS-H2"]
    fig, axes = plt.subplots(2, 5, figsize=(14.5, 6.2), sharex=True)
    for ax, scen in zip(axes.ravel(), SCENARIOS):
        def series(names):
            ys = []
            for name in names:
                vals = [summary[(ds, scen, name)]["post"][0] for ds in datasets]
                ys.append(float(np.median(vals)))
            return ys
        ax.plot(etas, series(ucb_names), marker="o", color="#54A24B", lw=1.8, label="UCB + share")
        ax.plot(etas, series(ts_names), marker="o", color="#B279A2", lw=1.8, label="TS + share")
        eg = float(np.median([summary[(ds, scen, "EG10")]["post"][0] for ds in datasets]))
        ducb = float(np.median([summary[(ds, scen, "DUCB")]["post"][0] for ds in datasets]))
        ax.axhline(eg, color="#E45756", ls="--", lw=1.0, label=r"ε = 0.10")
        ax.axhline(ducb, color="#72B7B2", ls=":", lw=1.2, label="dUCB")
        ax.set_title(SCEN_TEX[scen], fontsize=9)
        ax.set_xlabel(r"scale $\eta$")
        ax.set_ylabel("median post excess")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Modest scaled taxes versus uniform exploration", y=1.10, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "eta_profile.png"), bbox_inches="tight")
    plt.close(fig)


def plot_policy(curve_rows, summary, datasets):
    _style()
    show = ("UCB", "DUCB", "TS", "EG10", "UCB-H1", "TS-H1", "UCB-H2", "UCB-H1-on", "UCB-H1-fit")
    colors = {
        "UCB": "#4C78A8", "DUCB": "#72B7B2", "TS": "#F58518", "EG10": "#E45756",
        "UCB-H1": "#54A24B", "TS-H1": "#B279A2", "UCB-H2": "#1F7A1F",
        "UCB-H1-on": "#88D27A", "UCB-H1-fit": "#2E5A27",
    }
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in curve_rows:
        grouped[row["scenario"]][row["dataset"]][row["policy"]].append(
            (int(row["seed"]), int(row["t"]), float(row["cum"]))
        )
    for scen, cut in (("single", 50), ("double", 50)):
        fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2), sharex=False)
        for ax, ds in zip(axes.ravel(), datasets):
            for policy in show:
                seed_map = defaultdict(list)
                for seed, t, cum in grouped[scen][ds][policy]:
                    seed_map[seed].append((t, cum))
                mats = []
                for pairs in seed_map.values():
                    pairs.sort()
                    mats.append([c for _, c in pairs])
                mat = np.array(mats)
                mean = mat.mean(axis=0)
                tt = np.arange(len(mean))
                ax.plot(tt, mean, color=colors[policy], lw=1.5, label=policy)
            ax.axvline(cut, color="0.45", ls="--", lw=0.8)
            if scen == "double":
                ax.axvline(85, color="0.45", ls=":", lw=0.8)
            ax.set_title(DS_SHORT[ds])
            ax.set_xlabel("round")
            ax.set_ylabel("cumulative excess return")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
        fig.suptitle(SCEN_TEX[scen] + "  ·  policy horse-race", y=1.09, fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"policy_cum_{scen}.png"), bbox_inches="tight")
        plt.close(fig)

    policies = [
        "UCB", "DUCB", "TS", "EG10", "UCB-H1", "UCB-H2", "TS-H1", "TS-H2",
        "EG10-H1", "UCB-H1-on", "TS-H1-on", "UCB-H1-fit", "TS-H1-fit",
    ]
    for scen, phases in (("single", ("phase0", "phase1")), ("double", ("phase0", "phase1", "phase2"))):
        fig, axes = plt.subplots(1, len(phases), figsize=(4.4 * len(phases), 5.6), sharey=True)
        if len(phases) == 1:
            axes = [axes]
        for ax, phase in zip(axes, phases):
            med = []
            for policy in policies:
                vals = [summary[(ds, scen, policy)][phase][0] for ds in datasets]
                med.append(float(np.median(vals)))
            order = np.argsort(med)
            ax.barh(np.arange(len(order)), [med[i] for i in order], color="#4C78A8")
            ax.set_yticks(np.arange(len(order)))
            ax.set_yticklabels([policies[i] for i in order], fontsize=8)
            ax.set_xlabel("median excess return")
            ax.set_title(phase)
        fig.suptitle(SCEN_TEX[scen], y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"policy_bar_{scen}.png"), bbox_inches="tight")
        plt.close(fig)


def write_headline_txt(winners, path):
    lines = ["scenario best median ucb eg10 ducb ts removed_vs_ucb removed_vs_eg10"]
    for scen, best, med, ucb, eg, ducb, ts in winners:
        ru = (ucb - med) / ucb if ucb else float("nan")
        re = (eg - med) / eg if eg else float("nan")
        lines.append(f"{scen} {best} {med:.3f} {ucb:.3f} {eg:.3f} {ducb:.3f} {ts:.3f} {ru:.3f} {re:.3f}")
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    print(text)


def main():
    os.makedirs(FIG, exist_ok=True)
    score_rows = _read(os.path.join(OUT, "scorecard_detail.csv"))
    curve_rows = _read(os.path.join(OUT, "scorecard_curves.csv"))
    share_rows = _read(os.path.join(OUT, "scorecard_shares.csv"))
    summary = summarize_score(score_rows)
    winners, datasets = write_score_tables(summary, share_rows)
    plot_score_curves(curve_rows, datasets)
    plot_shares(share_rows, datasets)
    plot_bars(summary, datasets)
    plot_eta(summary, datasets)
    write_headline_txt(winners, os.path.join(OUT, "headline.txt"))
    pol_rows = _read(os.path.join(OUT, "policy_detail.csv"))
    pol_curves = _read(os.path.join(OUT, "policy_curves.csv"))
    pol_summary = summarize_policy(pol_rows)
    write_policy_tables(pol_summary)
    plot_policy(pol_curves, pol_summary, _datasets(pol_summary))
    print("figures and tables written", flush=True)


if __name__ == "__main__":
    main()
