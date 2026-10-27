"""Run the 20-site primary-driver experiment and write the tables."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Keep numeric libraries from spawning one thread per core on top of the
# site loop. The loop itself is serial so the run stays reproducible.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from federated_vimp.aggregate import render_executive
from federated_vimp.experiment import (
    ablate_z_scaling,
    run_replicate,
    save_scatter,
    summarize_metrics,
)
from federated_vimp.statistics import StatConfig


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value
    if hasattr(value, "item"):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--n-cap", type=int, default=320)
    parser.add_argument("--boot", type=int, default=0)
    parser.add_argument("--out", type=str, default=str(Path(__file__).with_name("results")))
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = StatConfig()
    metric_rows = []
    showcase = None
    for rep in range(args.reps):
        print("replicate {0}/{1}".format(rep + 1, args.reps), flush=True)
        result = run_replicate(
            seed=args.seed + 100 * rep,
            config=config,
            n_history=args.history,
            n_cap=args.n_cap,
            n_boot=args.boot if rep == 0 else 0,
            with_pooled=True,
            progress=True,
        )
        row = dict(result["metrics"])
        row["rep"] = rep
        row["seed"] = args.seed + 100 * rep
        metric_rows.append(row)
        print(
            "  routes primary={c}/{x} local={loc} breaker={b} large={g} type={t:.2f} ({sec:.0f}s)".format(
                c=row["route_concept_primary"],
                x=row["route_cov_primary"],
                loc=row["route_local"],
                b=row["route_breaker"],
                g=row["route_large_mild"],
                t=row["type_accuracy"],
                sec=row["seconds"],
            ),
            flush=True,
        )
        if rep == 0:
            showcase = result

    summary = summarize_metrics(metric_rows)
    if not args.skip_ablation:
        print("z-scaling ablation", flush=True)
        summary["z_scaling_ablation"] = ablate_z_scaling(config, n_history=min(12, args.history))

    (out / "mc_summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    (out / "showcase_metrics.json").write_text(json.dumps(_jsonable(metric_rows[0]), indent=2), encoding="utf-8")
    (out / "executive_summary.md").write_text(render_executive(showcase["report"]), encoding="utf-8")
    save_scatter(showcase["report"], str(out / "scatter_routes.png"))

    site_path = out / "showcase_sites.csv"
    with site_path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(showcase["report"]["sites"][0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for site in showcase["report"]["sites"]:
            writer.writerow(site)

    metric_path = out / "mc_metrics.csv"
    fieldnames = []
    for row in metric_rows:
        for key in row:
            if key not in fieldnames and isinstance(row[key], (int, float, str, bool)):
                fieldnames.append(key)
    with metric_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print("wrote {0}".format(out), flush=True)


if __name__ == "__main__":
    main()
