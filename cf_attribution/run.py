"""Run the counterfactual-attribution protocol on synthetic and real panels.

Examples
--------
python -m cf_attribution.run --mode fast --out artifacts/cf_attribution_fast
python -m cf_attribution.run --mode full --out artifacts/cf_attribution
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cf_attribution.dgp import make_synthetic_panel
from cf_attribution.engine import Config, run_panel
from cf_attribution.realdata import load_beijing, load_covid
from cf_attribution.report import write_report


def build_config(mode: str) -> Config:
    if mode == "fast":
        return Config(
            ddpm_epochs=8,
            flow_epochs=8,
            diff_steps=6,
            hidden=48,
            gbr_iter=70,
            n_boot=30,
            rff=48,
            batch=128,
            panels=("long",),
        )
    return Config(
        ddpm_epochs=40,
        flow_epochs=40,
        diff_steps=16,
        hidden=96,
        gbr_iter=80,
        n_boot=200,
        rff=96,
        batch=128,
        panels=("long", "short", "beijing", "covid"),
    )


def load_panels(names: tuple[str, ...]):
    panels = []
    for name in names:
        if name == "long":
            panels.append(make_synthetic_panel("long", seed=2026))
        elif name == "short":
            panels.append(make_synthetic_panel("short", seed=2027))
        elif name == "beijing":
            panel = load_beijing(peak_lag=10)
            if panel is None:
                print("beijing panel not found, skipping")
            else:
                panels.append(panel)
        elif name == "covid":
            panel = load_covid(peak_lag=10)
            if panel is None:
                print("covid panel not found, skipping")
            else:
                panels.append(panel)
        else:
            raise SystemExit(f"unknown panel {name}")
    return panels


def main() -> None:
    parser = argparse.ArgumentParser(description="Panel counterfactual attribution")
    parser.add_argument("--mode", choices=("fast", "full"), default="full")
    parser.add_argument("--out", default="artifacts/cf_attribution")
    parser.add_argument("--panels", default="", help="Comma-separated override: long,short,beijing,covid")
    args = parser.parse_args()
    cfg = build_config(args.mode)
    if args.panels:
        cfg.panels = tuple(p.strip() for p in args.panels.split(",") if p.strip())
    panels = load_panels(cfg.panels)
    results = [run_panel(panel, cfg) for panel in panels]
    path = write_report(results, Path(args.out))
    print(f"\nreport: {path}")


if __name__ == "__main__":
    main()
