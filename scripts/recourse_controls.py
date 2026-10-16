"""Reproduce development-only ablations and the no-measurement negative control."""

import argparse
from dataclasses import replace

from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.experiments.run import mission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-measurements", action="store_true")
    args = parser.parse_args()
    base = load_config("configs/recourse_dev.yaml")
    if args.no_measurements:
        cfg = replace(base, measurements=False)
        for method in ("proposed", "current_belief"):
            mission(cfg, method, 20, "results/recourse_no_measurements")
        return
    variants = {
        "threshold_feedback": replace(base, branch_rule="crosswind"),
        "task_openloop": replace(base, control_representation="acceleration"),
        "threshold_openloop": replace(
            base, control_representation="acceleration", branch_rule="crosswind"
        ),
    }
    for name, cfg in variants.items():
        for seed in base.seeds:
            mission(cfg, "proposed", seed, f"results/recourse_ablations/{name}")


if __name__ == "__main__":
    main()
