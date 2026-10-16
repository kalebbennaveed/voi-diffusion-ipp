"""Paired small-budget evaluation. Use --research for the much larger profile."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from ..config import Config, config_dict
from .run import mission


def completed_run(path, cfg):
    """Resume only complete results with the requested physical/numerical setup."""
    path = Path(path)
    if not path.exists() or not path.with_suffix(".npz").exists():
        return False
    saved = json.loads(path.read_text())["config"]
    requested = config_dict(cfg)
    # These select a batch; they do not affect an individual mission.
    for key in ("seeds", "methods"):
        saved.pop(key, None)
        requested.pop(key, None)
    if saved != requested:
        raise ValueError(
            f"Configuration differs for {path}; use a new output directory or --overwrite"
        )
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/evaluation")
    p.add_argument("--seeds", nargs="+", type=int, default=[10, 11, 12])
    p.add_argument("--horizons", type=int, default=5)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["proposed", "passive", "weighted_ig", "heuristic", "cem", "mppi", "oracle"],
    )
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=["corridor", "irrelevant_uncertainty", "no_worthwhile_learning"],
    )
    p.add_argument("--ablations", action="store_true")
    p.add_argument("--research", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Recompute existing mission files")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = replace(
        Config(),
        diffusion_steps=4,
        proposals=12,
        inner_samples=4,
        backup_steps=10,
        backup_dt=0.192,
        mission_horizons=a.horizons,
        seeds=tuple(a.seeds),
        candidate_chunk=1,
    )
    if a.research:
        cfg = replace(
            cfg,
            diffusion_steps=24,
            proposals=96,
            worlds=12,
            inner_samples=24,
            backup_steps=24,
            backup_dt=0.08,
        )
    variants = [("main", cfg)]
    if a.ablations:
        variants = [
            ("no_measurements", replace(cfg, measurements=False)),
            ("shared_future", replace(cfg, shared_future=True)),
            ("independent_cells", replace(cfg, independent_cells=True)),
            ("terminal", replace(cfg, terminal_seconds=3.0)),
            ("short_probe", replace(cfg, stage_steps=6, mission_horizons=2 * a.horizons)),
            ("fast_spline", replace(cfg, spline="fast")),
        ]
    failures = []
    for label, base in variants:
        dest = out if label == "main" else out / label
        for scenario in a.scenarios:
            c = replace(base, scenario=scenario)
            for method in a.methods:
                for seed in a.seeds:
                    if not a.overwrite and completed_run(
                        dest / f"{scenario}_{method}_{seed}.json", c
                    ):
                        continue
                    try:
                        mission(c, method, seed, dest)
                    except Exception as e:  # noqa: BLE001 - retain each failed mission and continue paired trials
                        failure = {
                            "variant": label,
                            "scenario": scenario,
                            "method": method,
                            "seed": seed,
                            "error": repr(e),
                        }
                        failures.append(failure)
                        print("FAILED", failure, flush=True)
                        (out / "failures.json").write_text(json.dumps(failures, indent=2))
    (out / "failures.json").write_text(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
