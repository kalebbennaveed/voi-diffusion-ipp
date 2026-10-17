"""Replay retained long-run observations to test numerical stability, without MBD.

The legacy failure fixtures and recordings are never overwritten. Small posterior
changes are expected from stable algebra; this is not a new physical mission.
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import yaml

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.wind.poam import POAM

ROOT = Path(__file__).resolve().parents[1]


def replay(path):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(path.with_suffix(".json").read_text())["metadata"]
    cfg = Config(**meta["config"])
    env, template, _, _, _ = make_two_region(
        yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text()), cfg
    )
    model = POAM.from_data(
        data["initial_observation_positions"],
        data["initial_observations"],
        jax.random.PRNGKey(600000),
        len(template.z),
        noise_std=cfg.observation_std,
        prior_std=cfg.wind_std,
        spectral_count=cfg.poam_spectral_nodes,
        prior_support=template.prior_support,
    )
    root = jax.random.PRNGKey(meta["seed"])
    rows = []
    for i in range(len(data["observations"]) // cfg.stage_steps):
        sl = slice(i * cfg.stage_steps, (i + 1) * cfg.stage_steps)
        losses = model.update(
            data["observation_positions"][sl],
            data["observations"][sl],
            jax.random.fold_in(root, 500000 + i),
            train_steps=cfg.poam_train_steps,
        )
        mu, cov = model.predict(env.cells)
        finite = all(
            np.isfinite(np.asarray(v)).all()
            for v in jax.tree.leaves(
                (model.parameters, model.mean, model.covariance, model.root_B, losses, mu, cov)
            )
        )
        row = {
            "period": i,
            "finite": finite,
            "min_variational_eigenvalue": float(
                np.linalg.eigvalsh(np.asarray(model.covariance, dtype=float)).min()
            ),
            "wind_rmse": float(np.sqrt(np.mean((np.asarray(mu) - data["physical_wind"]) ** 2))),
        }
        rows.append(row)
        if not finite:
            raise FloatingPointError(f"Nonfinite POAM at period {i}")
        if i % 20 == 0 or i == len(data["observations"]) // cfg.stage_steps - 1:
            print(path.stem, row, flush=True)
    return {"input": str(path.relative_to(ROOT)), "all_finite": True, "periods": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = [replay(p.resolve()) for p in args.paths]
    args.output.write_text(json.dumps(results, indent=2) + "\n")
