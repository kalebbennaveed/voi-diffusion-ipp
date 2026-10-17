"""Compare frozen-inducing forecasts against full POAM relocation on identical hypothetical data."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml

from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.experiments.two_region import make_two_region

ROOT = Path(__file__).resolve().parents[1]
spec = yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text())
base = load_config(ROOT / "configs/two_region_budgeted.yaml")
rows = []
for nx, ny in [(6, 4), (5, 5), (7, 5)]:
    cfg = replace(base, poam_inducing_x=nx, poam_inducing_y=ny)
    env, poam, x, q, truth = make_two_region(spec, cfg)
    pos = jnp.tile(x[:2], (8, 1))
    key = jax.random.PRNGKey(8101)
    poam.update(
        pos,
        truth.at(pos) + cfg.observation_std * jax.random.normal(key, (8, 2)),
        key,
        train_steps=10,
    )
    b = poam.planning_belief()
    for distance in [0.0, 0.5, 1.0]:
        point = -truth.tangent * distance
        pos = jnp.tile(point, (10, 1))
        y = b.mean_at(pos)
        frozen = b.condition(pos, y)
        moving = copy.deepcopy(poam)
        moving.update(pos, y, jax.random.PRNGKey(9320), train_steps=0, relocate=True)
        real = moving.planning_belief()
        row = {
            "grid": [nx, ny],
            "observed_distance_to_center": distance,
            "prior_sigma_center": float(jnp.sqrt(b.predict(env.probe)[1][0, 0, 0])),
            "fixed_sigma_center": float(jnp.sqrt(frozen.predict(env.probe)[1][0, 0, 0])),
            "relocated_sigma_center": float(jnp.sqrt(real.predict(env.probe)[1][0, 0, 0])),
            "fixed_sigma_observed": float(jnp.sqrt(frozen.predict(point)[1][0, 0, 0])),
            "relocated_sigma_observed": float(jnp.sqrt(real.predict(point)[1][0, 0, 0])),
            "minimum_inducing_distance_to_center": float(
                jnp.linalg.norm(b.basis.centers, axis=1).min()
            ),
        }
        rows.append(row)
        print(row, flush=True)
(ROOT / "docs/budgeted_planner/learning_resolution.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
