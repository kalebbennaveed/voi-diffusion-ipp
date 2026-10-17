"""Explicit known-wind reference at the recorded physical state, not a deployable policy."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.planning.budgeted import evaluate, warm_starts
from voi_diffusion_ipp.wind.oracle import KnownBasis, KnownCache, KnownWind

ROOT = Path(__file__).resolve().parents[1]
cfg = load_config(ROOT / "configs/two_region_budgeted.yaml")
spec = yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text())
env, poam, x, q, truth = make_two_region(spec, cfg)
root = jax.random.PRNGKey(8101)
pos = jnp.tile(x[:2], (8, 1))
poam.update(
    pos, truth.at(pos) + cfg.observation_std * jax.random.normal(root, (8, 2)), root, train_steps=10
)
a = np.load(ROOT / "results/budgeted_missions/dev_01/proposed.npz")
period = 8
for i in range(period):
    poam.update(
        a["x"][i * 20 + 1 : (i + 1) * 20 + 1, :2],
        a["observations"][i * 20 : (i + 1) * 20],
        jax.random.fold_in(root, 500000 + i),
        train_steps=10,
    )
b = poam.planning_belief()
np.testing.assert_allclose(b.mean_at(env.cells), a["mean"][period], atol=1e-5)
x, q = jnp.asarray(a["x"][period * 20]), jnp.asarray(a["q"][period * 20])
keys = jax.random.split(jax.random.fold_in(root, 100 + period), 4)
eps = jax.random.normal(keys[0], (cfg.worlds + cfg.inner_samples, *b.sample_shape))
noise = jax.random.normal(keys[1], (cfg.worlds, cfg.stage_steps + cfg.future_steps, 2))
b = KnownWind(KnownBasis(truth))
cache = KnownCache(jnp.ones((cfg.worlds + cfg.inner_samples, 1, 2)))
slow = warm_starts(x, env, cfg)
fast = []
for gi, goal in enumerate([env.roi_a, env.roi_b, x[:2]]):
    delta = goal - x[:2]
    direction = delta / jnp.maximum(jnp.linalg.norm(delta), 0.01)
    normal = jnp.array([-direction[1], direction[0]])
    for offset in [0.0, -1.5, 1.5] if gi < 2 else [0.0]:
        fraction = (
            jnp.array([0.7, 1.0, 1.0, 1.0]) if offset == 0 else jnp.array([0.5, 1.0, 1.0, 1.0])
        )
        positions = (
            x[:2]
            + fraction[:, None] * delta
            + offset * jnp.sin(jnp.pi * fraction)[:, None] ** 2 * normal
        )
        scale = 0.5 * (env.bounds[1] - env.bounds[0]) - 0.12
        mid = 0.5 * (env.bounds[0] + env.bounds[1])
        latent = jnp.arctanh(jnp.clip((positions - mid) / scale, -0.97, 0.97))
        fast.append(slow[len(fast)].at[4:12].set(latent.ravel()))
vectors = jnp.concatenate((slow, jnp.stack(fast)))
fn = jax.jit(
    lambda ys: jax.lax.map(lambda y: evaluate(y, x, q, b, cache, noise, env, cfg, True), ys)
)
costs, paths = fn(vectors)
rows = []
for i, cost in enumerate(costs):
    row = {
        "seed": i % 7,
        "timing": "slow" if i < 7 else "rapid",
        "objective": float(cost),
        "task_cost": float((paths[0].cost[i] + paths[1].cost[i]).mean()),
        "min_clearance": float(min(paths[0].clearance[i].min(), paths[1].clearance[i].min())),
        "terminal_positions": np.array(paths[1].x[i, :, -1, :2]).tolist(),
    }
    rows.append(row)
    print(row, flush=True)
(ROOT / "docs/budgeted_planner/oracle_seed_diagnostic.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
