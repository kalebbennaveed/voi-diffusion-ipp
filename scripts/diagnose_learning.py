"""Isolate posterior-to-RPCBF mechanism with matched state and nominal action."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.scenarios import make_scenario
from voi_diffusion_ipp.safety import filter_control

cfg = Config()
env, b, _, _ = make_scenario(cfg)
eps = jax.random.normal(jax.random.PRNGKey(73), (cfg.inner_samples, *b.mean.shape))
obs = jnp.tile(jnp.array([[-0.65, 0.0]]), (cfg.stage_steps, 1))
post = b.condition(obs, jnp.zeros_like(obs))
f = jax.jit(
    lambda x, belief: filter_control(
        x, jnp.array([0.65, 0.0]), belief, belief.inner_fields(eps), env, cfg
    )
)
rows = []
for px in [-1.2, -0.9, -0.65, -0.4, -0.2, 0.0]:
    x = jnp.array([px, 0.0, 0.2, 0.0])
    u, info = f(x, b)
    up, ip = f(x, post)
    rows.append(
        {
            "x": px,
            "prior_control": np.asarray(u).tolist(),
            "posterior_control": np.asarray(up).tolist(),
            "prior_correction": float(info.correction),
            "posterior_correction": float(ip.correction),
            "prior_admissible": float(info.admissible_fraction),
            "posterior_admissible": float(ip.admissible_fraction),
            "prior_feasible": bool(info.feasible),
            "posterior_feasible": bool(ip.feasible),
        }
    )
Path("results/learning_mechanism.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
