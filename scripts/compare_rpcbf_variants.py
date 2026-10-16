"""Compare the explicitly labeled local spline evaluator against full RPCBF."""

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.scenarios import make_scenario
from voi_diffusion_ipp.safety import filter_control

cfg = Config()
env, b, _, _ = make_scenario(cfg)
eps = jax.random.normal(jax.random.PRNGKey(71), (cfg.inner_samples, *b.mean.shape))
post = b.condition(jnp.array([[-0.8, 0.0], [-0.6, 0.0], [-0.4, 0.0]]), jnp.zeros((3, 2)))
xs = jax.random.uniform(
    jax.random.PRNGKey(72),
    (64, 4),
    minval=jnp.array([-1.5, -0.3, -0.5, -0.3]),
    maxval=jnp.array([1.5, 0.3, 0.5, 0.3]),
)
full = jax.jit(
    lambda x, bb: filter_control(x, jnp.array([0.5, 0.0]), bb, bb.inner_fields(eps), env, cfg)
)
fast = jax.jit(
    lambda x, bb: filter_control(
        x, jnp.array([0.5, 0.0]), bb, bb.inner_fields(eps), env, replace(cfg, spline="fast")
    )
)
rows = []
for label, bb in [("prior", b), ("posterior", post)]:
    errors = []
    for x in xs:
        u, a = full(x, bb)
        v, c = fast(x, bb)
        errors.append(
            [
                float(jnp.max(jnp.abs(a.value - c.value))),
                float(jnp.max(jnp.abs(a.gradient - c.gradient))),
                float(jnp.linalg.norm(u - v)),
                float(a.feasible != c.feasible),
            ]
        )
    e = np.array(errors)
    rows.append(
        {
            "belief": label,
            "n": len(xs),
            "max_value_error": float(e[:, 0].max()),
            "max_gradient_error": float(e[:, 1].max()),
            "max_control_error": float(e[:, 2].max()),
            "feasibility_disagreements": int(e[:, 3].sum()),
        }
    )
Path("results/rpcbf_variants.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
