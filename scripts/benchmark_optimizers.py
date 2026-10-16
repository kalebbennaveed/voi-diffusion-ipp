"""Equal-budget nonconvex double-integrator trajectory optimization."""

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.planning.mbd import optimize

cfg = replace(Config(), diffusion_steps=16, proposals=48, temperature=0.04, candidate_chunk=8)


def objective(y):
    u = jnp.tanh(y.reshape((6, 2)))

    def body(c, u):
        p, v = c
        v = v + 0.35 * u
        p = p + 0.35 * v
        # Two routes around a central obstacle make this a nonconvex TO benchmark.
        return (p, v), 10 * jax.nn.softplus(
            15 * (0.35**2 - jnp.sum((p - jnp.array([0.8, 0])) ** 2))
        ) / 15

    (p, v), obstacle = jax.lax.scan(body, (jnp.array([0.0, 0.0]), jnp.array([0.5, 0.0])), u)
    return (
        3 * jnp.sum((p - jnp.array([1.6, 0])) ** 2)
        + 0.2 * jnp.sum(v * v)
        + jnp.sum(obstacle)
        + 0.02 * jnp.sum(u * u)
    )


rows = []
for method in ["mbd", "cem", "mppi"]:
    f = jax.jit(lambda key, method=method: optimize(objective, jnp.zeros(12), key, cfg, method))
    for seed in range(10):
        r = f(jax.random.PRNGKey(seed))
        rows.append(
            {
                "method": method,
                "seed": seed,
                "cost": float(r.cost),
                "evaluations": int(r.evaluations),
            }
        )
Path("results/optimizer_benchmark.json").write_text(json.dumps(rows, indent=2))
for method in ["mbd", "cem", "mppi"]:
    a = [r["cost"] for r in rows if r["method"] == method]
    print(method, np.mean(a), np.std(a), flush=True)
