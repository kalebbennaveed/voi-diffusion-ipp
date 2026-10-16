"""Measure dense-prior-support resolution without changing POAM inference.

The current 667-support sampler is the reference. This diagnostic is separate
from the frozen integration runs and does not change their numerical model.
"""

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.experiments.scenarios import make_scenario, make_truth, new_poam

cfg = load_config("configs/field_poam.yaml")
rows = []
out = Path("docs/field_poam_validation/support_resolution.json")
for scenario, seed in [("field_vortices", 300), ("field_shear", 301), ("field_mixed", 302)]:
    c = replace(cfg, scenario=scenario)
    env, initial, _, _ = make_scenario(c)
    truth = make_truth(c, initial, seed)
    p = new_poam(c, initial.basis.centers)
    for h, y in enumerate(np.linspace(-1.65, 1.65, 12)):
        x = np.linspace(-2.2, 2.2, 12)[:: -1 if h % 2 else 1]
        pos = jnp.asarray(np.column_stack((x, np.full_like(x, y))))
        noise = jax.random.normal(
            jax.random.fold_in(jax.random.PRNGKey(seed), h + 800000), (len(pos), 2)
        )
        p.update(
            pos,
            truth.at(pos) + cfg.observation_std * noise,
            jax.random.PRNGKey(900000 + h),
            train_steps=cfg.poam_train_steps,
        )
    random = jax.random.uniform(
        jax.random.PRNGKey(661), (96, 2), minval=env.bounds[0], maxval=env.bounds[1]
    )
    query = jnp.concatenate((env.cells, random))
    exact = np.asarray(p.joint(query)[1])
    for nx, ny in [(11, 9), (15, 11), (19, 15), (25, 19), (29, 23)]:
        sx, sy = jnp.meshgrid(jnp.linspace(-2.7, 2.7, nx), jnp.linspace(-2.15, 2.15, ny))
        p.prior_support = jnp.stack((sx.ravel(), sy.ravel()), axis=-1)
        b = p.planning_belief()
        sampled = np.asarray(b.sample_moments(query)[1])
        row = {
            "scenario": scenario,
            "seed": seed,
            "support_nx": nx,
            "support_ny": ny,
            "relative_error": float(np.linalg.norm(sampled - exact) / np.linalg.norm(exact)),
            "max_variance_error": float(np.max(np.abs(np.diag(sampled - exact)))),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        out.write_text(json.dumps(rows, indent=2) + "\n")
