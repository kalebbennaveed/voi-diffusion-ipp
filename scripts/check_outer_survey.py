"""Declared common archived outer-route wind survey; no central wind leakage."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.planning.budgeted import evaluate, warm_starts
from voi_diffusion_ipp.viz.demo_data import serializable
from voi_diffusion_ipp.wind.cache import build_prior_cache
from voi_diffusion_ipp.wind.poam import POAM

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--data-supported", action="store_true")
args = p.parse_args()
spec = yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text())
base = load_config(ROOT / "configs/two_region_budgeted.yaml")
rows = []
for std in [0.8, 1.5]:
    cfg = replace(base, wind_std=std, hypothetical_relocation=True)
    env, poam, x, q, truth = make_two_region(spec, cfg)
    s = jnp.linspace(0, 1, 21)
    paths = [
        env.roi_a
        + s[:, None] * (env.roi_b - env.roi_a)
        + h * (16 * s**2 * (1 - s) ** 2)[:, None] * truth.normal
        for h in [-1.5, 1.5]
    ]
    points = jnp.concatenate(paths)
    assert (
        float(jax.vmap(lambda p, env=env: env.clearance(jnp.r_[p, jnp.zeros(2)]))(points).min())
        > 0.2
    )
    core = (jnp.abs(points @ truth.tangent) < 1.1) & (jnp.abs(points @ truth.normal) < 0.5)
    assert not np.any(core)
    key = jax.random.PRNGKey(610000 + 8101)
    y = truth.at(points) + cfg.observation_std * jax.random.normal(key, points.shape)
    if args.data_supported:
        poam = POAM.from_data(
            points,
            y,
            jax.random.PRNGKey(600000),
            len(poam.z),
            noise_std=cfg.observation_std,
            prior_std=cfg.wind_std,
            spectral_count=cfg.poam_spectral_nodes,
            prior_support=poam.prior_support,
        )
    else:
        poam.update(points, y, key, train_steps=10)
    b = poam.planning_belief()
    # An honest stressed task state: A currently serviced, B's clarity decayed 8 s without sensing.
    a_region = jnp.all(jnp.abs(env.cells - env.roi_a) <= jnp.array(spec["region_size"]) / 2, axis=1)
    q = jnp.where(a_region, q, q / (1 + env.process_noise * q * 8))
    key = jax.random.PRNGKey(9501)
    eps = jax.random.normal(key, (cfg.worlds + cfg.inner_samples, *b.sample_shape))
    noise = jax.random.normal(jax.random.PRNGKey(9502), (cfg.worlds, 50, 2))
    cache = jax.jit(lambda b, e: build_prior_cache(b, e, 129, True))(b, eps)
    seeds = warm_starts(x, env, cfg, rapid=True)
    mean, cov = b.predict(env.cells)
    passage = (jnp.abs(env.cells @ truth.tangent) < 1.1) & (
        jnp.abs(env.cells @ truth.normal) < 0.22
    )
    rms = float(jnp.sqrt(jnp.mean((mean[passage] - truth.at(env.cells[passage])) ** 2)))
    sigma = float(jnp.sqrt(cov[passage, 0, 0]).mean())
    row = {
        "prior_std": std,
        "data_supported_initialization": args.data_supported,
        "survey_points": np.array(points),
        "survey_measurements": np.array(y),
        "passage_rmse": rms,
        "passage_mean_sigma": sigma,
        "survey_mean_sigma": float(jnp.sqrt(b.predict(points)[1][:, 0, 0]).mean()),
        "methods": {},
    }
    for name, anticipate in [("proposed", True), ("current_belief", False)]:
        fn = jax.jit(
            lambda vs, x=x, q=q, b=b, cache=cache, noise=noise, env=env, cfg=cfg, anticipate=anticipate: (
                jax.lax.map(
                    lambda vector: evaluate(vector, x, q, b, cache, noise, env, cfg, anticipate), vs
                )
            )
        )
        costs, paths = fn(seeds)
        row["methods"][name] = {
            "costs": np.array(costs),
            "terminal_positions": np.array(paths[1].x[:, :, -1, :2]),
            "clearance": np.minimum(
                np.array(paths[0].clearance).min((1, 2)), np.array(paths[1].clearance).min((1, 2))
            ),
        }
    rows.append(serializable(row))
    print(
        json.dumps(
            {
                "prior_std": std,
                "passage_rmse": rms,
                "passage_mean_sigma": sigma,
                "methods": {k: serializable(v["costs"]) for k, v in row["methods"].items()},
            },
            indent=2,
        ),
        flush=True,
    )
    filename = (
        "outer_survey_data_supported.json" if args.data_supported else "outer_survey_screen.json"
    )
    (ROOT / "docs/budgeted_planner" / filename).write_text(json.dumps(rows, indent=2) + "\n")
