"""Declared same-state mechanism and fresh posterior-field stress checks."""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.replay_budgeted import snapshots
from voi_diffusion_ipp.planning import covariance
from voi_diffusion_ipp.planning.budgeted import warm_starts
from voi_diffusion_ipp.viz.demo_data import serializable
from voi_diffusion_ipp.wind.cache import build_prior_cache
from voi_diffusion_ipp.wind.covariance_forecast import CovarianceForecast
from voi_diffusion_ipp.wind.gp import WorldField


def make_evaluator(cfg, anticipate, adapt_reference=True, stress=False):
    @jax.jit
    def fn(v, x, q, b, cache, eps, obs_eps, env, field=None):
        world = WorldField(cache.basis.table, field) if stress else None
        cost, (first, future) = covariance.evaluate(
            v,
            x,
            q,
            b,
            cache,
            eps,
            obs_eps,
            env,
            cfg,
            anticipate,
            world=world,
            adapt_reference=adapt_reference,
        )
        metrics = jnp.array(
            [
                cost,
                first.cost[0],
                future.cost[0],
                future.correction.mean(),
                future.clearance.min(),
                future.slack.max(),
            ]
        )
        return metrics if stress else (metrics, first, future)

    return fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--periods", type=int, nargs="+", default=[0, 5, 18])
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "source_sha256": source_hashes(),
        "input": str(args.path),
        "input_sha256": hashlib.sha256(args.path.read_bytes()).hexdigest(),
        "periods": args.periods,
        "posterior_stress_worlds": args.worlds,
        "seed": 972000,
        "policy_selection": "all seven shared starts plus recorded winner",
        "claim": "Mechanism diagnostic; unchanged mean is deliberately stressed, not exact VoI",
        "status": "running",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    rows = []
    compiled = {}
    build = jax.jit(lambda b, e: build_prior_cache(b, e, 129, True))
    for i, cfg, env, b, x, q, row, meta in snapshots(args.path, args.periods):
        # Raw costs and contacts are reported separately, never inf-minus-inf.
        cfg = replace(cfg, reject_collisions=False)
        interval = cfg.budgeted_update_steps or cfg.future_steps
        capacity = cfg.stage_steps + cfg.future_steps - interval
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(972000 + i), 3)
        eps = jax.random.normal(k1, (cfg.inner_samples, *b.sample_shape))
        obs_eps = jax.random.normal(k2, (cfg.inner_samples, capacity, 2))
        cache = build(b, eps)
        vectors = jnp.concatenate(
            (
                warm_starts(x, env, cfg, meta["rapid_shared_initializations"]),
                jnp.array(row["vector"])[None],
            )
        )

        if repr(cfg) not in compiled:
            compiled[repr(cfg)] = (
                [
                    make_evaluator(cfg, False),
                    make_evaluator(cfg, True),
                    make_evaluator(cfg, True, False),
                ],
                [make_evaluator(cfg, False, stress=True), make_evaluator(cfg, True, stress=True)],
            )
        functions, stress = compiled[repr(cfg)]
        physical_eps = jax.random.normal(k3, (args.worlds, *b.sample_shape))
        physical_cache = build(b, physical_eps)
        _, physical_fields = physical_cache.tabulate(b)
        inputs = (x, q, b, cache, eps, obs_eps, env)
        for vi, vector in enumerate(vectors):
            start = time.perf_counter()
            f, l, fo = [jax.block_until_ready(fn(vector, *inputs)) for fn in functions]
            np.testing.assert_array_equal(f[1].x, l[1].x)
            np.testing.assert_array_equal(f[1].q, l[1].q)
            forecast = CovarianceForecast.empty(b, capacity).condition_locations(l[1].x[0, 1:, :2])
            before = np.sqrt(np.asarray(b.predict(env.cells)[1][:, 0, 0]))
            after = np.sqrt(np.asarray(forecast.predict(env.cells)[1][:, 0, 0]))
            probe = np.asarray(
                [b.predict(env.probe)[1][0, 0, 0], forecast.predict(env.probe)[1][0, 0, 0]]
            )
            values = []
            for field in physical_fields.coefficients:
                values.append([np.asarray(fn(vector, *inputs, field)) for fn in stress])
            rows.append(
                {
                    "period": i,
                    "candidate": vi,
                    "vector": vector,
                    "frozen": f[0],
                    "covariance": l[0],
                    "filter_only": fo[0],
                    "future_path_difference": float(jnp.linalg.norm(f[2].x - l[2].x)),
                    "first_stage_difference": float(jnp.abs(f[1].x - l[1].x).max()),
                    "mean_std_reduction": float(np.mean(before - after)),
                    "max_std_reduction": float(np.max(before - after)),
                    "probe_variance_before_after": probe,
                    "fresh_field_stress": values,
                    "wall_s_including_first_compile": time.perf_counter() - start,
                }
            )
            np.savez_compressed(
                args.output / f"paths_{i}_{vi}.npz",
                x_frozen=f[2].x,
                x_covariance=l[2].x,
                x_first=f[1].x,
                cells=env.cells,
                std_before=before,
                std_after=after,
                obstacles=env.obstacles,
            )
            (args.output / "results.json").write_text(
                json.dumps(
                    serializable(
                        {
                            "columns": [
                                "total_cost",
                                "first_cost",
                                "future_cost",
                                "mean_correction",
                                "min_clearance",
                                "max_slack",
                            ],
                            "rows": rows,
                        }
                    ),
                    indent=2,
                )
            )
            print(i, vi, "cost frozen/cov", float(f[0][0]), float(l[0][0]), flush=True)
    protocol["status"] = "complete"
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
