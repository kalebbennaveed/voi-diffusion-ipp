"""Short closed-loop diagnostics before authorizing expensive persistent studies.

Both arms update full POAM on real observations. Actual controls use uncached
spline RPCBF every 0.05 s. The 10-second sampled plan is verified uncached before
executing one second; failed verification invokes the declared backup policy.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from voi_diffusion_ipp.config import config_dict, load_config
from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.planning.budgeted import (
    evaluate,
    make_planner,
    segment,
    split_vector,
    warm_starts,
)
from voi_diffusion_ipp.viz.demo_data import serializable
from voi_diffusion_ipp.wind.cache import build_prior_cache

ROOT = Path(__file__).resolve().parents[1]


def timed(fn, *args):
    start = time.perf_counter()
    result = jax.block_until_ready(fn(*args))
    return result, time.perf_counter() - start


def run(method, periods, seed, directory):
    cfg = load_config(ROOT / "configs/two_region_budgeted.yaml")
    actual_cfg = replace(cfg, dt=0.05, stage_steps=20)
    spec = yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text())
    env, poam, x, q, truth = make_two_region(spec, cfg)
    root = jax.random.PRNGKey(seed)
    positions = jnp.tile(x[:2], (8, 1))
    measurements = truth.at(positions) + cfg.observation_std * jax.random.normal(root, (8, 2))
    _, initial_update = timed(lambda: poam.update(positions, measurements, root, train_steps=10))
    belief, initial_snapshot = timed(poam.planning_belief)
    build = jax.jit(lambda b, e: build_prior_cache(b, e, 129, True))
    anticipate = method == "proposed"
    plan = make_planner(cfg, anticipate)
    verify = jax.jit(
        lambda y, x, q, b, e, n: evaluate(y, x, q, b, None, n, env, cfg, anticipate, True, e)
    )
    execute = jax.jit(
        lambda y, x, q, b, e, n, backup: segment(
            x,
            q,
            split_vector(y, actual_cfg)[0],
            b,
            b.inner_fields(e),
            truth,
            n,
            env,
            actual_cfg,
            False,
            backup,
        )
    )
    snapshot = jax.jit(lambda b: b.predict(env.cells))
    records = []
    xs, qs, controls, observations, corrections, slacks, gaps = (
        [np.array(x)[None]],
        [np.array(q)[None]],
        [],
        [],
        [],
        [],
        [],
    )
    means, stds = [], []
    mean, cov = snapshot(belief)
    means.append(np.array(mean))
    stds.append(np.array(jnp.sqrt(cov[:, 0, 0])))
    start = time.perf_counter()
    total_cost = 0.0
    metadata = {
        "method": method,
        "seed": seed,
        "config": config_dict(cfg),
        "execution_dt": 0.05,
        "source_sha256": source_hashes(),
        "label": "Short development diagnostic, not statistical validation",
        "initial_poam_update_s": initial_update,
        "initial_snapshot_s": initial_snapshot,
    }
    last_update, last_snapshot = 0.0, 0.0
    for i in range(periods):
        period_start = time.perf_counter()
        keys = jax.random.split(jax.random.fold_in(root, 100 + i), 4)
        eps = jax.random.normal(keys[0], (cfg.worlds + cfg.inner_samples, *belief.sample_shape))
        noise = jax.random.normal(keys[1], (cfg.worlds, cfg.stage_steps + cfg.future_steps, 2))
        initials = warm_starts(x, env, cfg)
        cache, cache_s = timed(build, belief, eps)
        best, plan_s = timed(plan, x, q, belief, cache, noise, env, initials, keys[2])
        checked, verify_s = timed(verify, best.vector, x, q, belief, eps, noise)
        backup = not np.isfinite(float(best.cost)) or not np.isfinite(float(checked[0]))
        # Actual observation stream is shared across methods; positions are earned.
        actual_noise = jax.random.normal(jax.random.fold_in(root, 400000 + i), (20, 2))
        executed, execution_s = timed(
            execute, best.vector, x, q, belief, eps[cfg.worlds :], actual_noise, jnp.array(backup)
        )
        _, update_s = timed(
            lambda executed=executed, i=i: poam.update(
                executed.x[1:, :2],
                executed.observations,
                jax.random.fold_in(root, 500000 + i),
                train_steps=cfg.poam_train_steps,
            )
        )
        belief, snapshot_s = timed(poam.planning_belief)
        x, q = executed.x[-1], executed.q[-1]
        total_cost += float(executed.cost)
        mean, cov = snapshot(belief)
        means.append(np.array(mean))
        stds.append(np.array(jnp.sqrt(cov[:, 0, 0])))
        xs.append(np.array(executed.x[1:]))
        qs.append(np.array(executed.q[1:]))
        for store, values in [
            (controls, executed.controls),
            (observations, executed.observations),
            (corrections, executed.correction),
            (slacks, executed.slack),
            (gaps, executed.clearance),
        ]:
            store.append(np.array(values))
        planning_cycle = cache_s + plan_s + verify_s + last_update + last_snapshot
        row = {
            "period": i,
            "state": np.array(x),
            "integrated_deficit": total_cost,
            "cached_objective": float(best.cost),
            "full_objective": float(checked[0]),
            "backup_used": backup,
            "minimum_clearance": float(executed.clearance.min()),
            "max_slack": float(executed.slack.max()),
            "cache_s": cache_s,
            "plan_s": plan_s,
            "verify_s": verify_s,
            "execution_s": execution_s,
            "poam_update_s": update_s,
            "snapshot_s": snapshot_s,
            "planning_cycle_s": planning_cycle,
            "deadline_1s_missed": planning_cycle > 1.0,
            "wall_period_s": time.perf_counter() - period_start,
            "wind_rmse": float(jnp.sqrt(jnp.mean((mean - truth.at(env.cells)) ** 2))),
            "vector": np.array(best.vector),
            "trace": np.array(best.trace),
            "candidates": np.array(best.history.candidates),
            "weights": np.array(best.history.weights),
            "incumbents": np.array(best.history.incumbent),
            "selected_model_path": np.concatenate(
                (np.array(checked[1][0].x[0]), np.array(checked[1][1].x[0, 1:]))
            ),
        }
        records.append(serializable(row))
        last_update, last_snapshot = update_s, snapshot_s
        print(
            json.dumps(
                {
                    k: serializable(row[k])
                    for k in [
                        "period",
                        "state",
                        "integrated_deficit",
                        "backup_used",
                        "minimum_clearance",
                        "planning_cycle_s",
                        "poam_update_s",
                    ]
                }
            ),
            flush=True,
        )
        np.savez_compressed(
            directory / f"{method}.npz",
            x=np.concatenate(xs),
            q=np.concatenate(qs),
            controls=np.concatenate(controls),
            observations=np.concatenate(observations),
            correction=np.concatenate(corrections),
            slack=np.concatenate(slacks),
            clearance=np.concatenate(gaps),
            mean=np.stack(means),
            std=np.stack(stds),
            cells=np.array(env.cells),
            targets=np.array(env.targets),
            obstacles=np.array(env.obstacles),
            bounds=np.array(env.bounds),
            physical_wind=np.array(truth.at(env.cells)),
        )
        (directory / f"{method}.json").write_text(
            json.dumps(
                {
                    "metadata": metadata,
                    "periods": records,
                    "complete": i == periods - 1,
                    "wall_s": time.perf_counter() - start,
                },
                indent=2,
            )
            + "\n"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--periods", type=int, default=12)
    p.add_argument("--seed", type=int, default=8101)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=["proposed", "current_belief"])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for method in args.methods:
        if method not in ("proposed", "current_belief"):
            raise ValueError(method)
        print("METHOD", method, flush=True)
        run(method, args.periods, args.seed, args.output)


if __name__ == "__main__":
    main()
