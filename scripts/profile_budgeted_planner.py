"""Bounded full-POAM/actual-MBD/actual-RPCBF latency and mismatch experiment."""

import argparse
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.planning.budgeted import evaluate, make_planner, warm_starts
from voi_diffusion_ipp.wind.cache import build_prior_cache

ROOT = Path(__file__).resolve().parents[1]


def timed(fn, *args):
    start = time.perf_counter()
    result = jax.block_until_ready(fn(*args))
    return result, time.perf_counter() - start


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--relocate", action="store_true")
    parser.add_argument("--future-update-steps", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--active-cells", action="store_true")
    parser.add_argument("--proposals", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs/two_region_budgeted.yaml")
    cfg = replace(
        cfg,
        hypothetical_relocation=args.relocate,
        candidate_chunk=args.chunk,
        budgeted_update_steps=args.future_update_steps,
        proposals=args.proposals,
    )
    spec = yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text())
    env, poam, x, q, truth = make_two_region(spec, cfg)
    root = jax.random.PRNGKey(8101)
    # Explicit stationary initialization measurements; no unvisited observations.
    positions = jnp.tile(x[:2], (8, 1))
    observations = truth.at(positions) + cfg.observation_std * jax.random.normal(root, (8, 2))
    _, startup_update = timed(lambda: poam.update(positions, observations, root, train_steps=10))
    belief, startup_snapshot = timed(poam.planning_belief)
    print("initial POAM", startup_update, startup_snapshot, flush=True)
    build = jax.jit(lambda b, e: build_prior_cache(b, e, 129, True))
    plan = make_planner(
        cfg, True, np.flatnonzero(np.asarray(env.targets) > 0) if args.active_cells else None
    )
    approximate = jax.jit(lambda y, x, q, b, c, n: evaluate(y, x, q, b, c, n, env, cfg, True))
    exact = jax.jit(lambda y, x, q, b, e, n: evaluate(y, x, q, b, None, n, env, cfg, True, True, e))
    result = {
        "label": "Development timings; not yet a mission or learning comparison",
        "backend": jax.default_backend(),
        "machine": platform.machine(),
        "jax": jax.__version__,
        "active_cells_only": args.active_cells,
        "startup_poam_update_s": startup_update,
        "startup_snapshot_s": startup_snapshot,
        "calls": [],
    }
    for i in range(args.calls):
        keys = jax.random.split(jax.random.fold_in(root, i + 1), 4)
        eps = jax.random.normal(keys[0], (cfg.worlds + cfg.inner_samples, *belief.sample_shape))
        noise = jax.random.normal(keys[1], (cfg.worlds, cfg.stage_steps + cfg.future_steps, 2))
        # Vary actual clarity and current velocity, not identical cached calls.
        xi = x.at[2:].set(0.04 * jax.random.normal(keys[2], (2,)))
        qi = jnp.maximum(q - 0.015 * i, 0.0)
        initials = warm_starts(xi, env, cfg)
        cache, cache_s = timed(build, belief, eps)
        print("cache", i, cache_s, flush=True)
        best, plan_s = timed(plan, xi, qi, belief, cache, noise, env, initials, keys[3])
        print("plan", i, plan_s, "cost", float(best.cost), flush=True)
        fast, fast_check_s = timed(approximate, best.vector, xi, qi, belief, cache, noise)
        full, full_check_s = timed(exact, best.vector, xi, qi, belief, eps, noise)
        print("full check", i, full_check_s, "cost", float(full[0]), flush=True)
        row = {
            "index": i,
            "includes_first_compilation": i == 0,
            "cache_s": cache_s,
            "plan_s": plan_s,
            "exact_check_s": full_check_s,
            "telemetry_s": fast_check_s,
            "cache_plan_verify_s": cache_s + plan_s + full_check_s,
            "deadline_1s_missed": cache_s + plan_s + full_check_s > 1.0,
            "cached_objective": float(best.cost),
            "full_objective": float(full[0]),
            "full_task_cost": float((full[1][0].cost + full[1][1].cost).mean()),
            "minimum_full_clearance": float(
                min(full[1][0].clearance.min(), full[1][1].clearance.min())
            ),
            "max_control_disagreement": float(
                max(
                    jnp.abs(fast[1][0].controls - full[1][0].controls).max(),
                    jnp.abs(fast[1][1].controls - full[1][1].controls).max(),
                )
            ),
            "candidate_invalid_fraction": np.asarray(best.trace[:, 3]),
            "first_action": np.asarray(full[1][0].controls[:, 0]),
            "vector": np.asarray(best.vector),
            "trace": np.asarray(best.trace),
        }
        result["calls"].append(serial(row))
        (
            args.output
            or (
                ROOT
                / "docs/budgeted_planner"
                / (
                    f"profile_relocation_chunk{args.chunk}_updates{args.future_update_steps}.json"
                    if args.relocate
                    else f"profile_chunk{args.chunk}.json"
                )
            )
        ).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
