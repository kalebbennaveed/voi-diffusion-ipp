"""Bounded CPU experiment: tabulate coherent POAM fields, retain spline RPCBF.

This is a numerical field-interpolation experiment, NOT a deployable planner.
No POAM inference, backup integration, spline maximum, gradient or QP is replaced.
Out-of-table queries yield NaN so an invalid cache cannot silently extrapolate.
Run from the repository root with UV; no source/config of frozen studies changes.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.experiments.scenarios import make_scenario
from voi_diffusion_ipp.safety import filter_control
from voi_diffusion_ipp.wind.cache import CachedPosteriorMean, prior_cache_builder, table_builder


def ready(value):
    return jax.block_until_ready(value)


def timed(fn, args, repeats):
    start = time.perf_counter()
    executable = jax.jit(fn).lower(*args).compile()
    compile_s = time.perf_counter() - start
    ready(executable(*args))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = ready(executable(*args))
        times.append(time.perf_counter() - start)
    return out, {
        "compile_s": compile_s,
        "median_s": float(np.median(times)),
        "p95_s": float(np.percentile(times, 95)),
        "samples_s": times,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[65, 129])
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--decoupled-only", action="store_true")
    parser.add_argument("--cache-kernel", action="store_true")
    args = parser.parse_args()
    cfg = load_config("configs/persistent_archipelago.yaml")
    env, belief, _, _ = make_scenario(cfg)
    # A nonzero conditioned belief, with full POAM conditioning, exercises mean lookup.
    locations = jnp.array([[-1.8, -0.2], [-1.5, 0.0], [-1.3, 0.2], [-0.9, 0.3]])
    belief = belief.condition(
        locations, jnp.array([[0.2, -0.3], [0.3, -0.2], [0.35, -0.1], [0.4, 0.1]])
    )
    eps = jax.random.normal(jax.random.PRNGKey(7901), (cfg.inner_samples, *belief.sample_shape))
    inner = belief.inner_fields(eps)
    # Declared random test points, not selected after interpolation outcomes.
    states = jax.random.uniform(
        jax.random.PRNGKey(7902),
        (32, 4),
        minval=jnp.array([-2.0, -1.4, -0.5, -0.5]),
        maxval=jnp.array([2.0, 1.4, 0.5, 0.5]),
    )
    nominal = jnp.array([0.6, 0.2])
    full = lambda xs, b, ins: jax.lax.map(
        lambda x: filter_control(x, nominal, b, ins, env, cfg), xs
    )
    reference, baseline = timed(full, (states, belief, inner), args.repeats)
    print("full", json.dumps(baseline), flush=True)
    output = {
        "label": "component timing only; not a 10-second planner",
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "machine": platform.machine(),
        "jax": jax.__version__,
        "states": len(states),
        "full": baseline,
        "grids": [],
    }
    path = Path(
        "docs/runtime_redesign/"
        + (
            "fully_cached_profile.json"
            if args.cache_kernel
            else "decoupled_profile.json"
            if args.decoupled_only
            else "component_profile.json"
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    for nx in args.sizes:
        refresh_time = None
        if args.decoupled_only:
            cache, build_time = timed(
                prior_cache_builder(belief, eps, nx, args.cache_kernel), (belief,), args.repeats
            )
            cached_inner, refresh_time = timed(
                lambda b, c: c.condition(b), (belief, cache), args.repeats
            )
            mean = (
                CachedPosteriorMean(cache.basis, belief.inverse @ belief.mean)
                if args.cache_kernel
                else belief
            )
            cached = (mean, cached_inner)
        else:
            cached, build_time = timed(table_builder(belief, eps, nx), (belief,), args.repeats)
        approximate, filter_time = timed(full, (states, *cached), args.repeats)
        # Value and spatial-derivative checks for the same coherent field draws.
        points = states[:, :2]
        exact_wind = jax.jit(lambda p: inner.at(p))(points)
        table_wind = cached[1].at(points)
        jac = jax.jit(jax.vmap(jax.jacfwd(lambda p: inner.at(p)[:, 0])))(points)
        table_jac = jax.jit(jax.vmap(jax.jacfwd(lambda p, fields=cached[1]: fields.at(p)[:, 0])))(
            points
        )
        u, info = reference
        v, fast = approximate
        row = {
            "nx": nx,
            "build": build_time,
            "filter": filter_time,
            "posterior_refresh": refresh_time,
            "speedup_excluding_cache_build": baseline["median_s"] / filter_time["median_s"],
            "max_wind_error": float(jnp.max(jnp.abs(exact_wind - table_wind))),
            "max_wind_gradient_error": float(jnp.max(jnp.abs(jac - table_jac))),
            "max_barrier_error": float(jnp.max(jnp.abs(info.value - fast.value))),
            "max_barrier_gradient_error": float(jnp.max(jnp.abs(info.gradient - fast.gradient))),
            "max_control_error": float(jnp.max(jnp.linalg.norm(u - v, axis=1))),
            "feasibility_disagreements": int(jnp.sum(info.feasible != fast.feasible)),
            "finite": bool(jnp.all(jnp.isfinite(v)) & jnp.all(jnp.isfinite(fast.gradient))),
        }
        output["grids"].append(row)
        path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
