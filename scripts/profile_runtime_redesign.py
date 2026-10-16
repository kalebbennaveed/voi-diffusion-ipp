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
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.experiments.scenarios import make_scenario
from voi_diffusion_ipp.safety import filter_control
from voi_diffusion_ipp.wind import InnerFields


def cubic_weights(t):
    """Catmull–Rom tensor interpolation, C1 across interior cell boundaries."""
    return jnp.stack(
        (
            -0.5 * t + t * t - 0.5 * t**3,
            1 - 2.5 * t * t + 1.5 * t**3,
            0.5 * t + 2 * t * t - 1.5 * t**3,
            -0.5 * t * t + 0.5 * t**3,
        ),
        axis=-1,
    )


class TableBasis(NamedTuple):
    lower: jax.Array
    spacing: jax.Array

    def evaluate(self, locations, field):
        xy = (jnp.atleast_2d(locations) - self.lower) / self.spacing
        ij = jnp.floor(xy).astype(jnp.int32)
        w = cubic_weights(xy - ij)
        offsets = jnp.arange(-1, 3)
        ix = ij[:, 0, None] + offsets
        iy = ij[:, 1, None] + offsets
        patch = field[iy[:, :, None], ix[:, None, :]]
        result = jnp.einsum("ni,nj,nijc->nc", w[:, 1], w[:, 0], patch)
        valid = (
            jnp.all(ij >= 1, axis=1)
            & (ij[:, 0] < field.shape[1] - 2)
            & (ij[:, 1] < field.shape[0] - 2)
        )
        return jnp.where(valid[:, None], result, jnp.nan)


class TableMean(NamedTuple):
    basis: TableBasis
    field: jax.Array

    def mean_at(self, locations):
        return self.basis.evaluate(locations, self.field)


class DecoupledBasis(NamedTuple):
    table: TableBasis
    prior_fields: jax.Array
    exact_basis: object
    kernel_grid: jax.Array

    def kernel_centers(self, locations):
        if self.kernel_grid.size:
            return self.table.evaluate(locations, self.kernel_grid)
        return self.exact_basis.kernel_centers(locations)

    def evaluate(self, locations, coefficients):
        # The first row identifies an independent fixed prior draw, not a world
        # observed by the controller. Inner and outer banks must stay separate.
        index = coefficients[0, 0].astype(jnp.int32)
        return self.table.evaluate(locations, self.prior_fields[index]) + (
            self.kernel_centers(locations) @ coefficients[1:]
        )


class CachedPosteriorMean(NamedTuple):
    basis: DecoupledBasis
    weights: jax.Array

    def mean_at(self, locations):
        return self.basis.kernel_centers(locations) @ self.weights


class PriorCache(NamedTuple):
    basis: DecoupledBasis
    prior_at_inducing: jax.Array
    inducing_eps: jax.Array

    def condition(self, belief):
        """Only q(u) changes: same exact Matheron correction as POAMBelief.

        Valid only while kernel and inducing positions are frozen. Rebuild the
        entire bank after real POAM kernel training or inducing relocation.
        """
        m = len(belief.mean)
        L = jnp.linalg.cholesky(belief.covariance + 1e-7 * jnp.eye(m))
        u = belief.mean[None] + jnp.einsum("ij,njc->nic", L, self.inducing_eps)
        correction = jnp.einsum("ij,njc->nic", belief.inverse, u - self.prior_at_inducing)
        index = jnp.broadcast_to(jnp.arange(len(u))[:, None, None], (len(u), 1, 2))
        return InnerFields(self.basis, jnp.concatenate((index, correction), axis=1))


def prior_cache_builder(belief, eps, nx, cache_kernel=False):
    """Precompute the unchanged prior part once, not once per posterior."""
    ny = (nx - 1) * 5 // 6 + 1
    low, high = jnp.array([-4.8, -4.0]), jnp.array([4.8, 4.0])
    spacing = (high - low) / jnp.array([nx - 1, ny - 1])
    xx, yy = jnp.meshgrid(jnp.linspace(low[0], high[0], nx), jnp.linspace(low[1], high[1], ny))
    locations = jnp.stack((xx.ravel(), yy.ravel()), axis=-1)

    def build(b):
        m = len(b.mean)
        coefficients = b.sample_coefficients(eps).at[:, -m:].set(0)
        prior_fields = jax.vmap(lambda c: b.basis.evaluate(locations, c))(coefficients)
        gz = jax.vmap(lambda c: b.basis.evaluate(b.basis.centers, c))(coefficients)
        kernel = (
            b.basis.kernel_centers(locations).reshape(ny, nx, len(b.mean))
            if cache_kernel
            else jnp.zeros((0,))
        )
        basis = DecoupledBasis(
            TableBasis(low, spacing), prior_fields.reshape(len(eps), ny, nx, 2), b.basis, kernel
        )
        return PriorCache(basis, gz, eps[:, -m:])

    return build


def table_builder(belief, eps, nx):
    ny = (nx - 1) * 5 // 6 + 1
    low, high = jnp.array([-4.8, -4.0]), jnp.array([4.8, 4.0])
    spacing = (high - low) / jnp.array([nx - 1, ny - 1])
    xx, yy = jnp.meshgrid(jnp.linspace(low[0], high[0], nx), jnp.linspace(low[1], high[1], ny))
    locations = jnp.stack((xx.ravel(), yy.ravel()), axis=-1)
    basis = TableBasis(low, spacing)

    def build(b):
        coefficients = b.sample_coefficients(eps)
        fields = jax.vmap(lambda c: b.basis.evaluate(locations, c))(coefficients)
        fields = fields.reshape((len(eps), ny, nx, 2))
        mean = b.mean_at(locations).reshape(ny, nx, 2)
        return TableMean(basis, mean), InnerFields(basis, fields)

    return build


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
