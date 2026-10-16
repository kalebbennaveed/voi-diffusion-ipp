"""Experimental C1 spatial caches of full POAM sample paths for RPCBF.

POAM inference is unchanged. Interpolation is a numerical approximation whose
field/gradient/control errors must be checked. NaN outside the table rejects
invalid queries. Rebuild after each real kernel/inducing update.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .gp import InnerFields


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

    def tabulate(self, belief):
        """Fuse frozen prior and changed inducing correction into dense fields.

        This adds no inference approximation beyond the already-declared field
        table. Useful when many backup evaluations reuse one posterior.
        """
        inner = self.condition(belief)
        kernel = self.basis.kernel_grid
        if not kernel.size:
            raise ValueError("Tabulation requires a cached kernel grid")
        fields = self.basis.prior_fields + jnp.einsum(
            "yxm,nmc->nyxc", kernel, inner.coefficients[:, 1:]
        )
        mean = kernel @ (belief.inverse @ belief.mean)
        return TableMean(self.basis.table, mean), InnerFields(self.basis.table, fields)

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
    """Compatibility helper for the standalone component profiler."""
    return lambda b: build_prior_cache(b, eps, nx, cache_kernel)


def build_prior_cache(b, eps, nx=129, cache_kernel=True):
    """Dynamic belief/draw arguments avoid recompilation on each planning call."""
    ny = (nx - 1) * 5 // 6 + 1
    low, high = jnp.array([-4.8, -4.0]), jnp.array([4.8, 4.0])
    spacing = (high - low) / jnp.array([nx - 1, ny - 1])
    xx, yy = jnp.meshgrid(jnp.linspace(low[0], high[0], nx), jnp.linspace(low[1], high[1], ny))
    locations = jnp.stack((xx.ravel(), yy.ravel()), axis=-1)

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
