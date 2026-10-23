"""Zero-innovation conditioning of the *full* POAM predictive Gaussian process.

This is a planning approximation: kernel, inducing representation and mean are
frozen. It is not a variational POAM refit/relocation. Real POAM is untouched.
The conditional residual and all spatial cross-covariances are retained.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular

from .cache import TableMean
from .gp import InnerFields


def posterior_cross(base, x, y):
    """POAM Eq. 4 at two distinct location sets, including the GP residual."""
    kx = base.basis.kernel_centers(x)
    ky = base.basis.kernel_centers(y)
    ax, ay = kx @ base.inverse, ky @ base.inverse
    return base.kernel(x, y) - ax @ ky.T + ax @ base.covariance @ ay.T


class ConditionalBasis(NamedTuple):
    original: object
    forecast: object

    def evaluate(self, locations, coefficients):
        # Store the split in shapes instead of a traced integer index.
        n = len(self.forecast.positions)
        return self.original.evaluate(locations, coefficients[:-n]) + (
            self.forecast.observation_cross(locations) @ coefficients[-n:]
        )


class CovarianceForecast(NamedTuple):
    base: object
    positions: jax.Array
    count: jax.Array
    cholesky: jax.Array

    @classmethod
    def empty(cls, base, capacity):
        if capacity < 1:
            raise ValueError("Positive observation capacity required")
        return cls(base, jnp.zeros((capacity, 2)), jnp.array(0), jnp.eye(capacity))

    @property
    def active(self):
        return jnp.arange(len(self.positions)) < self.count

    def observation_cross(self, locations):
        return posterior_cross(self.base, locations, self.positions) * self.active[None]

    def condition_locations(self, locations):
        """Append earned *predicted executed* locations; no measurement values.

        Fixed storage keeps JAX scan shapes constant. Inactive observations have
        zero cross-covariance and unit diagonal. Overflow fails numerically,
        rather than silently dropping old measurements.
        """
        if len(locations) > len(self.positions):
            raise ValueError("Observation batch exceeds forecast capacity")
        positions = jax.lax.dynamic_update_slice(self.positions, locations, (self.count, 0))
        count = self.count + len(locations)
        active = jnp.arange(len(positions)) < count
        cov = posterior_cross(self.base, positions, positions)
        cov = (cov + cov.T) * 0.5 * active[:, None] * active[None]
        diagonal = jnp.where(active, self.base.noise_std**2 + 1e-7, 1.0)
        L = jnp.linalg.cholesky(cov + jnp.diag(diagonal))
        L = jnp.where(count <= len(positions), L, jnp.nan)
        return self._replace(positions=positions, count=count, cholesky=L)

    def mean_at(self, locations):
        return self.base.mean_at(locations)

    def joint(self, locations):
        mean, cov = self.base.joint(locations)
        whitened = solve_triangular(self.cholesky, self.observation_cross(locations).T, lower=True)
        updated = cov - whitened.T @ whitened
        return mean, (updated + updated.T) * 0.5

    def predict(self, locations):
        mean, cov = self.base.predict(locations)
        whitened = solve_triangular(self.cholesky, self.observation_cross(locations).T, lower=True)
        var = cov[:, 0, 0] - jnp.sum(whitened**2, axis=0)
        return mean, jnp.maximum(var, 0)[:, None, None] * jnp.eye(2)[None]

    def correction(self, original_fields, observation_eps):
        """Matheron conditional samples; sensor-noise draws are independent.

        Numerical original sample paths retain POAM's existing spectral/support
        approximation. Conditioning uses the full analytic covariance. Sample
        moments must therefore be checked separately from inference algebra.
        """
        residual = (
            original_fields.at(self.positions)
            - self.base.mean_at(self.positions)[None]
            + self.base.noise_std * observation_eps
        ) * self.active[None, :, None]
        return jax.vmap(lambda r: -cho_solve((self.cholesky, True), r))(residual)

    def inner_fields(self, original_fields, observation_eps):
        correction = self.correction(original_fields, observation_eps)
        return InnerFields(
            ConditionalBasis(original_fields.basis, self),
            jnp.concatenate((original_fields.coefficients, correction), axis=1),
        )

    def tabulate(self, original_mean, original_fields, observation_eps):
        """C1 table of conditioned fields; inference itself is never tabulated."""
        table = original_fields.basis
        ny, nx = original_fields.coefficients.shape[1:3]
        xx, yy = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny))
        grid = table.lower + jnp.stack((xx.ravel(), yy.ravel()), axis=-1) * table.spacing
        cross = self.observation_cross(grid).reshape(ny, nx, len(self.positions))
        corrections = self.correction(original_fields, observation_eps)
        fields = original_fields.coefficients + jnp.einsum("yxm,nmc->nyxc", cross, corrections)
        return TableMean(table, original_mean.field), InnerFields(table, fields)


class MeanWorld(NamedTuple):
    """Deterministic planning dynamics only; never physical hidden wind."""

    belief: object

    def at(self, locations):
        return self.belief.mean_at(locations)
