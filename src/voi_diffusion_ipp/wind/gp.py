"""Finite-rank spatial GP prototype, exact inference for the declared feature kernel.

Two independent wind components share a full spatial coefficient covariance. This is
not POAM, nor a full infinite-rank RBF GP. Features have unit squared norm so prior
pointwise variance is constant; cross-covariances are never discarded by default.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Basis(NamedTuple):
    centers: jax.Array
    length: jax.Array
    independent: jax.Array

    def features(self, locations):
        r = jnp.atleast_2d(locations)
        dist = jnp.sum(((r[:, None, :] - self.centers[None]) / self.length) ** 2, axis=-1)
        # Subtract minimum before exponentiating: stable outside the map as well.
        phi = jnp.exp(-0.5 * (dist - jnp.min(dist, axis=1, keepdims=True)))
        phi /= jnp.sqrt(jnp.sum(phi**2, axis=1, keepdims=True))
        wrong = jax.nn.one_hot(jnp.argmin(dist, axis=1), self.centers.shape[0])
        return jnp.where(self.independent, wrong, phi)


class WorldField(NamedTuple):
    """Hidden physics only. Never accepted by RPCBF or branch-router APIs."""

    basis: Basis
    coefficients: jax.Array

    def at(self, positions):
        return self.basis.features(positions) @ self.coefficients


class InnerFields(NamedTuple):
    """Controller disturbance samples, constructed exclusively from its belief."""

    basis: Basis
    coefficients: jax.Array  # N x M x 2


class WindBelief(NamedTuple):
    basis: Basis
    mean: jax.Array  # M x 2
    covariance: jax.Array  # M x M
    noise_std: jax.Array

    def joint(self, locations):
        phi = self.basis.features(locations)
        return phi @ self.mean, phi @ self.covariance @ phi.T

    def predict(self, locations):
        mean, cov = self.joint(locations)
        return mean, jnp.maximum(jnp.diag(cov), 0)[:, None, None] * jnp.eye(2)[None]

    def sample_field(self, key, n):
        eps = jax.random.normal(key, (n, *self.mean.shape))
        return self.sample_coefficients(eps)

    def sample_coefficients(self, eps):
        # Jitter is numerical regularization, not an added stochastic process.
        L = jnp.linalg.cholesky(self.covariance + 1e-7 * jnp.eye(self.mean.shape[0]))
        return self.mean[None] + jnp.einsum("ij,njk->nik", L, eps)

    def inner_fields(self, eps):
        return InnerFields(self.basis, self.sample_coefficients(eps))

    def condition(self, positions, measured_wind):
        phi = self.basis.features(positions)

        def update(carry, obs):
            mean, cov = carry
            h, y = obs
            v = cov @ h
            s = h @ v + self.noise_std**2
            gain = v / s
            mean = mean + gain[:, None] * (y - h @ mean)[None]
            # Joseph form maintains symmetry/PSD for many low-noise updates.
            A = jnp.eye(cov.shape[0]) - gain[:, None] * h[None]
            cov = A @ cov @ A.T + self.noise_std**2 * jnp.outer(gain, gain)
            return (mean, (cov + cov.T) * 0.5), None

        (mean, cov), _ = jax.lax.scan(update, (self.mean, self.covariance), (phi, measured_wind))
        return self._replace(mean=mean, covariance=cov)


def prior(centers, length=1.1, std=0.55, noise_std=0.1, independent=False):
    basis = Basis(
        jnp.asarray(centers, jnp.float32),
        jnp.asarray(length, jnp.float32),
        jnp.asarray(independent),
    )
    n = len(centers)
    return WindBelief(
        basis, jnp.zeros((n, 2)), std**2 * jnp.eye(n), jnp.asarray(noise_std, jnp.float32)
    )


def information_gain(before, after):
    n = before.covariance.shape[0]
    eps = 1e-7 * jnp.eye(n)
    # Two independent components: 2 * 1/2 log determinant ratio.
    return (
        jnp.linalg.slogdet(before.covariance + eps)[1]
        - jnp.linalg.slogdet(after.covariance + eps)[1]
    )
