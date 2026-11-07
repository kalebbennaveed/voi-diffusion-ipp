"""Full POAM predictive distribution and differentiable posterior field sampling.

Prediction is Eq. (4) of Chen et al. (RSS 2024), including Kxx-Kxz Kzz^-1 Kzx.
Conditioning uses POAM's unweighted sufficient statistics, Eqs. (19)-(21).
Continuous draws use decoupled Matheron sampling (Wilson et al., ICML 2020):
Fourier quadrature approximates ONLY the prior sample paths, never inference.
The residual process is retained, rather than setting it to zero between anchors.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .gp import InnerFields


def spectral_nodes(count, seed=7319):
    """Scrambled Sobol Gaussian spectral quadrature, reproducible and extensible."""
    if count < 2 or count & (count - 1):
        raise ValueError("POAM spectral node count must be a power of two >= 2")
    u = qmc.Sobol(2, scramble=True, seed=seed).random_base2(int(np.log2(count)))
    return jnp.asarray(ndtri(np.clip(u, 1e-7, 1 - 1e-7)), jnp.float32)


class PosteriorBasis(NamedTuple):
    centers: jax.Array
    parameters: dict
    lengths: jax.Array
    frequencies: jax.Array
    support: jax.Array
    support_inverse: jax.Array
    support_cholesky: jax.Array
    support_weights: jax.Array
    center_weights: jax.Array

    def spectral_features(self, locations):
        from .poam import attention_weights

        x = jnp.atleast_2d(locations)
        w = attention_weights(self.parameters, x)
        angles = (x @ self.frequencies.T)[:, None] / self.lengths[None, :, None]
        trig = jnp.concatenate((jnp.cos(angles), jnp.sin(angles)), axis=-1)
        phi = jnp.einsum("xj,xl,xlr->xjlr", w, w, trig)
        return phi.reshape(len(x), -1) * jnp.sqrt(
            jnp.exp(self.parameters["log_amplitude"]) / len(self.frequencies)
        )

    def cross(self, x, y, wy):
        from .poam import attention_weights

        x = jnp.atleast_2d(x)
        wx = attention_weights(self.parameters, x)
        dist = jnp.sum((x[:, None] - y[None]) ** 2, axis=-1)
        base = jnp.exp(-dist[:, :, None] / (2 * self.lengths**2))
        return (
            jnp.exp(self.parameters["log_amplitude"])
            * (wx @ wy.T)
            * jnp.einsum("il,jl,ijl->ij", wx, wy, base)
        )

    def kernel_support(self, x):
        return self.cross(x, self.support, self.support_weights)

    def kernel_centers(self, x):
        return self.cross(x, self.centers, self.center_weights)

    def support_correction(self, residual):
        """Apply the same regularized support covariance inverse to each draw.

        An empty inverse selects its Cholesky factor representation. This avoids
        forming a dense inverse after kernel fitting; no covariance is dropped.
        """
        if self.support_inverse.size:
            return jnp.einsum('st,ntc->nsc',self.support_inverse,residual)
        ns=len(self.support);n=len(residual)
        rhs=residual.transpose(1,0,2).reshape(ns,-1)
        solution=jsp.linalg.cho_solve((self.support_cholesky,True),rhs)
        return solution.reshape(ns,n,2).transpose(1,0,2)

    def prior_features(self, locations):
        """Moments of the support-corrected prior sampler, residual retained."""
        x = jnp.atleast_2d(locations)
        phi = self.spectral_features(x)
        if not len(self.support):
            return phi
        K=self.kernel_support(x)
        A = K @ self.support_inverse if self.support_inverse.size else jsp.linalg.cho_solve((self.support_cholesky,True),K.T).T
        residual = phi - A @ self.spectral_features(self.support)
        return jnp.concatenate((residual, A @ self.support_cholesky), axis=1)

    def features(self, locations):
        x = jnp.atleast_2d(locations)
        return jnp.concatenate(
            (self.spectral_features(x), self.kernel_support(x), self.kernel_centers(x)), axis=1
        )

    def evaluate(self, locations, coefficients):
        """Contract weights first; mathematically features(x) @ coefficients."""
        from .poam import attention_weights

        x = jnp.atleast_2d(locations)
        w = attention_weights(self.parameters, x)
        m, l = len(self.centers), len(self.lengths)
        f = l * l * 2 * len(self.frequencies)
        prior = coefficients[:f].reshape(l, l, 2 * len(self.frequencies), 2)
        combined = jnp.einsum("xj,jlrc->xlrc", w, prior)
        angles = (x @ self.frequencies.T)[:, None] / self.lengths[None, :, None]
        trig = jnp.concatenate((jnp.cos(angles), jnp.sin(angles)), axis=-1)
        draw = jnp.einsum("xl,xlr,xlrc->xc", w, trig, combined)
        draw *= jnp.sqrt(jnp.exp(self.parameters["log_amplitude"]) / len(self.frequencies))
        return (
            draw
            + self.kernel_support(x) @ coefficients[f:-m]
            + self.kernel_centers(x) @ coefficients[-m:]
        )


class POAMBelief(NamedTuple):
    basis: PosteriorBasis
    mean: jax.Array  # inducing distribution q(u), NOT coefficients of a finite field
    covariance: jax.Array
    inverse: jax.Array
    a: jax.Array
    B: jax.Array
    old_K: jax.Array
    noise_std: jax.Array
    root_B: jax.Array

    @property
    def sample_shape(self):
        components = len(self.basis.lengths)
        features = components**2 * 2 * len(self.basis.frequencies)
        return (features + len(self.basis.support) + len(self.mean), 2)

    def kernel(self, x, y):
        from .poam import attentive_kernel

        return attentive_kernel(
            self.basis.parameters, jnp.atleast_2d(x), jnp.atleast_2d(y), self.basis.lengths
        )

    def mean_at(self, locations):
        return self.basis.kernel_centers(locations) @ (self.inverse @ self.mean)

    def joint(self, locations):
        Kxz = self.basis.kernel_centers(locations)
        A = Kxz @ self.inverse
        cov = self.kernel(locations, locations) - A @ Kxz.T + A @ self.covariance @ A.T
        return A @ self.mean, (cov + cov.T) * 0.5

    def predict(self, locations):
        Kxz = self.basis.kernel_centers(locations)
        A = Kxz @ self.inverse
        # Attentive kernel diagonal is exactly its amplitude.
        var = jnp.exp(self.basis.parameters["log_amplitude"]) - jnp.sum(A * Kxz, axis=1)
        var += jnp.sum((A @ self.covariance) * A, axis=1)
        return A @ self.mean, jnp.maximum(var, 0)[:, None, None] * jnp.eye(2)[None]

    def condition(self, positions, measured_wind, relocate=False):
        from .poam import (
            attention_weights,
            chol_solve,
            select_inducing_jax,
            transport_statistic_root,
            variational_from_root,
            spatial_observations,
        )

        positions, measured_wind = spatial_observations(positions, measured_wind)
        z = self.basis.centers
        if relocate:
            z = select_inducing_jax(
                self.basis.parameters, jnp.concatenate((z, positions)), self.basis.lengths, len(z)
            )
        K = self.kernel(z, z)
        P = chol_solve(self.old_K, self.kernel(self.basis.centers, z))
        Kuf = self.kernel(z, positions)
        a = P.T @ self.a + Kuf @ measured_wind
        root_B = transport_statistic_root(self.root_B, P, Kuf)
        B = root_B @ root_B.T
        mean, cov = variational_from_root(K, a, root_B, self.noise_std)
        basis = self.basis._replace(
            centers=z, center_weights=attention_weights(self.basis.parameters, z)
        )
        return self._replace(
            basis=basis,
            mean=mean,
            covariance=(cov + cov.T) / 2,
            a=a,
            B=B,
            root_B=root_B,
            old_K=K,
            inverse=chol_solve(K, jnp.eye(len(z))),
        )

    def sample_coefficients(self, eps):
        """Two Matheron corrections: full prior on support, then sparse q(u).

        A spectral residual fills gaps between support locations; it is never
        dropped. Support locations have the full prior joint covariance up to
        numerical regularization, independently of the spectral resolution.
        """
        m, ns = len(self.mean), len(self.basis.support)
        nf = self.sample_shape[0] - m - ns
        prior_eps, support_eps, inducing_eps = eps[:, :nf], eps[:, nf : nf + ns], eps[:, -m:]
        L = jnp.linalg.cholesky(self.covariance + 1e-7 * jnp.eye(m))
        u = self.mean[None] + jnp.einsum("ij,njc->nic", L, inducing_eps)
        gz = jnp.einsum("mf,nfc->nmc", self.basis.spectral_features(self.basis.centers), prior_eps)
        if ns:
            gs = jnp.einsum(
                "sf,nfc->nsc", self.basis.spectral_features(self.basis.support), prior_eps
            )
            vs = jnp.einsum("st,ntc->nsc", self.basis.support_cholesky, support_eps)
            support_coeff = self.basis.support_correction(vs-gs)
            gz += jnp.einsum(
                "ms,nsc->nmc", self.basis.kernel_support(self.basis.centers), support_coeff
            )
        else:
            support_coeff = support_eps
        correction = jnp.einsum("ij,njc->nic", self.inverse, u - gz)
        return jnp.concatenate((prior_eps, support_coeff, correction), axis=1)

    def sample_field(self, key, n):
        return self.sample_coefficients(jax.random.normal(key, (n, *self.sample_shape)))

    def inner_fields(self, eps):
        return InnerFields(self.basis, self.sample_coefficients(eps))

    def sample_moments(self, locations):
        """Analytic moments of the numerical sampler, for covariance convergence checks."""
        x, z = jnp.atleast_2d(locations), self.basis.centers
        A = self.kernel(x, z) @ self.inverse
        residual_features = self.basis.prior_features(x) - A @ self.basis.prior_features(z)
        cov = residual_features @ residual_features.T + A @ self.covariance @ A.T
        return A @ self.mean, cov
