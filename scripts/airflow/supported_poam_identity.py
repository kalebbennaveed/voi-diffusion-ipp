"""Exact identity transport when the fixed POAM basis has not changed.

All inducing allocation, full statistics, residual covariance and noise remain
the supported POAM implementation; only a redundant numerical solve is removed.
"""
from functools import partial

import jax
import jax.numpy as jnp

from observed_poam import _double
from voi_diffusion_ipp.wind.poam import (
    attention_weights,
    chol_solve,
    transport_statistic_root,
    variational_from_root,
)


def capacity_saturated(mask):
    """True once every inducing slot contains observation-supported geometry."""
    return jnp.all(mask)


@partial(jax.jit, static_argnames=("minimum_distance",))
def _condition(belief, positions, measurements, mask, *, minimum_distance):
    def allocate(state, point):
        centers, protected = state
        distances = jnp.linalg.norm(centers - point, axis=1)
        covered = jnp.min(jnp.where(protected, distances, jnp.inf)) <= minimum_distance
        nearest_free = jnp.argmin(jnp.where(protected, jnp.inf, distances))
        nearest = jnp.argmin(distances)
        index = jnp.where(jnp.all(protected), nearest, nearest_free)

        def insert(state):
            z, supported = state
            return z.at[index].set(point), supported.at[index].set(True)

        state = jax.lax.cond(covered, lambda state: state, insert, state)
        return state, None

    (centers, mask), _ = jax.lax.scan(
        allocate, (belief.basis.centers, mask), positions
    )
    K = belief.kernel(centers, centers)
    # With unchanged inducing locations and kernel the transport is exactly
    # identity. Re-solving (K+jitter I)^-1 K on every observation spuriously
    # attenuates accumulated statistics, especially in small-eigenvalue modes.
    unchanged=jnp.all(centers==belief.basis.centers) & jnp.all(K==belief.old_K)
    P = jax.lax.cond(unchanged,
        lambda _:jnp.eye(len(centers),dtype=centers.dtype),
        lambda _:chol_solve(belief.old_K,belief.kernel(belief.basis.centers,centers)),None)
    Kuf = belief.kernel(centers, positions)
    a = P.T @ belief.a + Kuf @ measurements
    root = transport_statistic_root(belief.root_B, P, Kuf)
    mean, covariance = variational_from_root(K, a, root, belief.noise_std)
    basis = belief.basis._replace(
        centers=centers,
        center_weights=attention_weights(belief.basis.parameters, centers),
    )
    updated = belief._replace(
        basis=basis,
        mean=mean,
        covariance=(covariance + covariance.T) / 2,
        a=a,
        B=root @ root.T,
        root_B=root,
        old_K=K,
        inverse=chol_solve(K, jnp.eye(len(centers), dtype=K.dtype)),
    )
    return updated, mask


def condition_supported(belief, positions, measurements, mask, *, minimum_distance=0.15):
    """Assimilate actual observations, returning the belief and persistent mask.

    Initialize the mask to ``zeros(len(belief.mean), dtype=bool)``. Call directly;
    the update is internally compiled. The caller must retain the returned mask
    and report ``capacity_saturated(mask)``; it must supply only actual readings.
    """
    if minimum_distance < 0.10:
        raise ValueError("Observation-supported centers require at least 0.10 m spacing")
    with jax.experimental.enable_x64():
        b = _double(belief)
        p = jnp.atleast_2d(jnp.asarray(positions, dtype=jnp.float64))
        y = jnp.atleast_2d(jnp.asarray(measurements, dtype=jnp.float64))
        supported = jnp.asarray(mask, dtype=bool)
        if p.shape != y.shape or p.shape[1] != 2:
            raise ValueError("Wind observations require matching N-by-2 arrays")
        if supported.shape != (len(b.mean),):
            raise ValueError("Support mask must have one entry per inducing variable")
        return _condition(b, p, y, supported, minimum_distance=minimum_distance)
