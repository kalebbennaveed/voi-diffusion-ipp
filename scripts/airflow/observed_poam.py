"""Observation-supported inducing allocation for the added survey experiment.

This changes only where the existing 128 POAM inducing variables are placed.
Every new center is an actually measured position. The kernel, inducing count,
unweighted sufficient statistics, full conditional residual, and variational
update equations are retained. Old experiments do not import this module.

The old prior-kernel coverage pivot can retain all unobserved grid centers even
after repeated local measurements, leaving a large conditional-residual floor
at the sensor. Here a measurement farther than 0.15 m from every center replaces
the nearest center. Accumulated statistics are transported, never reset. Moving
a center may increase uncertainty in its old neighborhood; that is retained.

Inference uses float64 because locally concentrated centers can make float32
kernel inverses inaccurate. Forecasts can retain their separately declared
float32 numerics. No posterior variance is clipped or artificially reduced.
"""
from functools import partial

import jax
import jax.numpy as jnp

from voi_diffusion_ipp.wind.poam import (
    attention_weights,
    chol_solve,
    transport_statistic_root,
    variational_from_root,
)


def _double(tree):
    return jax.tree.map(
        lambda a: a.astype(jnp.float64)
        if hasattr(a, "dtype") and jnp.issubdtype(a.dtype, jnp.floating)
        else a,
        tree,
    )


@partial(jax.jit, static_argnames=("minimum_distance",))
def _condition(belief, positions, measurements, *, minimum_distance):
    def allocate(centers, point):
        distances = jnp.linalg.norm(centers - point, axis=1)
        nearest = jnp.argmin(distances)
        centers = jax.lax.cond(
            distances[nearest] > minimum_distance,
            lambda z: z.at[nearest].set(point),
            lambda z: z,
            centers,
        )
        return centers, None

    centers, _ = jax.lax.scan(allocate, belief.basis.centers, positions)
    K = belief.kernel(centers, centers)
    P = chol_solve(belief.old_K, belief.kernel(belief.basis.centers, centers))
    Kuf = belief.kernel(centers, positions)
    a = P.T @ belief.a + Kuf @ measurements
    root = transport_statistic_root(belief.root_B, P, Kuf)
    mean, covariance = variational_from_root(K, a, root, belief.noise_std)
    basis = belief.basis._replace(
        centers=centers,
        center_weights=attention_weights(belief.basis.parameters, centers),
    )
    return belief._replace(
        basis=basis,
        mean=mean,
        covariance=(covariance + covariance.T) / 2,
        a=a,
        B=root @ root.T,
        root_B=root,
        old_K=K,
        inverse=chol_solve(K, jnp.eye(len(centers), dtype=K.dtype)),
    )


def condition_observed(belief, positions, measurements, *, minimum_distance=0.15):
    """Assimilate actual wind observations with full POAM statistic transport.

    Call this wrapper directly; its internal fixed-shape update is already
    compiled. Position/measurement batches must be matching N-by-2 arrays.
    The caller is responsible for supplying only physically acquired data.
    """
    if minimum_distance < 0.10:
        raise ValueError("Observation-supported centers require at least 0.10 m spacing")
    with jax.experimental.enable_x64():
        b = _double(belief)
        p = jnp.atleast_2d(jnp.asarray(positions, dtype=jnp.float64))
        y = jnp.atleast_2d(jnp.asarray(measurements, dtype=jnp.float64))
        if p.shape != y.shape or p.shape[1] != 2:
            raise ValueError("Wind observations require matching N-by-2 arrays")
        return _condition(b, p, y, minimum_distance=minimum_distance)
