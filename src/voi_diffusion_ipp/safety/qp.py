"""Exact Euclidean projection onto a 2D polytope by active-set enumeration.

Every feasible minimizer has zero, one, or two linearly independent active
constraints. Infeasibility is explicit; fallback minimizes a common CBF slack
by bisection without relaxing actuator bounds. It is not declared safe.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class QPResult(NamedTuple):
    control: jax.Array
    feasible: jax.Array
    slack: jax.Array
    residual: jax.Array


def project(u, A, b):
    norm = jnp.sum(A * A, axis=1)
    foot = u[None] - ((A @ u - b) / jnp.maximum(norm, 1e-12))[:, None] * A
    i, j = np.triu_indices(A.shape[0], 1)
    a, c = A[i], A[j]
    bi, bj = b[i], b[j]
    det = a[:, 0] * c[:, 1] - a[:, 1] * c[:, 0]
    safe = jnp.where(jnp.abs(det) > 1e-8, det, 1.0)
    intersections = jnp.stack(
        ((bi * c[:, 1] - a[:, 1] * bj) / safe, (a[:, 0] * bj - bi * c[:, 0]) / safe), axis=-1
    )
    candidates = jnp.concatenate((u[None], foot, intersections))
    valid = jnp.concatenate((jnp.array([True]), norm > 1e-12, jnp.abs(det) > 1e-8))
    valid &= jnp.all(candidates @ A.T <= b[None] + 2e-5, axis=1)
    costs = jnp.where(valid, jnp.sum((candidates - u) ** 2, axis=1), jnp.inf)
    return candidates[jnp.argmin(costs)], jnp.any(valid)


def solve(u_nom, A, b, umax, fallback):
    box = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    AA = jnp.concatenate((A, box))
    bb = jnp.concatenate((b, jnp.full(4, umax)))
    u, feasible = project(u_nom, AA, bb)

    def relax(_):
        # Max row residual at zero is a guaranteed feasible upper bound.
        hi = jnp.maximum(jnp.max(-b), 0.0) + 1.0

        def body(_, interval):
            lo, hi = interval
            mid = (lo + hi) * 0.5
            _, ok = project(fallback, AA, jnp.concatenate((b + mid, jnp.full(4, umax))))
            return jnp.where(ok, lo, mid), jnp.where(ok, mid, hi)

        _, slack = jax.lax.fori_loop(0, 18, body, (jnp.array(0.0), hi))
        uf, _ = project(fallback, AA, jnp.concatenate((b + slack + 3e-5, jnp.full(4, umax))))
        return uf, slack

    u, slack = jax.lax.cond(feasible, lambda _: (u, jnp.array(0.0)), relax, None)
    return QPResult(u, feasible, slack, jnp.max(A @ u - b))
