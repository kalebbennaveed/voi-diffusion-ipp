"""Exact Euclidean projection onto a 2D polytope by active-set enumeration.

Every feasible minimizer has zero, one, or two linearly independent active
constraints. Infeasibility is explicit; fallback minimizes a common CBF slack
by a common-slack linear program without relaxing actuator bounds. It is not declared safe.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from itertools import combinations


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
        slack=minimum_slack(A,b,umax)
        uf, _ = project(fallback, AA, jnp.concatenate((b + slack + 3e-5, jnp.full(4, umax))))
        return uf, slack

    u, slack = jax.lax.cond(feasible, lambda _: (u, jnp.array(0.0)), relax, None)
    return QPResult(u, feasible, slack, jnp.max(A @ u - b))


def minimum_slack(A,b,umax):
    """Solve min s >= 0, A u <= b+s, |u| <= umax by LP vertices.

    The actuator box makes the minimum attainable at a 3D polyhedron vertex.
    This is the same common-slack problem previously solved by 18 bisections.
    Explicit determinants avoid batched LAPACK and preserve vectorization.
    """
    box=jnp.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    matrix=jnp.concatenate((jnp.c_[A,-jnp.ones(len(A))],jnp.c_[box,jnp.zeros(4)],jnp.array([[0.,0.,-1.]])))
    bound=jnp.r_[b,jnp.full(4,umax),0.]
    triples=np.array(list(combinations(range(len(bound)),3)))
    rows=matrix[triples];rhs=bound[triples]
    c0=jnp.cross(rows[:,1],rows[:,2]);c1=jnp.cross(rows[:,2],rows[:,0]);c2=jnp.cross(rows[:,0],rows[:,1])
    determinant=jnp.sum(rows[:,0]*c0,axis=1)
    valid=jnp.abs(determinant)>1e-8
    vertices=(rhs[:,0,None]*c0+rhs[:,1,None]*c1+rhs[:,2,None]*c2)/jnp.where(valid,determinant,1.)[:,None]
    valid&=jnp.all(vertices@matrix.T<=bound[None]+2e-5,axis=1)
    optimum=jnp.min(jnp.where(valid,vertices[:,2],jnp.inf))
    # Zero input and this slack are always feasible, even if all computed
    # vertices were rejected numerically. This fallback is explicitly finite.
    upper=jnp.maximum(jnp.max(-b),0.)+1.
    return jnp.maximum(jnp.where(jnp.isfinite(optimum),optimum,upper),0.)
