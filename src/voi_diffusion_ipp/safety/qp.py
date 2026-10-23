"""Euclidean projection onto a 2D polytope by its feasible boundary intervals.

Every feasible minimizer has zero, one, or two linearly independent active
constraints. Infeasibility is explicit; fallback minimizes a common CBF slack
by a common-slack linear program without relaxing actuator bounds. It is not declared safe.
"""

from itertools import combinations
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
    """Find the closest feasible point on each boundary line.

    On line i, all other half-planes reduce to a one-dimensional interval.
    Clamping the perpendicular foot to that interval includes intersections
    with other edges without enumerating every pair and testing it against
    every constraint. This uses quadratic rather than cubic constraint work.
    """
    norm = jnp.sum(A * A, axis=1)
    foot = u[None] - ((A @ u - b) / jnp.maximum(norm, 1e-12))[:, None] * A
    direction = jnp.stack((-A[:, 1], A[:, 0]), axis=-1)
    slope = direction @ A.T
    rhs = b[None] - foot @ A.T
    diagonal = jnp.diag_indices(A.shape[0])
    # A line is exactly parallel to itself. Matrix multiplication roundoff
    # must not turn its defining equality into a spurious interval bound.
    slope = slope.at[diagonal].set(0.)
    rhs = rhs.at[diagonal].set(0.)
    parallel = jnp.abs(slope) <= 1e-8
    bound = rhs / jnp.where(parallel, 1., slope)
    lower = jnp.max(jnp.where(slope < -1e-8, bound, -jnp.inf), axis=1)
    upper = jnp.min(jnp.where(slope > 1e-8, bound, jnp.inf), axis=1)
    coordinate = jnp.minimum(jnp.maximum(jnp.zeros_like(lower), lower), upper)
    edges = foot + coordinate[:, None] * direction
    valid = (norm > 1e-12) & jnp.all(~parallel | (rhs >= -2e-5), axis=1)
    valid &= lower <= upper + 1e-6
    candidates = jnp.concatenate((u[None], edges))
    valid = jnp.concatenate((jnp.array([True]), valid))
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
    """Solve min s >= 0, A u <= b+s, |u| <= umax by 2D minimax vertices.

    Eliminating s gives max(0, max(Au-b)) on the actuator box. A minimum
    occurs at a box corner, where two residual planes meet a box edge, or
    where three residual planes meet. For six barriers this is 84 candidates
    rather than the 165 vertices of the equivalent 3D LP. No barrier or
    actuator constraint is removed; explicit determinants retain vectorization.
    """
    corners=umax*jnp.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    i,j=np.triu_indices(len(A),1)
    difference,rhs=A[i]-A[j],b[i]-b[j]
    edges=jnp.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    a,c=difference[:,None,:],edges[None]
    det=a[...,0]*c[...,1]-a[...,1]*c[...,0]
    safe=jnp.where(jnp.abs(det)>1e-8,det,1.)
    edge_points=jnp.stack(((rhs[:,None]*c[...,1]-a[...,1]*umax)/safe,
                           (a[...,0]*umax-rhs[:,None]*c[...,0])/safe),axis=-1)
    edge_valid=jnp.abs(det)>1e-8
    triples=np.array(list(combinations(range(len(A)),3)),dtype=np.int32).reshape(-1,3)
    a=A[triples[:,0]]-A[triples[:,1]]
    c=A[triples[:,0]]-A[triples[:,2]]
    bi=b[triples[:,0]]-b[triples[:,1]]
    bj=b[triples[:,0]]-b[triples[:,2]]
    det=a[:,0]*c[:,1]-a[:,1]*c[:,0]
    safe=jnp.where(jnp.abs(det)>1e-8,det,1.)
    triple_points=jnp.stack(((bi*c[:,1]-a[:,1]*bj)/safe,
                             (a[:,0]*bj-bi*c[:,0])/safe),axis=-1)
    points=jnp.concatenate((corners,edge_points.reshape(-1,2),triple_points))
    valid=jnp.concatenate((jnp.ones(4,bool),edge_valid.ravel(),jnp.abs(det)>1e-8))
    valid&=jnp.all(jnp.abs(points)<=umax+2e-5,axis=1)
    residual=jnp.max(points@A.T-b[None],axis=1)
    optimum=jnp.min(jnp.where(valid,residual,jnp.inf))
    # Zero input and this slack are always feasible, even if all computed
    # vertices were rejected numerically. This fallback is explicitly finite.
    upper=jnp.maximum(jnp.max(-b),0.)+1.
    return jnp.maximum(jnp.where(jnp.isfinite(optimum),optimum,upper),0.)
