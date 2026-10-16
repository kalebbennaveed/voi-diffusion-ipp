"""Independent not-a-knot cubic interpolant and analytic interval maximum.

Implements the construction in Safety on the Fly, Sec. III-C. The interpolation
matrix is shape-only and computed once on the host, not solved in every rollout.
"""
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp


@lru_cache(None)
def second_derivative_operator(n):
    if n < 4:
        raise ValueError('Not-a-knot cubic requires at least four samples')
    A = np.zeros((n,n)); D = np.zeros((n,n))
    A[0,:3] = [-1,2,-1]; A[-1,-3:] = [-1,2,-1]
    for i in range(1,n-1):
        A[i,i-1:i+2] = [1,4,1]
        D[i,i-1:i+2] = [6,-12,6]
    return np.linalg.solve(A,D).astype(np.float32)


def coefficients(y):
    # Unit-time intervals; extrema locations rescaled by caller if needed.
    m = jnp.asarray(second_derivative_operator(y.shape[0])) @ y
    return jnp.stack(((m[1:]-m[:-1])/6, m[:-1]/2,
                       y[1:]-y[:-1]-(2*m[:-1]+m[1:])/6, y[:-1]), axis=-1)


def maximum(y, mode='full'):
    if mode == 'discrete':
        return jnp.max(y)
    if mode == 'fast' and y.shape[0] > 7:
        start = jnp.clip(jnp.argmax(y)-3, 0, y.shape[0]-7)
        return maximum(jax.lax.dynamic_slice(y, (start,), (7,)))
    co = coefficients(y); a,b,c,d = co.T
    A = 3*a; B = 2*b; C = c
    disc = B*B - 4*A*C
    sqrt_disc = jnp.sqrt(jnp.where(disc > 1e-12, disc, 1.))
    # Stable quadratic formula with safe denominators for degenerate cubics.
    q = -.5 * (B + jnp.where(B >= 0, sqrt_disc, -sqrt_disc))
    div = lambda n, den: n / jnp.where(jnp.abs(den)>1e-8, den, 1.)
    linear = jnp.abs(A) < 1e-7
    r1 = jnp.where(linear, div(-C,B), div(q,A))
    r2 = div(C,q)
    valid1 = jnp.where(linear, jnp.abs(B)>1e-8, disc>0)
    valid2 = (~linear) & (disc>0) & (jnp.abs(q)>1e-8)
    def val(r, valid):
        v = ((a*r+b)*r+c)*r+d
        return jnp.where(valid & (r>=0) & (r<=1), v, -jnp.inf)
    return jnp.max(jnp.concatenate((y, val(r1,valid1), val(r2,valid2))))
