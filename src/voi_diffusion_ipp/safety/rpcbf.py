"""Sampled robust policy-value CBF, spline extrema, autodiff and CBF-QP.

No true-world argument is accepted. Both backup compensation and inner samples
come from the currently available belief. Finite horizon/sample safety is empirical.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..dynamics import step
from ..wind import InnerFields
from .qp import solve
from .spline import maximum


class FilterInfo(NamedTuple):
    value: jax.Array
    gradient: jax.Array
    correction: jax.Array
    feasible: jax.Array
    slack: jax.Array
    residual: jax.Array
    admissible_fraction: jax.Array


def design_policy(x, belief, cfg):
    mean = (belief.basis.features(x[:2]) @ belief.mean)[0]
    return jnp.clip(-cfg.backup_gain * x[2:] - mean, -cfg.umax, cfg.umax)


def values(x, belief, inner, env, cfg):
    if not isinstance(inner, InnerFields):
        raise TypeError("RPCBF requires belief-derived InnerFields, never WorldField")

    def one(coeff):
        wind = lambda p: (inner.basis.features(p) @ coeff)[0]

        def body(x, _):
            u = design_policy(x, belief, cfg)
            xn = step(x, u, wind, cfg.backup_dt)
            return xn, env.violations(xn, cfg.safety_margin)

        _, h = jax.lax.scan(body, x, None, length=cfg.backup_steps)
        h = jnp.concatenate((env.violations(x, cfg.safety_margin)[None], h))
        return jax.vmap(partial(maximum, mode=cfg.spline), in_axes=1)(h)

    return jnp.max(jax.vmap(one)(inner.coefficients), axis=0)


def constraints(x, belief, inner, env, cfg):
    f = lambda z: values(z, belief, inner, env, cfg)
    B = f(x)
    grad = jax.jacfwd(f)(x)
    # All sampled instantaneous winds, not just the value-maximizing field.
    winds = jnp.einsum("m,nmc->nc", inner.basis.features(x[:2])[0], inner.coefficients)
    worst_drift = grad[:, :2] @ x[2:] + jnp.max(grad[:, 2:] @ winds.T, axis=1)
    return grad[:, 2:], -worst_drift - cfg.cbf_alpha * B, B, grad


def filter_control(x, u_nom, belief, inner, env, cfg):
    A, b, B, grad = constraints(x, belief, inner, env, cfg)
    sol = solve(u_nom, A, b, cfg.umax, design_policy(x, belief, cfg))
    # Fixed-grid estimate of admissible area, diagnostic only.
    grid = jnp.linspace(-cfg.umax, cfg.umax, 7)
    ux, uy = jnp.meshgrid(grid, grid)
    us = jnp.stack((ux.ravel(), uy.ravel()), axis=-1)
    area = jnp.mean(jnp.all(us @ A.T <= b[None], axis=1))
    return sol.control, FilterInfo(
        B, grad, jnp.linalg.norm(sol.control - u_nom), sol.feasible, sol.slack, sol.residual, area
    )
