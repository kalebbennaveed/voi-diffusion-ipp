"""Clarity dynamics from the supplied Stein paper, Eq. (12)."""

import jax.numpy as jnp


def sensing(position, cells, sigma):
    return jnp.exp(-jnp.sum((cells - position) ** 2, axis=-1) / (2 * sigma**2))


def rate(q, position, cells, sigma, R, Q):
    C = sensing(position, cells, sigma)
    return C**2 / R * (1 - q) ** 2 - Q * q**2


def step(q, p0, p1, cells, sigma, R, Q, dt):
    """RK4 along executed segment; endpoint/midpoint quadrature of sensing."""
    f = lambda qq, pp: rate(qq, pp, cells, sigma, R, Q)
    k1 = f(q, p0)
    k2 = f(q + dt * k1 / 2, (p0 + p1) / 2)
    k3 = f(q + dt * k2 / 2, (p0 + p1) / 2)
    k4 = f(q + dt * k3, p1)
    return jnp.clip(q + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, 0, 1)


def deficit(q, target):
    return jnp.mean(jnp.maximum(target - q, 0))
