"""Planar translation with optional isotropic linear air-relative drag.

v_dot = u - drag*v + d(p). For drag > 0, d = drag*w_air is the
wind-induced acceleration learned by POAM. The default recovers the original
disturbed double integrator. This is not a full helicopter flight model.
"""

import jax.numpy as jnp


def derivative(x, u, wind, drag=0.0):
    return jnp.concatenate((x[2:], u - drag * x[2:] + wind(x[:2])))


def step(x, u, wind, dt, drag=0.0):
    """RK4 with held control; query the SAME field at each integration location."""
    k1 = derivative(x, u, wind, drag)
    k2 = derivative(x + dt * k1 / 2, u, wind, drag)
    k3 = derivative(x + dt * k2 / 2, u, wind, drag)
    k4 = derivative(x + dt * k3, u, wind, drag)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
