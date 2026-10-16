"""Double integrator with a fixed, spatial acceleration disturbance."""

import jax.numpy as jnp


def derivative(x, u, wind):
    return jnp.concatenate((x[2:], u + wind(x[:2])))


def step(x, u, wind, dt):
    """RK4 with held control; query the SAME field at each integration location."""
    k1 = derivative(x, u, wind)
    k2 = derivative(x + dt * k1 / 2, u, wind)
    k3 = derivative(x + dt * k2 / 2, u, wind)
    k4 = derivative(x + dt * k3, u, wind)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
