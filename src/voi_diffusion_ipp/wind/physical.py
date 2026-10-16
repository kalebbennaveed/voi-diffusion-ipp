"""Held-out analytic wind fields, independent of the estimator's kernel and inducing set.

A fixed draw of vortex/jet parameters defines the complete episode. Only the
simulator sees this object; POAM gets noisy measurements at executed positions.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class PhysicalWind(NamedTuple):
    centers: jax.Array
    widths: jax.Array
    strengths: jax.Array
    bias: jax.Array
    shear: jax.Array

    def at(self, positions):
        x = jnp.atleast_2d(positions)
        r = (x[:, None] - self.centers[None]) / self.widths[None, :, None]
        radial = jnp.exp(-0.5 * jnp.sum(r**2, axis=-1))
        rotations = jnp.stack((-r[..., 1], r[..., 0]), axis=-1)
        vortices = jnp.sum(radial[..., None] * rotations * self.strengths[None, :, None], axis=1)
        jet = jnp.stack((jnp.tanh(2.5 * x[:, 1]), 0.4 * jnp.sin(1.5 * x[:, 0])), axis=1)
        return self.bias[None] + vortices + self.shear * jet


def physical_wind(seed, scenario, std):
    root = jax.random.fold_in(jax.random.PRNGKey(seed), 100000)
    k1, k2, k3, k4 = jax.random.split(root, 4)
    centers = jax.random.uniform(
        k1, (5, 2), minval=jnp.array([-1.8, -1.3]), maxval=jnp.array([1.8, 1.3])
    )
    widths = jax.random.uniform(k2, (5,), minval=0.35, maxval=1.05)
    strengths = std * jax.random.normal(k3, (5,))
    bias = std * 0.25 * jax.random.normal(k4, (2,))
    shear = (
        std
        * {
            "field_vortices": 0.15,
            "field_shear": 0.8,
            "field_mixed": 0.45,
            "persistent_archipelago": 0.15,
            "persistent_channels": 0.8,
            "persistent_mosaic": 0.45,
        }[scenario]
    )
    return PhysicalWind(centers, widths, strengths, bias, jnp.array(shear))
