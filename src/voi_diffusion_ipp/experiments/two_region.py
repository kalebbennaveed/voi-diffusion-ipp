"""User-motivated large-region monitoring and a held-out analytic passage jet."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..environment import Environment
from ..wind.poam import POAM


class PassageWind(NamedTuple):
    tangent: jax.Array
    normal: jax.Array
    amplitude: jax.Array
    cross_fraction: jax.Array
    longitudinal_width: jax.Array
    transverse_width: jax.Array

    def at(self, positions):
        p = jnp.atleast_2d(positions)
        axial, lateral = p @ self.tangent, p @ self.normal
        envelope = jnp.exp(
            -0.5 * ((axial / self.longitudinal_width) ** 4 + (lateral / self.transverse_width) ** 2)
        )
        return (
            self.amplitude * envelope[:, None] * (self.tangent + self.cross_fraction * self.normal)
        )


def outer_survey_positions(env, count=21):
    """Declared historical flight lines; geometry only, independent of wind values."""
    tangent = (env.roi_b - env.roi_a) / jnp.linalg.norm(env.roi_b - env.roi_a)
    normal = jnp.array([-tangent[1], tangent[0]])
    s = jnp.linspace(0, 1, count)
    return jnp.concatenate(
        [
            env.roi_a
            + s[:, None] * (env.roi_b - env.roi_a)
            + h * (16 * s**2 * (1 - s) ** 2)[:, None] * normal
            for h in [-1.5, 1.5]
        ]
    )


def make_two_region(spec, cfg, amplitude=1.8, poam_key=600000):
    """Every cell evolves; only the two large regions have positive targets.

    POAM uses the full attentive kernel and residual covariance, including online
    kernel training and inducing relocation when the caller assimilates data.
    The analytic physical field is never supplied to the estimator or planner.
    """
    bounds = jnp.asarray(spec["bounds"])
    nx, ny = spec["grid_shape"]
    spacing = (bounds[1] - bounds[0]) / jnp.array([nx, ny])
    xx, yy = jnp.meshgrid(
        jnp.linspace(bounds[0, 0] + spacing[0] / 2, bounds[1, 0] - spacing[0] / 2, nx),
        jnp.linspace(bounds[0, 1] + spacing[1] / 2, bounds[1, 1] - spacing[1] / 2, ny),
    )
    cells = jnp.stack((xx.ravel(), yy.ravel()), axis=-1)
    a, b = jnp.array(spec["region_a"]), jnp.array(spec["region_b"])
    tangent = (b - a) / jnp.linalg.norm(b - a)
    normal = jnp.array([-tangent[1], tangent[0]])
    masks = jnp.all(
        jnp.abs(cells[:, None] - jnp.stack((a, b))[None]) <= jnp.array(spec["region_size"]) / 2,
        axis=-1,
    )
    targets = jnp.where(jnp.any(masks, axis=1), spec["target"], 0.0)
    centers = jnp.array([-1.0, 1.0])[:, None] * normal * spec["obstacle_normal_offset"]
    obstacles = jnp.concatenate((centers, jnp.full((2, 1), spec["obstacle_radius"])), axis=1)
    env = Environment(
        cells,
        targets,
        obstacles,
        bounds,
        a,
        b,
        jnp.zeros(2),
        jnp.full(len(cells), cfg.clarity_decay),
    )
    zx, zy = jnp.meshgrid(
        jnp.linspace(bounds[0, 0] + 0.2, bounds[1, 0] - 0.2, cfg.poam_inducing_x),
        jnp.linspace(bounds[0, 1] + 0.2, bounds[1, 1] - 0.2, cfg.poam_inducing_y),
    )
    sx, sy = jnp.meshgrid(
        jnp.linspace(bounds[0, 0] - 0.2, bounds[1, 0] + 0.2, 29),
        jnp.linspace(bounds[0, 1] - 0.2, bounds[1, 1] + 0.2, 23),
    )
    poam = POAM(
        jnp.stack((zx.ravel(), zy.ravel()), axis=-1),
        jax.random.PRNGKey(poam_key),
        noise_std=cfg.observation_std,
        prior_std=cfg.wind_std,
        spectral_count=cfg.poam_spectral_nodes,
        prior_support=jnp.stack((sx.ravel(), sy.ravel()), axis=-1),
    )
    truth = PassageWind(
        tangent, normal, jnp.array(amplitude), jnp.array(0.2), jnp.array(1.1), jnp.array(0.35)
    )
    return (
        env,
        poam,
        jnp.r_[a, jnp.zeros(2)],
        jnp.full(len(cells), spec["initial_clarity"], dtype=jnp.float32),
        truth,
    )
