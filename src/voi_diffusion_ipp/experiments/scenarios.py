"""Three fixed-physics scenarios. Truth uses the same correlated basis in all ablations."""

import jax.numpy as jnp

from ..environment import Environment
from ..wind import prior
from ..wind.gp import LocalizedBasis


def make_scenario(cfg):
    a = jnp.array([1.15, 0.0])
    b = jnp.array([-1.45, -0.6])
    offsets = jnp.array(
        [
            [-0.16, -0.16],
            [0, -0.16],
            [0.16, -0.16],
            [-0.16, 0],
            [0, 0],
            [0.16, 0],
            [-0.16, 0.16],
            [0, 0.16],
            [0.16, 0.16],
        ]
    )
    cells = jnp.concatenate((a + offsets, b + offsets[:4]))
    targets = jnp.concatenate((jnp.full(9, 0.8), jnp.full(4, 0.65)))
    obstacles = jnp.array([[0.0, 1.05, cfg.obstacle_radius], [0.0, -1.05, cfg.obstacle_radius]])
    bounds = jnp.array([[-2.4, -1.85], [2.4, 1.85]])
    probe = jnp.array([-0.55, 0.0])
    if cfg.scenario == "irrelevant_uncertainty":
        # Useful monitoring is entirely on the left; uncertainty to the right has no task value.
        cells = cells.at[:9].set(jnp.array([-1.35, 0.5]) + offsets)
        a = jnp.array([-1.35, 0.5])
    elif cfg.scenario not in ("corridor", "no_worthwhile_learning", "localized_corridor"):
        raise ValueError(f"Unknown scenario {cfg.scenario}")
    env = Environment(cells, targets, obstacles, bounds, a, b, probe)
    cx, cy = jnp.meshgrid(jnp.linspace(-2, 2, 5), jnp.linspace(-1.5, 1.5, 3))
    centers = jnp.stack((cx.ravel(), cy.ravel()), axis=-1)
    noise = cfg.observation_std if cfg.scenario != "no_worthwhile_learning" else 10.0
    belief = prior(centers, cfg.wind_length, cfg.wind_std, noise, cfg.independent_cells)
    if cfg.scenario == "irrelevant_uncertainty":
        # Low uncertainty on useful left routes, large uncertainty beyond the passage.
        std = jnp.where(centers[:, 0] < 0, 0.08, 0.9)
        belief = belief._replace(covariance=jnp.diag(std**2))
    if cfg.scenario == "localized_corridor":
        centers = jnp.array([[0.0, 0.0]])
        belief = prior(centers, cfg.wind_length, cfg.wind_std, noise)
        belief = belief._replace(
            basis=LocalizedBasis(centers, jnp.array(cfg.wind_length), jnp.array(False))
        )
    x = jnp.array([-1.45, -0.15, 0.0, 0.0])
    q = jnp.full(cells.shape[0], 0.15)
    return env, belief, x, q
