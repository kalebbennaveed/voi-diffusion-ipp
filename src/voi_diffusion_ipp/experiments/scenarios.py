"""Legacy controls and whole-field persistent monitoring with held-out physical wind."""

import jax
import jax.numpy as jnp

from ..environment import Environment
from ..wind import prior
from ..wind.gp import LocalizedBasis


def make_scenario(cfg):
    if cfg.whole_field:
        return field_scenario(cfg)
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


def new_poam(cfg, centers):
    from ..wind.poam import POAM

    support = None
    if cfg.whole_field:
        dx, dy = 4.8 / cfg.grid_nx, 3.7 / cfg.grid_ny
        sx, sy = jnp.meshgrid(
            jnp.linspace(-2.4 - 1.5 * dx, 2.4 + 1.5 * dx, cfg.grid_nx + 4),
            jnp.linspace(-1.85 - 1.5 * dy, 1.85 + 1.5 * dy, cfg.grid_ny + 4),
        )
        support = jnp.stack((sx.ravel(), sy.ravel()), axis=-1)
        if cfg.poam_support_x:
            sx, sy = jnp.meshgrid(
                jnp.linspace(-2.7, 2.7, cfg.poam_support_x),
                jnp.linspace(-2.15, 2.15, cfg.poam_support_y),
            )
            support = jnp.stack((sx.ravel(), sy.ravel()), axis=-1)
    return POAM(
        centers,
        jax.random.PRNGKey(600000),
        noise_std=cfg.observation_std,
        prior_std=cfg.wind_std,
        spectral_count=cfg.poam_spectral_nodes,
        prior_support=support,
    )


def field_scenario(cfg):
    """Uniform-area grid over the WHOLE rectangular monitoring field.

    Flight obstacles do not delete ground cells: they remain remotely observable
    through the sensing footprint. Every cell contributes equally to mean deficit.
    """
    bounds = jnp.array([[-2.4, -1.85], [2.4, 1.85]])
    dx, dy = 4.8 / cfg.grid_nx, 3.7 / cfg.grid_ny
    xx, yy = jnp.meshgrid(
        jnp.linspace(-2.4 + dx / 2, 2.4 - dx / 2, cfg.grid_nx),
        jnp.linspace(-1.85 + dy / 2, 1.85 - dy / 2, cfg.grid_ny),
    )
    cells = jnp.stack((xx.ravel(), yy.ravel()), axis=-1)
    hotspots = jnp.array(
        [
            [-1.65, -0.95],
            [-1.6, 1.0],
            [-0.5, 0.1],
            [0.1, -1.2],
            [0.5, 1.15],
            [1.55, -0.6],
            [1.65, 0.95],
        ]
    )
    need = jnp.exp(-jnp.sum((cells[:, None] - hotspots[None]) ** 2, axis=-1) / (2 * 0.42**2))
    targets = 0.45 + 0.35 * jnp.max(need, axis=1)
    layouts = {
        "field_vortices": [[-0.55, -0.65, 0.32], [0.35, 0.65, 0.40], [1.1, -0.7, 0.30]],
        "field_shear": [[-0.8, 0.55, 0.38], [0.2, -0.55, 0.38], [1.15, 0.65, 0.33]],
        "field_mixed": [[-1.0, -0.7, 0.30], [-0.2, 0.65, 0.38], [0.9, -0.5, 0.42]],
        "persistent_archipelago": [
            [-1.2, -0.65, 0.30],
            [-0.7, 0.65, 0.32],
            [0.05, -0.35, 0.31],
            [0.65, 0.85, 0.32],
            [1.25, -0.75, 0.35],
            [1.65, 0.55, 0.24],
        ],
        "persistent_channels": [
            [-0.9, 0.9, 0.39],
            [-0.9, -0.55, 0.39],
            [0.7, 0.5, 0.40],
            [0.7, -1.0, 0.38],
            [1.7, 1.05, 0.25],
        ],
        "persistent_mosaic": [
            [-1.2, -0.8, 0.28],
            [-0.5, 0.55, 0.35],
            [0.4, -0.55, 0.33],
            [1.35, 0.75, 0.34],
            [1.6, -0.9, 0.25],
        ],
    }
    if cfg.scenario not in layouts:
        raise ValueError(f"Unknown whole-field scenario {cfg.scenario}")
    obstacles = jnp.array(layouts[cfg.scenario])
    # These legacy landmarks are unused by whole-field policies and evaluation.
    if cfg.scenario == "persistent_channels":
        targets = 0.45 + 0.25 * jnp.exp(-(((jnp.abs(cells[:, 1]) - 0.95) / 0.48) ** 2))
    elif cfg.scenario == "persistent_mosaic":
        targets = 0.45 + 0.27 * (
            0.5 + 0.5 * jnp.sin(1.4 * cells[:, 0]) * jnp.cos(2.0 * cells[:, 1])
        )
    if cfg.clarity_decay_pattern == "patchy":
        process = cfg.clarity_decay * (0.5 + jnp.max(need, axis=1))
    elif cfg.clarity_decay_pattern == "striped":
        process = cfg.clarity_decay * (0.5 + jnp.sin(1.8 * cells[:, 0] + 0.7 * cells[:, 1]) ** 2)
    else:
        process = jnp.full(len(cells), cfg.clarity_decay)
    env = Environment(
        cells, targets, obstacles, bounds, hotspots[0], hotspots[1], jnp.zeros(2), process
    )
    zx, zy = jnp.meshgrid(
        jnp.linspace(-2.2, 2.2, cfg.poam_inducing_x), jnp.linspace(-1.65, 1.65, cfg.poam_inducing_y)
    )
    z = jnp.stack((zx.ravel(), zy.ravel()), axis=-1)
    belief = new_poam(cfg, z).planning_belief()
    x = jnp.array([-1.85, -0.05, 0.0, 0.0])
    # Spatially varying initial certainty, identical across methods and seeds.
    q = 0.12 + 0.20 * jnp.exp(-jnp.sum((cells - x[:2]) ** 2, axis=1) / (2 * 0.6**2))
    if cfg.initial_clarity == "maintained":
        q = jnp.minimum(targets + 0.04, 0.95)
    return env, belief, x, q


def make_truth(cfg, prior_belief, seed):
    from ..wind import WorldField

    if cfg.whole_field:
        from ..wind.physical import physical_wind

        return physical_wind(seed, cfg.scenario, cfg.wind_std)
    truth_belief = prior_belief._replace(
        basis=prior_belief.basis._replace(independent=jnp.array(False))
    )
    return WorldField(
        truth_belief.basis,
        truth_belief.sample_field(jax.random.fold_in(jax.random.PRNGKey(seed), 100000), 1)[0],
    )
