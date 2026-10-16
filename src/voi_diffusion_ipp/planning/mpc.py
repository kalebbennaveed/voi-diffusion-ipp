from dataclasses import replace
from functools import lru_cache

import jax
import jax.numpy as jnp

from ..dynamics import step
from .mbd import optimize
from .two_stage import decode, make_buffers, objective, route

METHODS = (
    "proposed",
    "passive",
    "weighted_ig",
    "heuristic",
    "draft_selection",
    "cem",
    "mppi",
    "oracle",
    "current_belief",
    "fixed",
)


def planner(cfg, method):
    return _planner(
        replace(cfg, scenario="corridor", seeds=(), methods=(), mission_horizons=1), method
    )


@lru_cache(None)
def _planner(cfg, method):
    if method not in METHODS:
        raise ValueError(method)
    mode = "proposed" if method in ("proposed", "cem", "mppi") else method
    if method == "oracle":
        mode = "current_belief"
    if method == "draft_selection":
        mode = "passive"
    alg = method if method in ("cem", "mppi") else "mbd"

    @jax.jit
    def plan(x, q, belief, key, initial, env):
        ks, ko = jax.random.split(key)
        buffers = make_buffers(belief, ks, cfg)
        fn = lambda y: objective(y, x, q, belief, buffers, env, cfg, mode)
        extra = 0
        if cfg.warm_starts > 1:
            seeds = jnp.stack(
                (
                    initial,
                    initial_policy(
                        x,
                        q,
                        belief,
                        env,
                        cfg,
                        task_goals(x, q, env, cfg)[1] if cfg.whole_field else env.roi_a,
                    ),
                    initial_policy(x, q, belief, env, cfg, x[:2]),
                )
            )
            costs = jax.lax.map(fn, seeds)
            initial = seeds[jnp.argmin(costs)]
            extra = 3
        result = optimize(fn, initial, ko, cfg, alg)
        if method == "draft_selection":
            from ..baselines.draft import select_candidates

            result = select_candidates(result, x, q, belief, buffers, env, cfg)
        return result._replace(evaluations=result.evaluations + extra)

    return plan


def task_goals(x, q, env, cfg):
    """Data-dependent warm starts from distributed sensing benefit, no fixed ROI."""
    process = cfg.clarity_decay if env.process_noise is None else env.process_noise
    horizon = (cfg.stage_steps + cfg.future_steps) * cfg.dt
    unobserved = q / (1 + process * q * horizon)
    need = jnp.maximum(env.targets - unobserved, 0.0)
    dist2 = jnp.sum((env.cells[:, None] - env.cells[None]) ** 2, axis=-1)
    benefit = jnp.exp(-dist2 / cfg.sensor_sigma**2) @ need
    distance = jnp.linalg.norm(env.cells - x[:2], axis=1)
    clearance = jax.vmap(lambda p: env.clearance(jnp.r_[p, jnp.zeros(2)]))(env.cells)
    score = jnp.where(clearance > cfg.safety_margin, benefit / (1 + 0.25 * distance), -jnp.inf)
    first = env.cells[jnp.argmax(score)]
    separated = jnp.linalg.norm(env.cells - first, axis=1) > 0.8
    second = env.cells[jnp.argmax(jnp.where(separated, score, -jnp.inf))]
    return jnp.stack((first, second))


def initial_policy(x, q, belief, env, cfg, first_goal=None):
    """Task-only seed for every solver. No posterior outcomes or truths are used."""
    task = task_goals(x, q, env, cfg) if cfg.whole_field else jnp.stack((env.roi_b, env.roi_a))
    if cfg.control_representation == "waypoint":
        first = task[0] if first_goal is None else first_goal
        goals = jnp.stack((first, task[1], task[0]))[: 1 + cfg.branches]
        scale = jnp.array([cfg.waypoint_x_limit, cfg.waypoint_y_limit])
        latent = jnp.arctanh(jnp.clip(goals / scale, -0.95, 0.95))
        return jnp.tile(latent[:, None], (1, cfg.knots, 1)).ravel()
    wind = lambda p: belief.mean_at(p)[0]

    def toward(x, goal, n):
        def body(x, _):
            u = jnp.clip(
                1.2 * (goal - x[:2]) + (cfg.linear_drag - 1.8) * x[2:] - wind(x[:2]),
                -0.92 * cfg.umax,
                0.92 * cfg.umax,
            )
            return step(x, u, wind, cfg.dt, cfg.linear_drag), u

        xf, us = jax.lax.scan(body, x, None, length=n)
        idx = jnp.linspace(0, n - 1, cfg.knots)
        knots = jax.vmap(lambda u: jnp.interp(idx, jnp.arange(n), u), in_axes=1, out_axes=1)(us)
        return xf, jnp.arctanh(knots / cfg.umax)

    xL, first = toward(x, task[0] if first_goal is None else first_goal, cfg.stage_steps)
    _, a = toward(xL, task[1], cfg.future_steps)
    _, b = toward(xL, task[0], cfg.future_steps)
    futures = jnp.stack((a, b))[: cfg.branches]
    return jnp.concatenate((first[None], futures)).ravel()


def next_initial(previous, x, q, belief, env, cfg, updates=None):
    # Reuse the observation-selected future as a warm first period. Remaining
    # branches are task-only seeds; the whole vector is optimized again.
    base = initial_policy(x, q, belief, env, cfg).reshape((1 + cfg.branches, cfg.knots, 2))
    prior = previous.reshape(base.shape)
    b = route(x, q, belief, env, cfg, decode(previous, cfg)[1], updates=updates)
    # First period only covers the first TL/TF of previous future.
    times = jnp.linspace(0, min(1, cfg.stage_steps / cfg.future_steps), cfg.knots)
    knots = jax.vmap(
        lambda u: jnp.interp(times, jnp.linspace(0, 1, cfg.knots), u), in_axes=1, out_axes=1
    )(prior[b + 1])
    return base.at[0].set(knots).ravel()
