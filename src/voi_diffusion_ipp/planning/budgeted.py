"""Experimental 10-second causal MBD with cached fields and compact feedback.

Actual MBD and sampled spline RPCBF are retained. Only continuous field
evaluation is tabulated. Future trajectory shifts depend on earned posterior
summaries, never the outer draw index/truth. There is no inner recourse solver.
This restricted feedback policy is a budgeted approximation, not the old full
branch-bank formulation or a demonstrated one-second result.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline

from .. import clarity
from ..dynamics import step
from ..safety.rpcbf import design_policy, filter_control
from ..wind import InnerFields, WorldField
from .mbd import optimize


class Segment(NamedTuple):
    x: jax.Array
    q: jax.Array
    cost: jax.Array
    observations: jax.Array
    controls: jax.Array
    correction: jax.Array
    slack: jax.Array
    clearance: jax.Array


def reference_matrices(seconds, steps, knots=5):
    """Linear cubic interpolation, clamped to actual initial velocity and zero final velocity."""
    t = np.linspace(0, seconds, knots)
    query = np.arange(steps) * seconds / steps
    basis = CubicSpline(t, np.eye(knots), bc_type=((1, np.zeros(knots)), (1, np.zeros(knots))))
    velocity = CubicSpline(t, np.zeros(knots), bc_type=((1, 1.0), (1, 0.0)))
    return jnp.asarray(np.stack([basis(query, k) for k in range(3)])), jnp.asarray(
        np.stack([velocity(query, k) for k in range(3)])
    )


def segment(
    x0, q0, references, controller_belief, inner, world, noise, env, cfg, tracking, backup=False
):
    wind = lambda p: world.at(p)[0]

    def tick(carry, item):
        x, q, cost = carry
        reference, eps = item
        mean = controller_belief.mean_at(x[:2])[0]
        if tracking:
            p, v, a = reference
            desired = a + 1.2 * (p - x[:2]) + 1.8 * (v - x[2:])
        else:
            desired = reference
        nominal = jnp.clip(desired + cfg.linear_drag * x[2:] - mean, -cfg.umax, cfg.umax)
        nominal = jnp.where(backup, design_policy(x, controller_belief, cfg), nominal)
        safe, info = filter_control(x, nominal, controller_belief, inner, env, cfg)
        xn = step(x, safe, wind, cfg.dt, cfg.linear_drag)
        qn = clarity.step(
            q, x[:2], xn[:2], env.cells, cfg.sensor_sigma, cfg.sensor_R, env.process_noise, cfg.dt
        )
        incurred = (
            0.5 * cfg.dt * (clarity.deficit(q, env.targets) + clarity.deficit(qn, env.targets))
        )
        observation = world.at(xn[:2])[0] + cfg.observation_std * eps
        mids = jax.vmap(lambda t: step(x, safe, wind, t, cfg.linear_drag))(
            cfg.dt * jnp.array([0.25, 0.5, 0.75, 1.0])
        )
        gap = jnp.min(jax.vmap(env.clearance)(mids))
        return (xn, qn, cost + incurred), (
            xn,
            qn,
            observation,
            safe,
            info.correction,
            info.slack,
            gap,
        )

    (_, _, cost), (xs, qs, obs, us, corrections, slacks, gaps) = jax.lax.scan(
        tick, (x0, q0, jnp.array(0.0)), (references, noise)
    )
    return Segment(
        jnp.concatenate((x0[None], xs)),
        jnp.concatenate((q0[None], qs)),
        cost,
        obs,
        us,
        corrections,
        slacks,
        gaps,
    )


def split_vector(vector, cfg):
    first = cfg.umax * jnp.tanh(vector[:4].reshape(2, 2))
    grid = jnp.linspace(0, 1, cfg.stage_steps)
    controls = jax.vmap(
        lambda u: jnp.interp(grid, jnp.array([0.0, 1.0]), u), in_axes=1, out_axes=1
    )(first)
    future = jnp.tanh(vector[4:12].reshape(4, 2))
    gain = vector[12:].reshape(2, 3)
    return controls, future, gain


def future_reference(vector, terminal, posterior, env, matrices, cfg):
    _, normalized, gain = split_vector(vector, cfg)
    mean, cov = posterior.predict(env.probe)
    features = jnp.r_[mean[0] / cfg.wind_std, jnp.sqrt(cov[0, 0, 0]) / cfg.wind_std]
    # Same learned function for every scenario. No access to world identity.
    shift = 0.6 * jnp.tanh(gain @ features)
    midpoint = 0.5 * (env.bounds[0] + env.bounds[1])
    scale = 0.5 * (env.bounds[1] - env.bounds[0]) - 0.12
    positions = (
        midpoint + scale * normalized + shift * jnp.sin(jnp.linspace(0, jnp.pi, 5)[1:])[:, None]
    )
    positions = jnp.clip(positions, env.bounds[0] + 0.08, env.bounds[1] - 0.08)
    knots = jnp.concatenate((terminal[None, :2], positions))
    M, V = matrices
    return jnp.swapaxes(M @ knots + V[:, :, None] * terminal[None, None, 2:], 0, 1)


def evaluate(vector, x, q, belief, cache, noise, env, cfg, anticipate=True, exact=False, eps=None):
    """Causal two-stage rollouts; current-belief ablation omits hypothetical updates only."""
    matrices = reference_matrices(cfg.future_steps * cfg.dt, cfg.future_steps)
    if exact:
        current_mean = belief
        all_inner = belief.inner_fields(eps)
    else:
        current_mean, all_inner = cache.tabulate(belief)
    first_controls, _, _ = split_vector(vector, cfg)
    inner = InnerFields(all_inner.basis, all_inner.coefficients[cfg.worlds :])

    def one(inputs):
        coeff, observation_noise = inputs
        world = WorldField(all_inner.basis, coeff)
        first = segment(
            x,
            q,
            first_controls,
            current_mean,
            inner,
            world,
            observation_noise[: cfg.stage_steps],
            env,
            cfg,
            False,
        )
        posterior = belief.condition(first.x[1:, :2], first.observations) if anticipate else belief
        if exact:
            future_mean, all_future = posterior, posterior.inner_fields(eps)
        else:
            future_mean, all_future = cache.tabulate(posterior)
        future_inner = InnerFields(all_future.basis, all_future.coefficients[cfg.worlds :])
        reference = future_reference(vector, first.x[-1], posterior, env, matrices, cfg)
        future = segment(
            first.x[-1],
            first.q[-1],
            reference,
            future_mean,
            future_inner,
            world,
            observation_noise[cfg.stage_steps :],
            env,
            cfg,
            True,
        )
        cost = first.cost + future.cost
        valid = jnp.isfinite(cost)
        if cfg.reject_collisions:
            valid &= (first.clearance.min() >= 0) & (future.clearance.min() >= 0)
        return jnp.where(valid, cost, jnp.inf), (first, future)

    costs, trajectories = jax.lax.map(one, (all_inner.coefficients[: cfg.worlds], noise))
    return costs.mean(), trajectories


def make_planner(cfg, anticipate=True):
    @jax.jit
    def plan(x, q, belief, cache, noise, env, initials, key):
        objective = lambda vector: evaluate(
            vector, x, q, belief, cache, noise, env, cfg, anticipate
        )[0]
        costs = jax.lax.map(objective, initials)
        initial = initials[jnp.argmin(costs)]
        result = optimize(objective, initial, key, cfg)
        return result._replace(evaluations=result.evaluations + len(initials))

    return plan


def warm_starts(x, env, cfg):
    """Shared geometric seeds, including staying. MBD selects by task cost only."""
    candidates = []
    # Both region goals remain available regardless of current location/clarity.
    for goal_index, goal in enumerate([env.roi_a, env.roi_b, x[:2]]):
        delta = goal - x[:2]
        direction = delta / jnp.maximum(jnp.linalg.norm(delta), 0.01)
        normal = jnp.array([-direction[1], direction[0]])
        for offset in [0.0, -1.5, 1.5] if goal_index < 2 else [0.0]:
            fractions = jnp.array([0.25, 0.5, 0.8, 1.0])
            positions = (
                x[:2]
                + fractions[:, None] * delta
                + offset * jnp.sin(jnp.pi * fractions)[:, None] ** 2 * normal
            )
            scale = 0.5 * (env.bounds[1] - env.bounds[0]) - 0.12
            mid = 0.5 * (env.bounds[0] + env.bounds[1])
            latent = jnp.arctanh(jnp.clip((positions - mid) / scale, -0.97, 0.97))
            accel = 0.6 * direction + 0.15 * jnp.sign(offset) * normal
            first = jnp.tile(jnp.arctanh(accel), (2, 1)).ravel()
            candidates.append(jnp.r_[first, latent.ravel(), jnp.zeros(6)])
    return jnp.stack(candidates)
