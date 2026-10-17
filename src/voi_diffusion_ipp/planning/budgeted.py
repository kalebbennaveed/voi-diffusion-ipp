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
    x0,
    q0,
    references,
    controller_belief,
    inner,
    world,
    noise,
    env,
    cfg,
    tracking,
    backup=False,
    nominal_wind_aware=True,
    apply_safety=True,
):
    wind = lambda p: world.at(p)[0]

    def tick(carry, item):
        x, q, cost = carry
        reference, eps = item
        mean = controller_belief.mean_at(x[:2])[0] if nominal_wind_aware else jnp.zeros(2)
        if tracking:
            p, v, a = reference
            desired = a + 1.2 * (p - x[:2]) + 1.8 * (v - x[2:])
        else:
            desired = reference
        nominal = jnp.clip(desired + cfg.linear_drag * x[2:] - mean, -cfg.umax, cfg.umax)
        nominal = jnp.where(backup, design_policy(x, controller_belief, cfg), nominal)
        if apply_safety:
            safe, info = filter_control(x, nominal, controller_belief, inner, env, cfg)
            correction, slack = info.correction, info.slack
        else:
            safe, correction, slack = nominal, jnp.array(0.0), jnp.array(0.0)
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
            correction,
            slack,
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
    if cfg.budgeted_policy == "branches":
        return controls, jnp.tanh(vector[4:20].reshape(2, 4, 2)), vector[20:24]
    future = jnp.tanh(vector[4:12].reshape(4, 2))
    gain = vector[12:].reshape(2, 3)
    return controls, future, gain


def future_reference(vector, terminal, posterior, env, matrices, cfg):
    _, normalized, gain = split_vector(vector, cfg)
    mean, cov = posterior.predict(env.probe)
    features = jnp.r_[mean[0] / cfg.wind_std, jnp.sqrt(cov[0, 0, 0]) / cfg.wind_std]
    # Same learned function for every scenario. No access to world identity.
    if cfg.budgeted_policy == "branches":
        # A discrete route choice is valid for gradient-free MBD. Only the
        # desired reference switches; actual robot and clarity states continue.
        branch = (gain @ jnp.r_[1.0, features] > 0).astype(jnp.int32)
        normalized = normalized[branch]
        shift = jnp.zeros(2)
    else:
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
        posterior = (
            belief.condition(
                first.x[1:, :2], first.observations, relocate=cfg.hypothetical_relocation
            )
            if anticipate
            else belief
        )
        interval = cfg.budgeted_update_steps or cfg.future_steps
        periods = cfg.future_steps // interval

        def future_period(carry, index):
            xf, qf, bf = carry
            if exact:
                future_mean, all_future = bf, bf.inner_fields(eps)
            else:
                # Recompute the small inducing-kernel table. Carrying the entire
                # prior cache through scan was measured to be slower on M3 CPU.
                future_cache = (
                    cache.relocate(bf) if anticipate and cfg.hypothetical_relocation else cache
                )
                future_mean, all_future = future_cache.tabulate(bf)
            future_inner = InnerFields(all_future.basis, all_future.coefficients[cfg.worlds :])
            # Global time reference; only current and later controls see this posterior.
            reference = future_reference(vector, first.x[-1], bf, env, matrices, cfg)
            reference = jax.lax.dynamic_slice_in_dim(reference, index * interval, interval)
            ns = jax.lax.dynamic_slice_in_dim(
                observation_noise, cfg.stage_steps + index * interval, interval
            )
            out = segment(xf, qf, reference, future_mean, future_inner, world, ns, env, cfg, True)
            if anticipate and periods > 1:
                bf = jax.lax.cond(
                    index < periods - 1,
                    lambda old_b: old_b.condition(
                        out.x[1:, :2], out.observations, relocate=cfg.hypothetical_relocation
                    ),
                    lambda old_b: old_b,
                    bf,
                )
            return (out.x[-1], out.q[-1], bf), out

        _, parts = jax.lax.scan(
            future_period,
            (first.x[-1], first.q[-1], posterior),
            jnp.arange(periods),
        )
        flatten = lambda a: a.reshape((-1, *a.shape[2:]))
        future = Segment(
            jnp.concatenate((first.x[-1, None], flatten(parts.x[:, 1:]))),
            jnp.concatenate((first.q[-1, None], flatten(parts.q[:, 1:]))),
            parts.cost.sum(),
            flatten(parts.observations),
            flatten(parts.controls),
            flatten(parts.correction),
            flatten(parts.slack),
            flatten(parts.clearance),
        )
        cost = first.cost + future.cost
        valid = jnp.isfinite(cost)
        if cfg.reject_collisions:
            valid &= (first.clearance.min() >= 0) & (future.clearance.min() >= 0)
        return jnp.where(valid, cost, jnp.inf), (first, future)

    costs, trajectories = jax.lax.map(one, (all_inner.coefficients[: cfg.worlds], noise))
    return costs.mean(), trajectories


def make_planner(cfg, anticipate=True, objective_cells=None):
    """Optionally omit zero-target cells from candidate scoring, exactly.

    Their nonnegative clarity contributes zero deficit, and neither dynamics,
    wind inference nor this policy depends on them. Normalize by the ORIGINAL
    grid size. Execution and independent verification still evolve every cell.
    The caller must pass precisely the positive-target indices.
    """

    @jax.jit
    def plan(x, q, belief, cache, noise, env, initials, key):
        if objective_cells is None:
            qe, ee, scale = q, env, 1.0
        else:
            idx = jnp.asarray(objective_cells)
            qe = q[idx]
            ee = env._replace(
                cells=env.cells[idx], targets=env.targets[idx], process_noise=env.process_noise[idx]
            )
            scale = len(objective_cells) / len(q)
        objective = lambda vector: (
            evaluate(vector, x, qe, belief, cache, noise, ee, cfg, anticipate)[0] * scale
        )
        costs = jax.lax.map(objective, initials)
        initial = initials[jnp.argmin(costs)]
        result = optimize(objective, initial, key, cfg)
        return result._replace(evaluations=result.evaluations + len(initials))

    return plan


def warm_starts(x, env, cfg, rapid=False):
    """Shared geometric seeds, including staying. MBD selects by task cost only."""
    candidates = []
    # Both region goals remain available regardless of current location/clarity.
    for goal_index, goal in enumerate([env.roi_a, env.roi_b, x[:2]]):
        delta = goal - x[:2]
        direction = delta / jnp.maximum(jnp.linalg.norm(delta), 0.01)
        normal = jnp.array([-direction[1], direction[0]])
        for offset in [0.0, -1.5, 1.5] if goal_index < 2 else [0.0]:
            fractions = jnp.array([0.25, 0.5, 0.8, 1.0])
            if rapid:
                fractions = (
                    jnp.array([0.7, 1.0, 1.0, 1.0])
                    if offset == 0
                    else jnp.array([0.5, 1.0, 1.0, 1.0])
                )
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
    compact = jnp.stack(candidates)
    if cfg.budgeted_policy == "compact":
        return compact
    # Seven common starts, as before; two independently adjustable route
    # continuations replace a bounded shift. No route is compulsory.
    branches = []
    separation = jnp.linalg.norm(env.obstacles[0, :2] - env.obstacles[1, :2]) / 2
    free_half_width = jnp.maximum(separation - env.obstacles[:, 2].max() - cfg.safety_margin, 0.01)
    acceleration_scale = 2 * free_half_width / (cfg.backup_steps * cfg.backup_dt) ** 2
    for gi, goal in enumerate([env.roi_a, env.roi_b]):
        direction = (goal - x[:2]) / jnp.maximum(jnp.linalg.norm(goal - x[:2]), 0.01)
        normal = jnp.array([-direction[1], direction[0]])
        for first_route, second_route in [(0, 1), (0, 2), (1, 2)]:
            first = compact[3 * gi + first_route]
            second = compact[3 * gi + second_route]
            # Geometry-based initialization only; all four gate coefficients
            # are optimized jointly with both routes by MBD. The acceleration
            # scale is not an RPCBF replacement or a safety certificate.
            gate = jnp.r_[-acceleration_scale / cfg.wind_std, -direction, 1.0]
            if first_route == 1:
                gate = jnp.r_[0.0, normal, 0.0]
            branches.append(jnp.r_[first[:4], first[4:12], second[4:12], gate])
    branches.append(jnp.r_[compact[-1, :12], compact[-1, 4:12], jnp.zeros(4)])
    return jnp.stack(branches)
