"""Mean-dynamics, covariance-anticipated MBD + full sampled spline RPCBF.

No outer physical worlds or hypothetical measurement values enter this planning
objective. The same monitoring cost, causal update boundaries, nominal policy
coordinates and real POAM/execution filter are used as the original method.
"""

import jax
import jax.numpy as jnp

from ..wind.covariance_forecast import CovarianceForecast, MeanWorld
from .budgeted import Segment, future_reference, reference_matrices, segment, split_vector
from .mbd import optimize, target_objective


def evaluate(
    vector,
    x,
    q,
    belief,
    cache,
    inner_eps,
    observation_eps,
    env,
    cfg,
    anticipate=True,
    exact=False,
    world=None,
    adapt_reference=True,
):
    """Optional world is for offline physical stress tests, never online planning.

    Output has one leading scenario dimension for existing mission recorders.
    Full-field covariance updates occur at boundaries only. Mean stays fixed.
    """
    interval = cfg.budgeted_update_steps or cfg.future_steps
    periods = cfg.future_steps // interval
    capacity = cfg.stage_steps + cfg.future_steps - interval
    forecast = CovarianceForecast.empty(belief, capacity)
    if exact:
        original_mean, original_inner = belief, belief.inner_fields(inner_eps)
    else:
        original_mean, original_inner = cache.tabulate(belief)
    physics = MeanWorld(original_mean) if world is None else world
    first_controls, _, _ = split_vector(vector, cfg)
    first = segment(
        x,
        q,
        first_controls,
        original_mean,
        original_inner,
        physics,
        jnp.zeros((cfg.stage_steps, 2)),
        env,
        cfg,
        False,
    )
    if anticipate:
        forecast = forecast.condition_locations(first.x[1:, :2])
    matrices = reference_matrices(cfg.future_steps * cfg.dt, cfg.future_steps)

    def future_period(carry, index):
        xf, qf, bf = carry
        if anticipate:
            if exact:
                mean, inner = belief, bf.inner_fields(original_inner, observation_eps)
            else:
                mean, inner = bf.tabulate(original_mean, original_inner, observation_eps)
        else:
            mean, inner = original_mean, original_inner
        reference = future_reference(
            vector, first.x[-1], bf if adapt_reference else belief, env, matrices, cfg
        )
        reference = jax.lax.dynamic_slice_in_dim(reference, index * interval, interval)
        out = segment(
            xf, qf, reference, mean, inner, physics, jnp.zeros((interval, 2)), env, cfg, True
        )
        if anticipate and periods > 1:
            bf = jax.lax.cond(
                index < periods - 1, lambda b: b.condition_locations(out.x[1:, :2]), lambda b: b, bf
            )
        return (out.x[-1], out.q[-1], bf), out

    _, parts = jax.lax.scan(
        future_period, (first.x[-1], first.q[-1], forecast), jnp.arange(periods)
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
        flatten(parts.shield_active),
    )
    cost = first.cost + future.cost
    valid = jnp.isfinite(cost)
    if cfg.reject_collisions:
        valid &= (first.clearance.min() >= 0) & (future.clearance.min() >= 0)
    return jnp.where(valid, cost, jnp.inf), jax.tree.map(lambda v: v[None], (first, future))


def make_planner(cfg, anticipate=True, objective_cells=None):
    if cfg.safety_filter != "rpcbf":
        raise ValueError("Covariance experiment retains sampled spline RPCBF")

    @jax.jit
    def plan(x, q, belief, cache, inner_eps, observation_eps, env, initials, key):
        if objective_cells is None:
            qe, ee, scale = q, env, 1.0
        else:
            idx = jnp.asarray(objective_cells)
            qe = q[idx]
            ee = env._replace(
                cells=env.cells[idx], targets=env.targets[idx], process_noise=env.process_noise[idx]
            )
            scale = len(objective_cells) / len(q)
        objective = lambda v: (
            evaluate(v, x, qe, belief, cache, inner_eps, observation_eps, ee, cfg, anticipate)[0]
            * scale
        )
        costs = jax.lax.map(target_objective(objective, cfg), initials)
        result = optimize(objective, initials[jnp.argmin(costs)], key, cfg)
        return result._replace(evaluations=result.evaluations + len(initials))

    return plan
