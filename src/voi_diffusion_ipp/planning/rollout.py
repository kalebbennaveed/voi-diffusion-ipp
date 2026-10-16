from typing import NamedTuple

import jax
import jax.numpy as jnp

from .. import clarity
from ..dynamics import step
from ..safety import filter_control


class Rollout(NamedTuple):
    x: jax.Array
    q: jax.Array
    nominal: jax.Array
    controls: jax.Array
    observations: jax.Array
    cost: jax.Array
    correction: jax.Array
    feasible: jax.Array
    clearance: jax.Array
    admissible: jax.Array
    slack: jax.Array


def continuation(
    x, q, controls, belief, inner_eps, world, noise, env, cfg, updates=False, feedback=False
):
    """Execute shared policy parameters, conditioning only at complete period boundaries.

    The physical world remains fixed. Inner fields are posterior transforms of
    fixed independent standard draws. Measurements enter only subsequent periods.
    """
    if not updates or not cfg.measurements:
        return rollout(
            x, q, controls, belief, belief.inner_fields(inner_eps), world, noise, env, cfg, feedback
        )
    periods = len(controls) // cfg.stage_steps
    us = controls.reshape(periods, cfg.stage_steps, 2)
    ns = noise.reshape(periods, cfg.stage_steps, 2)

    def period(carry, inputs):
        x, q, b = carry
        u, n = inputs
        out = rollout(x, q, u, b, b.inner_fields(inner_eps), world, n, env, cfg, feedback)
        updated = b.condition(out.x[1:, :2], out.observations)
        return (out.x[-1], out.q[-1], updated), out

    _, out = jax.lax.scan(period, (x, q, belief), (us, ns))
    flatten = lambda a: a.reshape((-1, *a.shape[2:]))
    return Rollout(
        jnp.concatenate((x[None], flatten(out.x[:, 1:]))),
        jnp.concatenate((q[None], flatten(out.q[:, 1:]))),
        flatten(out.nominal),
        flatten(out.controls),
        flatten(out.observations),
        out.cost.sum(),
        flatten(out.correction),
        flatten(out.feasible),
        flatten(out.clearance),
        flatten(out.admissible),
        flatten(out.slack),
    )


def clarity_controller(x, q, belief, env, cfg):
    """Fixed task-aware continuation approximation; no wind-information reward."""
    need = jnp.maximum(env.targets - q, 0)
    attractiveness = need / (0.25 + jnp.sum((env.cells - x[:2]) ** 2, axis=1))
    goal = env.cells[jnp.argmax(attractiveness)]
    mean = (belief.basis.features(x[:2]) @ belief.mean)[0]
    return jnp.clip(1.3 * (goal - x[:2]) - 1.8 * x[2:] - mean, -cfg.umax, cfg.umax)


def waypoint_controller(x, waypoint, belief, cfg):
    """Shared state-feedback policy; only the available wind mean compensates drift."""
    mean = belief.predict(x[:2])[0][0]
    return jnp.clip(1.2 * (waypoint - x[:2]) - 1.8 * x[2:] - mean, -cfg.umax, cfg.umax)


def rollout(x0, q0, controls, belief, inner, world, noise, env, cfg, feedback=False):
    """Belief and inner fields remain fixed for this entire horizon."""
    wind = lambda p: world.at(p)[0]

    def body(carry, inputs):
        x, q = carry
        u, eps = inputs
        if feedback:
            u = clarity_controller(x, q, belief, env, cfg)
        elif cfg.control_representation == "waypoint":
            u = waypoint_controller(x, u, belief, cfg)
        safe, info = filter_control(x, u, belief, inner, env, cfg)
        xn = step(x, safe, wind, cfg.dt)
        qn = clarity.step(
            q, x[:2], xn[:2], env.cells, cfg.sensor_sigma, cfg.sensor_R, cfg.clarity_decay, cfg.dt
        )
        cost = 0.5 * cfg.dt * (clarity.deficit(q, env.targets) + clarity.deficit(qn, env.targets))
        obs = world.at(xn[:2])[0] + belief.noise_std * eps
        # Sample RK4 trajectory at intermediate times for empirical collision diagnostics.
        mids = jax.vmap(lambda t: step(x, safe, wind, t))(
            cfg.dt * jnp.array([0.25, 0.5, 0.75, 1.0])
        )
        clearance = jnp.min(jax.vmap(env.clearance)(mids))
        return (xn, qn), (
            xn,
            qn,
            u,
            safe,
            obs,
            cost,
            info.correction,
            info.feasible,
            clearance,
            info.admissible_fraction,
            info.slack,
        )

    _, out = jax.lax.scan(body, (x0, q0), (controls, noise))
    x, q, u, safe, obs, cost, corr, feasible, clearance, area, slack = out
    return Rollout(
        jnp.concatenate((x0[None], x)),
        jnp.concatenate((q0[None], q)),
        u,
        safe,
        obs,
        jnp.sum(cost),
        corr,
        feasible,
        clearance,
        area,
        slack,
    )
