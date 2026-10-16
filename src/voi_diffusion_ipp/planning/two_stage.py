"""Causal two-stage stochastic evaluator with shared observation-conditioned recourse."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..clarity import deficit
from ..wind import WorldField, information_gain
from .rollout import continuation, rollout


class ScenarioBuffers(NamedTuple):
    world_coefficients: jax.Array
    observation_noise: jax.Array
    inner_prior_eps: jax.Array
    inner_future_eps: jax.Array
    future_noise: jax.Array


class TwoStageResult(NamedTuple):
    cost: jax.Array
    first: object
    future: object
    posterior: object
    branch: jax.Array
    info_gain: jax.Array


def make_buffers(belief, key, cfg):
    kw, ko, ki, kf = jax.random.split(key, 4)
    shape = (cfg.inner_samples, *belief.mean.shape)
    return ScenarioBuffers(
        belief.sample_field(kw, cfg.worlds),
        jax.random.normal(ko, (cfg.worlds, cfg.stage_steps, 2)),
        jax.random.normal(ki, shape),
        jax.random.normal(kf, shape),
        jax.random.normal(jax.random.fold_in(ko, 81001), (cfg.worlds, cfg.future_steps, 2)),
    )


def route(x, q, belief, env, cfg, futures=None, inner_eps=None, updates=None):
    """Only observable state and posterior enter this API. No field or scenario ID."""
    if cfg.shared_future or cfg.branches == 1:
        return jnp.array(0, jnp.int32)
    if cfg.branch_rule == "task":
        if futures is None:
            raise ValueError(
                "Task-conditioned recourse requires the candidate future policy library"
            )
        costs = recourse_costs(x, q, belief, futures, env, cfg, inner_eps, updates)
        return jnp.argmin(costs).astype(jnp.int32)
    mean, cov = belief.predict(jnp.array([[0.0, 0.0]]))
    crosswind = jnp.abs(mean[0, 1]) + 1.5 * jnp.sqrt(cov[0, 1, 1])
    # Fixed physical crosswind threshold; not trained on outer truths.
    return (crosswind > 0.45).astype(jnp.int32)


def recourse_costs(x, q, belief, futures, env, cfg, inner_eps=None, updates=None):
    """Posterior predictive action values, with no outer-world argument.

    These independent, fixed standardized draws are common random numbers across
    candidates. They are transformed by the available posterior, never the true
    outer coefficient. No future observation or hidden scenario ID is available.
    """
    coefficients = belief.sample_field(jax.random.PRNGKey(910031), cfg.recourse_samples)
    if inner_eps is None:
        inner_eps = jax.random.normal(
            jax.random.PRNGKey(910032), (cfg.inner_samples, *belief.mean.shape)
        )
    updates = cfg.future_updates if updates is None else updates
    noises = jax.random.normal(
        jax.random.PRNGKey(910033), (cfg.recourse_samples, cfg.future_steps, 2)
    )

    def policy_cost(controls):
        def world_cost(inputs):
            coeff, noise = inputs
            out = continuation(
                x,
                q,
                controls,
                belief,
                inner_eps,
                WorldField(belief.basis, coeff),
                noise,
                env,
                cfg,
                updates=updates,
            )
            cost = out.cost + cfg.terminal_seconds * deficit(out.q[-1], env.targets)
            invalid = (cfg.reject_collisions & (jnp.min(out.clearance) < 0)) | (
                cfg.require_feasible_candidates & ~jnp.all(out.feasible)
            )
            return jnp.where(invalid, jnp.inf, cost)

        return jnp.mean(jax.lax.map(world_cost, (coefficients, noises)))

    return jax.lax.map(policy_cost, futures)


def decode(vector, cfg):
    scale = (
        jnp.array([cfg.waypoint_x_limit, cfg.waypoint_y_limit])
        if cfg.control_representation == "waypoint"
        else cfg.umax
    )
    knots = scale * jnp.tanh(vector.reshape((1 + cfg.branches, cfg.knots, 2)))

    def interpolate(k, n):
        t = jnp.linspace(0, 1, n)
        tk = jnp.linspace(0, 1, cfg.knots)
        return jax.vmap(lambda v: jnp.interp(t, tk, v), in_axes=1, out_axes=1)(k)

    return interpolate(knots[0], cfg.stage_steps), jax.vmap(
        lambda k: interpolate(k, cfg.future_steps)
    )(knots[1:])


def evaluate_one(
    vector,
    x,
    q,
    belief,
    coeff,
    noise,
    inner_eps,
    future_eps,
    env,
    cfg,
    mode="proposed",
    future_noise=None,
):
    first_controls, futures = decode(vector, cfg)
    world = WorldField(belief.basis, coeff)
    inner = belief.inner_fields(inner_eps)
    first = rollout(x, q, first_controls, belief, inner, world, noise, env, cfg)
    posterior = (
        belief.condition(first.x[1:, :2], first.observations) if cfg.measurements else belief
    )
    stage_belief = posterior if mode in ("proposed", "fixed") else belief
    updates = cfg.future_updates and mode in ("proposed", "fixed")
    if future_noise is None:
        future_noise = jax.random.normal(jax.random.PRNGKey(910034), (cfg.future_steps, 2))
    branch = route(first.x[-1], first.q[-1], stage_belief, env, cfg, futures, future_eps, updates)
    future = continuation(
        first.x[-1],
        first.q[-1],
        futures[branch],
        stage_belief,
        future_eps,
        world,
        future_noise,
        env,
        cfg,
        updates=updates,
        feedback=(mode == "fixed"),
    )
    ig = information_gain(belief, posterior)
    cost = first.cost + future.cost + cfg.terminal_seconds * deficit(future.q[-1], env.targets)
    if mode == "passive":
        cost = first.cost + cfg.terminal_seconds * deficit(first.q[-1], env.targets)
    elif mode == "weighted_ig":
        cost = cost - cfg.ig_weight * ig
    elif mode == "heuristic":
        # Deliberately noncausal draft baseline, isolated and explicitly labeled.
        replay = rollout(
            x,
            q,
            first_controls,
            posterior,
            posterior.inner_fields(future_eps),
            world,
            noise,
            env,
            cfg,
        )
        cost = first.cost - jnp.maximum(first.cost - replay.cost, 0.0)
    if cfg.reject_collisions:
        feasible = jnp.min(first.clearance) >= 0
        if mode not in ("passive", "heuristic"):
            feasible &= jnp.min(future.clearance) >= 0
        cost = jnp.where(feasible, cost, jnp.inf)
    if cfg.require_feasible_candidates:
        feasible_qp = jnp.all(first.feasible)
        if mode not in ("passive", "heuristic"):
            feasible_qp &= jnp.all(future.feasible)
        cost = jnp.where(feasible_qp, cost, jnp.inf)
    return TwoStageResult(cost, first, future, posterior, branch, ig)


def objective(vector, x, q, belief, buffers, env, cfg, mode="proposed"):
    def one(args):
        coeff, noise, future_noise = args
        result = evaluate_one(
            vector,
            x,
            q,
            belief,
            coeff,
            noise,
            buffers.inner_prior_eps,
            buffers.inner_future_eps,
            env,
            cfg,
            mode,
            future_noise,
        )
        return result.cost

    # Sequential scenario map bounds memory; all candidates share these exact random buffers.
    return jnp.mean(
        jax.lax.map(
            one, (buffers.world_coefficients, buffers.observation_noise, buffers.future_noise)
        )
    )


def matched_no_information(result, vector, old_belief, coeff, future_eps, env, cfg):
    """Matched physical start and truth; branch chosen from OLD available belief.

    Same policy-vector library is held fixed. This measures the filter/router effect,
    not an independently optimized value of perfect or imperfect information.
    """
    _, futures = decode(vector, cfg)
    x = result.first.x[-1]
    q = result.first.q[-1]
    b = route(x, q, old_belief, env, cfg, futures, future_eps, updates=False)
    return rollout(
        x,
        q,
        futures[b],
        old_belief,
        old_belief.inner_fields(future_eps),
        WorldField(old_belief.basis, coeff),
        jnp.zeros((cfg.future_steps, 2)),
        env,
        cfg,
    )
