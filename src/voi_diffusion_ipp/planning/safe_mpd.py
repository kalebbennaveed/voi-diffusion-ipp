"""Safe-MPD shielded-control score update, adapted to belief-derived wind.

Independent implementation of Kim et al., Algorithm 3 (arXiv:2512.06261).
This full-control experiment freezes the belief within each planning horizon.
Online POAM still learns between actual replans. It is NOT an anticipation arm.
The deterministic posterior-mean rollout maps a common control sequence to its
shielded controls; those controls, not the raw proposal, enter the score update.
Costs are evaluated across independent outer wind scenarios with the sampled
belief-based shield. No outer scenario contributes privileged control labels.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..wind import InnerFields, WorldField
from .budgeted import future_reference, reference_matrices, segment, split_vector
from .mbd import OptimizationResult, reverse_update, schedule, score


class MeanField(NamedTuple):
    belief: object

    def at(self, p):
        return self.belief.mean_at(p)


class ShieldedHistory(NamedTuple):
    candidates: jax.Array
    costs: jax.Array
    weights: jax.Array
    noisy_before: jax.Array
    noisy_after: jax.Array
    score: jax.Array
    incumbent: jax.Array
    alpha: jax.Array
    alphabar: jax.Array
    effective_candidates: jax.Array


def optimize(project_and_score, initial, key, cfg):
    """Each evaluation returns cost AND the transformed vector actually scored."""
    alpha, ab = schedule(cfg.diffusion_steps, cfg.beta_min, cfg.beta_max)
    key, ki = jax.random.split(key)
    y = jnp.sqrt(ab[-1]) * initial + jnp.sqrt(1 - ab[-1]) * jax.random.normal(ki, initial.shape)
    cost, effective = project_and_score(initial)

    def body(carry, i):
        y, best, best_cost, key = carry
        key, k = jax.random.split(key)
        candidates = y / jnp.sqrt(ab[i]) + jnp.sqrt((1 - ab[i]) / ab[i]) * jax.random.normal(
            k, (cfg.proposals, initial.size)
        )
        costs, effective = jax.lax.map(
            project_and_score,
            candidates,
            batch_size=None if cfg.candidate_chunk == 1 else cfg.candidate_chunk,
        )
        winner = jnp.argmin(jnp.where(jnp.isfinite(costs), costs, jnp.inf))
        improved = costs[winner] < best_cost
        best = jnp.where(improved, effective[winner], best)
        best_cost = jnp.where(improved, costs[winner], best_cost)
        s, w = score(y, effective, costs, ab[i], cfg.temperature)
        yn = reverse_update(y, s, alpha[i], ab[i])
        finite = jnp.isfinite(costs)
        trace = jnp.array(
            [
                best_cost,
                jnp.where(finite, costs, 0).sum() / jnp.maximum(finite.sum(), 1),
                1 / jnp.maximum((w * w).sum(), 1e-30),
                1 - finite.mean(),
            ]
        )
        history = ShieldedHistory(candidates, costs, w, y, yn, s, best, alpha[i], ab[i], effective)
        return (yn, best, best_cost, key), (trace, yn, history)

    (y, best, cost, _), (trace, iterates, history) = jax.lax.scan(
        body, (y, effective, cost, key), jnp.arange(cfg.diffusion_steps - 1, -1, -1)
    )
    final_cost, final_vector = project_and_score(y)
    improve = final_cost < cost
    return OptimizationResult(
        jnp.where(improve, final_vector, best),
        jnp.where(improve, final_cost, cost),
        trace,
        jnp.array(cfg.diffusion_steps * cfg.proposals + 2),
        iterates,
        history,
        initial,
    )


def roll_controls(y, x, q, belief, inner, world, noise, env, cfg):
    controls = cfg.umax * jnp.clip(y.reshape(-1, 2), -1, 1)
    n = cfg.stage_steps
    first = segment(
        x, q, controls[:n], belief, inner, world, noise[:n], env, cfg, False, actuator_controls=True
    )
    future = segment(
        first.x[-1],
        first.q[-1],
        controls[n:],
        belief,
        inner,
        world,
        noise[n:],
        env,
        cfg,
        False,
        shield_latched=first.shield_active[-1],
        actuator_controls=True,
    )
    return first, future


def evaluate(y, x, q, belief, cache, noise, env, cfg, exact=False, eps=None):
    mean, fields = (belief, belief.inner_fields(eps)) if exact else cache.tabulate(belief)
    inner = InnerFields(fields.basis, fields.coefficients[cfg.worlds :])

    def one(inputs):
        coeff, ns = inputs
        first, future = roll_controls(
            y, x, q, mean, inner, WorldField(fields.basis, coeff), ns, env, cfg
        )
        cost = first.cost + future.cost
        valid = jnp.isfinite(cost)
        if cfg.reject_collisions:
            valid &= (first.clearance.min() >= 0) & (future.clearance.min() >= 0)
        return jnp.where(valid, cost, jnp.inf), (first, future)

    costs, paths = jax.lax.map(one, (fields.coefficients[: cfg.worlds], noise))
    return costs.mean(), paths


def make_planner(cfg, objective_cells=None, transformed=True):
    """Matched full-control MBD comparator uses transformed=False (same seeds)."""

    @jax.jit
    def plan(x, q, belief, cache, noise, env, initials, key):
        scale = 1.0
        if objective_cells is not None:
            idx = jnp.asarray(objective_cells)
            scale = len(objective_cells) / len(q)
            q = q[idx]
            env = env._replace(
                cells=env.cells[idx], targets=env.targets[idx], process_noise=env.process_noise[idx]
            )
        mean, fields = cache.tabulate(belief)
        inner = InnerFields(fields.basis, fields.coefficients[cfg.worlds :])
        world = MeanField(mean)
        ns = jnp.zeros((cfg.stage_steps + cfg.future_steps, 2))

        def seed_controls(vector):
            controls = split_vector(vector, cfg)[0]
            first = segment(
                x,
                q,
                controls,
                mean,
                inner,
                world,
                ns[: cfg.stage_steps],
                env,
                cfg,
                False,
                apply_safety=False,
            )
            matrices = reference_matrices(cfg.future_steps * cfg.dt, cfg.future_steps)
            refs = future_reference(vector, first.x[-1], belief, env, matrices, cfg)
            future = segment(
                first.x[-1],
                first.q[-1],
                refs,
                mean,
                inner,
                world,
                ns[cfg.stage_steps :],
                env,
                cfg,
                True,
                apply_safety=False,
            )
            return jnp.concatenate((first.controls, future.controls)).ravel() / cfg.umax

        def objective(vector):
            effective = jnp.clip(vector, -1, 1)
            if transformed:
                first, future = roll_controls(effective, x, q, mean, inner, world, ns, env, cfg)
                effective = jnp.concatenate((first.controls, future.controls)).ravel() / cfg.umax
            cost = evaluate(effective, x, q, belief, cache, noise, env, cfg)[0] * scale
            return cost, effective

        seeds = jax.lax.map(seed_controls, initials)
        costs, _effective = jax.lax.map(objective, seeds)
        # Keep the nominal seed until optimize applies its own final projection.
        initial = seeds[jnp.argmin(costs)]
        result = optimize(objective, initial, key, cfg)
        return result._replace(evaluations=result.evaluations + len(initials))

    return plan
