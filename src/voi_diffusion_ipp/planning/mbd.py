"""Training-free Model-Based Diffusion (Pan et al., Eqs. 6, 8, 9).

Gaussian proposals live in unconstrained normalized coordinates. The simulator
maps coordinates through tanh to actuator limits; no clipped proposal is falsely
claimed to be Gaussian. Warm starts are forward-noised before all reverse levels.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class OptimizationResult(NamedTuple):
    vector: jax.Array
    cost: jax.Array
    trace: jax.Array  # incumbent, population mean, effective sample size, invalid fraction
    evaluations: jax.Array
    iterates: jax.Array


def schedule(steps, beta_min, beta_max):
    alpha = 1 - jnp.linspace(beta_min, beta_max, steps)
    return alpha, jnp.cumprod(alpha)


def weights(cost, temperature):
    valid = jnp.isfinite(cost)
    minimum = jnp.min(jnp.where(valid, cost, jnp.inf))
    minimum = jnp.where(jnp.any(valid), minimum, 0.0)
    w = jnp.where(valid, jnp.exp(-(cost - minimum) / temperature), 0.0)
    # All invalid has zero weights; callers explicitly keep their old iterate.
    return w / jnp.maximum(jnp.sum(w), 1e-30)


def score(y_i, candidates, costs, alphabar, temperature):
    w = weights(costs, temperature)
    mean = jnp.einsum("n,nd->d", w, candidates)
    mean = jnp.where(jnp.any(jnp.isfinite(costs)), mean, y_i / jnp.sqrt(alphabar))
    return (-y_i + jnp.sqrt(alphabar) * mean) / (1 - alphabar), w


def reverse_update(y_i, estimated_score, alpha, alphabar):
    return (y_i + (1 - alphabar) * estimated_score) / jnp.sqrt(alpha)


def optimize(objective, initial, key, cfg, algorithm="mbd"):
    if algorithm not in ("mbd", "cem", "mppi"):
        raise ValueError(algorithm)
    alpha, ab = schedule(cfg.diffusion_steps, cfg.beta_min, cfg.beta_max)
    key, kinit = jax.random.split(key)
    y = jnp.sqrt(ab[-1]) * initial + jnp.sqrt(1 - ab[-1]) * jax.random.normal(kinit, initial.shape)
    if algorithm != "mbd":
        y = initial
    initial_cost = objective(initial)

    def evaluate(xs):
        if cfg.candidate_chunk == 1:
            return jax.lax.map(objective, xs)
        return jax.lax.map(objective, xs, batch_size=cfg.candidate_chunk)

    def iteration(carry, i):
        y, scale, best, best_cost, key = carry
        key, k = jax.random.split(key)
        eps = jax.random.normal(k, (cfg.proposals, initial.size))
        if algorithm == "mbd":
            candidates = y / jnp.sqrt(ab[i]) + jnp.sqrt((1 - ab[i]) / ab[i]) * eps
        else:
            candidates = y + scale * eps
        costs = evaluate(candidates)
        ii = jnp.argmin(jnp.where(jnp.isfinite(costs), costs, jnp.inf))
        better = costs[ii] < best_cost
        best = jnp.where(better, candidates[ii], best)
        best_cost = jnp.where(better, costs[ii], best_cost)
        if algorithm == "mbd":
            s, w = score(y, candidates, costs, ab[i], cfg.temperature)
            y = reverse_update(y, s, alpha[i], ab[i])
        elif algorithm == "cem":
            elite_n = max(2, cfg.proposals // 4)
            idx = jnp.argsort(jnp.where(jnp.isfinite(costs), costs, jnp.inf))[:elite_n]
            elite = candidates[idx]
            ok = jnp.any(jnp.isfinite(costs))
            y = jnp.where(ok, jnp.mean(elite, axis=0), y)
            scale = jnp.where(ok, jnp.maximum(jnp.std(elite, axis=0), 0.06), scale)
            w = jnp.zeros(cfg.proposals).at[idx].set(1 / elite_n)
        else:
            w = weights(costs, cfg.temperature)
            y = jnp.where(jnp.any(jnp.isfinite(costs)), w @ candidates, y)
        finite = jnp.isfinite(costs)
        avg = jnp.sum(jnp.where(finite, costs, 0)) / jnp.maximum(jnp.sum(finite), 1)
        trace = jnp.array(
            [best_cost, avg, 1 / jnp.maximum(jnp.sum(w * w), 1e-30), 1 - jnp.mean(finite)]
        )
        normalized = (
            y / jnp.sqrt(jnp.where(i > 0, ab[jnp.maximum(i - 1, 0)], 1.0))
            if algorithm == "mbd"
            else y
        )
        return (y, scale, best, best_cost, key), (trace, normalized)

    (y, _, best, cost, _), (trace, iterates) = jax.lax.scan(
        iteration,
        (y, jnp.ones_like(initial) * 0.8, initial, initial_cost, key),
        jnp.arange(cfg.diffusion_steps - 1, -1, -1),
    )
    final_cost = objective(y)
    better = final_cost < cost
    return OptimizationResult(
        jnp.where(better, y, best),
        jnp.where(better, final_cost, cost),
        trace,
        jnp.array(cfg.diffusion_steps * cfg.proposals + 2),
        iterates,
    )
