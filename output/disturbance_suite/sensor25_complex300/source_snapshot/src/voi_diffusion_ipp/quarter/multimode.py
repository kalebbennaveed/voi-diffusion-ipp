"""Independent MBD reverse chains with one shared rollout budget.

Each supplied full-control seed owns a reverse chain and an archive. Proposal
weights are normalized within that chain, so incompatible routes never enter
the same weighted average. ``cfg.proposals`` is the *total* number of proposals
at each reverse level, distributed round-robin over the seeds. Evaluation is a
single flattened batch, allowing JAX to parallelize trajectories across modes.

The seeds only initialize the diffusion state. Every proposal retains all
decision coordinates, the Gaussian proposal is not projected, and the score,
reverse update, cosine schedule, favorable-tail objective and archive density
correction are the equations used in :mod:`mbd_core`.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .diffusion import cosine_schedule, importance_log_ratio, lower_tail
from .mbd_core import guided_log_weights, proposal_parameters, score_and_update


def mode_score_and_update(y, samples, log_weights, proposal_modes, alpha, cumulative_alpha):
    """Apply ordinary MBD independently to each mode's proposal subset.

The returned flattened weights sum to one *per mode*, rather than across
incompatible routes. A one-mode call is exactly ``mbd_core.score_and_update``.
    """
    assigned = proposal_modes[None, :] == jnp.arange(y.shape[0])[:, None]
    local_log_weights = jnp.where(assigned, log_weights[None, :], -jnp.inf)
    # Recenter before the shared equation helper so its finite sentinel cannot
    # defeat extremely negative finite log likelihoods in a partially invalid
    # batch. This leaves the normalized MBD weights unchanged.
    finite = jnp.isfinite(local_log_weights)
    maximum = jnp.max(jnp.where(finite, local_log_weights, -jnp.inf), axis=1, keepdims=True)
    local_log_weights = jnp.where(finite, local_log_weights-maximum, -jnp.inf)
    # Invalid trajectories carry zero weight; avoid IEEE ``0 * NaN`` leaking
    # from another mode into a valid mode's weighted sum.
    finite_samples = jnp.where(jnp.isfinite(samples), samples, 0.)
    updated, scores, weights, valid = jax.vmap(
        score_and_update, in_axes=(0, None, 0, None, None)
    )(y, finite_samples, local_log_weights, alpha, cumulative_alpha)
    return updated, scores, jnp.sum(weights, axis=0), valid


def unique_rows_mask(parameters):
    """Keep the first exact occurrence; no approximate route merging."""
    same = jnp.all(parameters[:, None, :] == parameters[None, :, :], axis=-1)
    earlier = jnp.arange(len(parameters))[None, :] < jnp.arange(len(parameters))[:, None]
    return ~jnp.any(same & earlier, axis=1)


class SearchResult(NamedTuple):
    # The ordinary search's public audit fields, with flattened proposal axes.
    finalists: jax.Array
    finalist_indices: jax.Array
    final: jax.Array
    initial_cost: jax.Array
    trace: jax.Array
    samples: jax.Array
    sample_costs: jax.Array
    weights: jax.Array
    before: jax.Array
    after: jax.Array
    scores: jax.Array
    field_costs: jax.Array
    log_importance_ratios: jax.Array
    archive_centers: jax.Array
    archive_probabilities: jax.Array
    evaluations: jax.Array
    guidance_log_weights: jax.Array
    initial_parameters: jax.Array
    initial_field_costs: jax.Array
    # before/after/scores and the archives retain an explicit mode dimension.
    proposal_modes: jax.Array
    mode_initial_indices: jax.Array
    mode_leader_indices: jax.Array
    mode_finals: jax.Array
    mode_trace: jax.Array
    finalist_unique: jax.Array
    evaluation_counts: jax.Array


def search(evaluate, seeds, previous, key, cfg, eta, rho, cold, log_guidance=None):
    """Optimize every seed mode, returning at most ``2*modes+1`` finalists.

    Modes are forward-noised from their individual seeds on every call. Thus
    this function is intended for the explicit mode initialization; ``cold``
    is accepted for the ordinary-search API, and the caller supplies shifted
    seeds when continuing MPC. ``previous`` is evaluated once and remains a
    final option, without pulling all mode centers toward a common incumbent.

    Finalists contain each mode's best evaluated member, each evaluated final
    reverse iterate, and the previous control sequence. ``finalist_unique``
    permits skipping exact duplicate evaluations with ``lax.cond``. The caller
    must evaluate every retained finalist on the same fresh validation fields.
    ``evaluation_counts`` records initial/proposal/terminal training counts;
    ``evaluations`` is their sum, independent of the objective or mixture rho.
    """
    del cold
    modes, dim = seeds.shape
    n, k = cfg.iterations, cfg.proposals
    if modes < 1 or k < modes:
        raise ValueError('Multimode search needs at least one proposal per seed mode')
    if previous.shape != (dim,):
        raise ValueError('Every mode and previous sequence must have the same full dimension')
    if min(n, cfg.batch_size, cfg.archive_size) < 1:
        raise ValueError('Iterations, batch size and archive size must be positive')
    initial = jnp.concatenate((seeds, previous[None]))
    initial_count = modes + 1
    size = initial_count + n*k + modes
    proposal_modes = jnp.arange(k) % modes
    # Previous is an independent final option; archive entries stay mode-local.
    owners = jnp.concatenate((jnp.arange(modes), jnp.array([-1]),
                              jnp.tile(proposal_modes, n), jnp.arange(modes)))
    membership = owners[None, :] == jnp.arange(modes)[:, None]
    alpha, abar = cosine_schedule(n, cfg.cosine_end)

    def batch(parameters):
        values = jax.lax.map(evaluate, parameters, batch_size=cfg.batch_size)
        valid = (jnp.all(jnp.isfinite(values), axis=1)
                 & jnp.all(jnp.isfinite(parameters), axis=1)
                 & jnp.all(jnp.abs(parameters) <= cfg.latent_bound, axis=1))
        return (jnp.where(valid, jnp.mean(values, axis=1), jnp.inf),
                jnp.where(valid, lower_tail(values, eta), jnp.inf), values)

    means0, opts0, initial_fields = batch(initial)
    parameters = jnp.zeros((size, dim), dtype=seeds.dtype).at[:initial_count].set(initial)
    means = jnp.full(size, jnp.inf).at[:initial_count].set(means0)
    opts = jnp.full(size, jnp.inf).at[:initial_count].set(opts0)
    key, noise_key = jax.random.split(key)
    # Independent Gaussian forward noise preserves the individual initial modes.
    y = (jnp.sqrt(abar[-1])*seeds
         + jnp.sqrt(1-abar[-1])*jax.random.normal(noise_key, (modes, dim)))
    archive_count = min(cfg.archive_size, size)

    def leaders(costs):
        finite = membership & jnp.isfinite(costs)[None, :]
        best = jnp.argmin(jnp.where(membership, costs[None, :], jnp.inf), axis=1)
        return jnp.where(jnp.any(finite, axis=1), best, jnp.arange(modes))

    def iteration(carry, j):
        y, parameters, means, opts, key = carry
        i = n-1-j
        centers, sigma = proposal_parameters(y, abar[i])
        incumbent = leaders(means)
        priorities = jnp.where(
            membership & jnp.isfinite(opts)[None, :] & ((means-opts) > 1e-6)[None, :],
            jnp.maximum(opts[incumbent, None]-opts[None, :], 0.), 0.)
        priority, indices = jax.lax.top_k(priorities, archive_count)
        total = jnp.sum(priority, axis=1)
        nu = priority/jnp.maximum(total[:, None], 1e-30)
        fraction = jnp.where(total > 0, rho, 0.)
        archive = parameters[indices]
        key, component_key, proposal_key = jax.random.split(key, 3)
        probabilities = jnp.concatenate(((1-fraction)[:, None], fraction[:, None]*nu), axis=1)
        logits = jnp.where(probabilities > 0, jnp.log(jnp.maximum(probabilities, 1e-30)), -jnp.inf)
        component = jax.random.categorical(component_key, logits[proposal_modes])
        mixture_centers = jnp.concatenate((centers[:, None, :], archive), axis=1)
        samples = (mixture_centers[proposal_modes, component]
                   + sigma*jax.random.normal(proposal_key, (k, dim)))
        mm, oo, field_costs = batch(samples)
        ratio = jax.vmap(importance_log_ratio, in_axes=(0, 0, 0, 0, None, 0))(
            samples[:, None, :], centers[proposal_modes], archive[proposal_modes],
            nu[proposal_modes], sigma, fraction[proposal_modes])[:, 0]
        demonstration = jnp.full_like(oo, -jnp.inf)
        if log_guidance is not None:
            demonstration = jax.vmap(log_guidance)(samples, seeds[proposal_modes], opts0[proposal_modes])
            demonstration = jnp.where(jnp.isfinite(oo), demonstration, -jnp.inf)
        logw = guided_log_weights(-oo/cfg.temperature, demonstration, ratio)
        yn, score, weights, ok = mode_score_and_update(
            y, samples, logw, proposal_modes, alpha[i], abar[i])
        start = initial_count+j*k
        parameters = jax.lax.dynamic_update_slice(parameters, samples, (start, 0))
        means = jax.lax.dynamic_update_slice(means, mm, (start,))
        opts = jax.lax.dynamic_update_slice(opts, oo, (start,))
        best = leaders(opts)
        # An empty score restarts at an evaluated member of that same mode.
        # Only the no-finite-incumbent recovery uses a bounded sanitized seed.
        fallback = jnp.where(jnp.isfinite(opts[best, None]), parameters[best],
                             jnp.clip(jnp.nan_to_num(seeds), -cfg.latent_bound, cfg.latent_bound))
        previous_abar = jnp.where(i > 0, abar[jnp.maximum(i-1, 0)], 1.)
        yn = jnp.where(ok[:, None], yn, jnp.sqrt(previous_abar)*fallback)
        assigned = proposal_modes[None, :] == jnp.arange(modes)[:, None]
        squared = jnp.sum(jnp.where(assigned, weights[None, :]**2, 0.), axis=1)
        ess = jnp.where(ok, 1/jnp.maximum(squared, 1e-30), 0.)
        finite_fraction = jnp.sum(assigned & jnp.isfinite(logw)[None, :], axis=1)/jnp.sum(assigned, axis=1)
        mode_trace = jnp.stack((opts[best], means[leaders(means)], ess, fraction,
                                finite_fraction, jnp.full(modes, sigma), (~ok).astype(y.dtype)), axis=1)
        trace = jnp.array([jnp.min(opts), jnp.min(means), jnp.sum(ess),
                           jnp.mean(fraction), jnp.mean(jnp.isfinite(logw)), sigma,
                           jnp.sum(~ok)])
        logs = (trace, samples, oo, weights, y, yn, score, field_costs, ratio,
                archive, nu, demonstration, mode_trace)
        return (yn, parameters, means, opts, key), logs

    (finals, parameters, means, opts, key), logs = jax.lax.scan(
        iteration, (y, parameters, means, opts, key), jnp.arange(n))
    fm, fo, _ = batch(finals)
    parameters = parameters.at[-modes:].set(finals)
    means = means.at[-modes:].set(fm)
    opts = opts.at[-modes:].set(fo)
    mode_leaders = leaders(opts)
    terminal_indices = jnp.arange(size-modes, size)
    # Even an entirely undefined objective must return a finite recovered
    # decision, rather than resurrecting a nonfinite unevaluable seed.
    mode_leaders = jnp.where(jnp.any(membership & jnp.isfinite(opts)[None, :], axis=1),
                             mode_leaders, terminal_indices)
    indices = jnp.concatenate((mode_leaders, terminal_indices, jnp.array([modes])))
    finalists = parameters[indices]
    counts = jnp.array([initial_count, n*k, modes])
    final = jnp.where(jnp.any(jnp.isfinite(opts)), parameters[jnp.argmin(opts)], finals[0])
    return SearchResult(
        finalists, indices, final, jnp.min(opts0),
        *logs[:11], jnp.sum(counts), logs[11], initial, initial_fields,
        proposal_modes, jnp.arange(modes), mode_leaders, finals, logs[12],
        unique_rows_mask(finalists), counts)
