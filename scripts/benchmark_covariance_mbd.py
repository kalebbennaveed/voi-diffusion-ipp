"""Fixed-state MBD budget, equal-cost restarts, and independent wind-bank checks."""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.replay_budgeted import snapshots
from voi_diffusion_ipp.planning import budgeted, covariance
from voi_diffusion_ipp.planning.mbd import optimize, target_objective
from voi_diffusion_ipp.viz.demo_data import serializable
from voi_diffusion_ipp.wind.cache import build_prior_cache

VARIANTS = {
    "4x8": (4, 8, 1),
    "8x16": (8, 16, 1),
    "16x32": (16, 32, 1),
    "one_chain_4x7": (4, 7, 1),
    "three_chains_4x2": (4, 2, 3),
}


def make_optimizer(cfg, mode, chains, objective_cells):
    @jax.jit
    def run(x, q, b, cache, eps, noise, obs_eps, env, initials, key):
        idx = jnp.asarray(objective_cells)
        ee = env._replace(
            cells=env.cells[idx], targets=env.targets[idx], process_noise=env.process_noise[idx]
        )
        scale = len(idx) / len(q)
        if mode == "realization":
            raw = lambda v: (
                budgeted.evaluate(v, x, q[idx], b, cache, noise, ee, cfg, True)[0] * scale
            )
        else:
            raw = lambda v: (
                covariance.evaluate(v, x, q[idx], b, cache, eps, obs_eps, ee, cfg, True)[0] * scale
            )
        objective = target_objective(raw, cfg)
        seed_costs = jax.lax.map(objective, initials)
        if chains == 1:
            indices = jnp.array([jnp.argmin(seed_costs)])
        else:
            # Shared geometric families, both A/B directions; staying belongs to
            # the central family. This seeds search, never forces execution.
            families = (jnp.array([0, 3, 6]), jnp.array([1, 4]), jnp.array([2, 5]))
            indices = jnp.stack([g[jnp.argmin(seed_costs[g])] for g in families])

        def one(item):
            index, k = item
            return optimize(raw, initials[index], k, cfg)

        keys = jax.random.split(key, chains) if chains > 1 else key[None]
        results = jax.lax.map(one, (indices, keys))
        best = jnp.argmin(results.cost)
        return results, best, seed_costs, indices

    return run


def make_auditor(cfg, objective_cells):
    # Evaluates earned feedback with full hypothetical POAM, including means.
    # Physical truth is an independent posterior draw, not the environment truth.
    @jax.jit
    def run(vectors, x, q, b, cache, noise, env):
        idx = jnp.asarray(objective_cells)
        ee = env._replace(
            cells=env.cells[idx], targets=env.targets[idx], process_noise=env.process_noise[idx]
        )

        def one(v):
            _, (first, future) = budgeted.evaluate(v, x, q[idx], b, cache, noise, ee, cfg, True)
            per_world = (first.cost + future.cost) * len(idx) / len(q)
            clear = jnp.minimum(first.clearance.min(axis=1), future.clearance.min(axis=1))
            return per_world, clear

        return jax.lax.map(one, vectors)

    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--periods", nargs="+", type=int, default=[0, 5, 18])
    p.add_argument("--seeds", nargs="+", type=int, default=[973101, 973102])
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["realization", "covariance"],
        default=["realization", "covariance"],
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "source_sha256": source_hashes(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input": str(args.path),
        "periods": args.periods,
        "seeds": args.seeds,
        "variants": args.variants,
        "modes": args.modes,
        "latent_bound": 6.0,
        "fixed_noise_endpoints": "original four-level schedule",
        "train_worlds": 2,
        "selection_worlds": 4,
        "evaluation_worlds": 8,
        "independent_inner_banks": False,
        "inner_policy": "fixed across outer-bank changes to isolate outer sampling",
        "shortlist": "training winner plus two distinct lowest-cost evaluated proposals",
        "multi_chain_evaluations": 37,
        "matched_single_evaluations": 37,
        "status": "running",
        "claim": "Development diagnostic; test bank is not a method-selection validation set",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    compiled = {}
    build = jax.jit(lambda b, e: build_prior_cache(b, e, 129, True))
    auditors = {}
    rows = []
    for period, original, env, b, x, q, row, meta in snapshots(args.path, args.periods):
        cfg = replace(original, mbd_latent_bound=6.0, diffusion_reference_steps=4)
        idx = np.flatnonzero(np.asarray(env.targets) > 0)
        interval = cfg.budgeted_update_steps or cfg.future_steps
        capacity = cfg.stage_steps + cfg.future_steps - interval
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(973000 + period), 3)
        eps = jax.random.normal(k1, (cfg.worlds + cfg.inner_samples, *b.sample_shape))
        noise = jax.random.normal(k2, (cfg.worlds, cfg.stage_steps + cfg.future_steps, 2))
        obs_eps = jax.random.normal(k3, (cfg.inner_samples, capacity, 2))
        train_cache = build(b, eps)
        inner_eps = eps[cfg.worlds :]
        inner_cache = build(b, inner_eps)
        initials = budgeted.warm_starts(x, env, cfg, meta["rapid_shared_initializations"])
        banks = []
        for bi, worlds in enumerate((4, 8)):
            bk1, bk2 = jax.random.split(jax.random.PRNGKey(974000 + 100 * period + bi))
            draws = jnp.concatenate((jax.random.normal(bk1, (worlds, *b.sample_shape)), inner_eps))
            ns = jax.random.normal(bk2, (worlds, cfg.stage_steps + cfg.future_steps, 2))
            bc = build(b, draws)
            if worlds not in auditors:
                auditors[worlds] = make_auditor(
                    replace(cfg, worlds=worlds, reject_collisions=False), idx
                )
            banks.append((auditors[worlds], bc, ns))
        for mode in args.modes:
            for variant in args.variants:
                levels, proposals, chains = VARIANTS[variant]
                vc = replace(cfg, diffusion_steps=levels, proposals=proposals)
                cache = train_cache if mode == "realization" else inner_cache
                peps = eps if mode == "realization" else inner_eps
                inputs = (x, q, b, cache, peps, noise, obs_eps, env, initials)
                key = (mode, variant)
                compile_s = 0.0
                if key not in compiled:
                    fn = make_optimizer(vc, mode, chains, idx)
                    begin = time.perf_counter()
                    compiled[key] = fn.lower(*inputs, jax.random.PRNGKey(args.seeds[0])).compile()
                    compile_s = time.perf_counter() - begin
                for seed in args.seeds:
                    begin = time.perf_counter()
                    result, best, seed_costs, indices = jax.block_until_ready(
                        compiled[key](*inputs, jax.random.PRNGKey(seed))
                    )
                    seconds = time.perf_counter() - begin
                    best = int(best)
                    candidates = np.asarray(result.history.candidates).reshape(-1, 18)
                    costs = np.asarray(result.history.costs).reshape(-1)
                    shortlist = [np.asarray(result.vector[best])]
                    for ci in np.argsort(costs):
                        if np.isfinite(costs[ci]) and not any(
                            np.array_equal(candidates[ci], v) for v in shortlist
                        ):
                            shortlist.append(candidates[ci])
                        if len(shortlist) == 3:
                            break
                    while len(shortlist) < 3:
                        shortlist.append(shortlist[0])
                    selected_audit = []
                    for fn, bc, ns in banks:
                        perworld, clear = fn(jnp.array(shortlist), x, q, b, bc, ns, env)
                        selected_audit.append(
                            {"costs": np.asarray(perworld), "clearance": np.asarray(clear)}
                        )
                    validation = selected_audit[0]
                    acceptable = np.all(validation["clearance"] >= 0, axis=1)
                    valcost = np.where(acceptable, validation["costs"].mean(axis=1), np.inf)
                    selected = int(np.argmin(valcost)) if acceptable.any() else 0
                    item = {
                        "period": period,
                        "mode": mode,
                        "variant": variant,
                        "optimizer_seed": seed,
                        "train_cost": float(result.cost[best]),
                        "seed_costs": seed_costs,
                        "seed_indices": indices,
                        "optimizer_s": seconds,
                        "compile_s": compile_s,
                        "objective_evaluations": int(jnp.sum(result.evaluations)) + len(initials),
                        "shortlist": shortlist,
                        "selection_bank": selected_audit[0],
                        "evaluation_bank": selected_audit[1],
                        "selection_index": selected,
                        "selection_has_feasible_candidate": bool(acceptable.any()),
                    }
                    rows.append(item)
                    np.savez_compressed(
                        args.output / f"trace_{period}_{mode}_{variant}_{seed}.npz",
                        candidates=result.history.candidates,
                        costs=result.history.costs,
                        weights=result.history.weights,
                        trace=result.trace,
                        vectors=result.vector,
                    )
                    (args.output / "results.json").write_text(
                        json.dumps(serializable({"rows": rows}), indent=2)
                    )
                    print(
                        period,
                        mode,
                        variant,
                        seed,
                        "train",
                        item["train_cost"],
                        "optimizer_s",
                        round(seconds, 3),
                        "test_before/after",
                        *[
                            round(float(selected_audit[1]["costs"][k].mean()), 5)
                            for k in (0, selected)
                        ],
                        flush=True,
                    )
                    compile_s = 0.0
    protocol["status"] = "complete"
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
