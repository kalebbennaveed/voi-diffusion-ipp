"""Separate MBD search budget, outer sampling, and restart effects at saved states."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_mbd_reliability import digest, make_evaluator, save

from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.replay_budgeted import snapshots
from voi_diffusion_ipp.planning import budgeted
from voi_diffusion_ipp.planning.mbd import optimize, target_objective
from voi_diffusion_ipp.wind.cache import build_prior_cache


def make_optimizer(cfg, anticipate, chains, cells):
    @jax.jit
    def run(x, q, belief, cache, noise, env, initials, key):
        idx = jnp.asarray(cells)
        ee = env._replace(cells=env.cells[idx], targets=env.targets[idx], process_noise=env.process_noise[idx])
        raw = lambda v: budgeted.evaluate(v, x, q[idx], belief, cache, noise, ee, cfg, anticipate)[0] * len(idx) / len(q)
        objective = target_objective(raw, cfg)
        seed_costs = jax.lax.map(objective, initials)
        if chains == 1:
            indices = jnp.array([jnp.argmin(seed_costs)])
            keys = key[None]
        else:
            groups = (jnp.array([0, 3, 6]), jnp.array([1, 4]), jnp.array([2, 5]))
            indices = jnp.stack([g[jnp.argmin(seed_costs[g])] for g in groups])
            keys = jax.random.split(key, chains)
        results = jax.lax.map(lambda item: optimize(raw, initials[item[0]], item[1], cfg), (indices, keys))
        return results, jnp.argmin(results.cost), seed_costs
    return run


def independent_draws(belief, cfg, seed, worlds):
    a, b = jax.random.split(jax.random.PRNGKey(seed))
    return (
        jax.random.normal(a, (worlds, *belief.sample_shape)),
        jax.random.normal(b, (worlds, cfg.stage_steps + cfg.future_steps, 2)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    assert source_hashes() == protocol["source_sha256"]
    assert digest(__file__) == protocol["driver_sha256"]
    assert digest("scripts/diagnose_mbd_reliability.py") == protocol["evaluator_sha256"]
    out = Path(protocol["output"])
    out.mkdir(parents=True, exist_ok=True)
    compiled, evaluators = {}, {}
    build = jax.jit(lambda b, e: build_prior_cache(b, e, 129, True))
    for job_index, job in enumerate(protocol["jobs"]):
        path = Path(job["input"])
        assert digest(path) == job["input_sha256"] and digest(path.with_suffix(".json")) == job["json_sha256"]
        result_path = out / (job["name"] + ".json")
        report = json.loads(result_path.read_text()) if result_path.exists() else {"job": job, "optimization": [], "evaluation": [], "complete": False}
        if report["complete"]:
            continue
        for period, base, env, belief, x, q, row, meta in snapshots(path, [job["period"]]):
            cfg = replace(base, mbd_latent_bound=6.0, diffusion_reference_steps=4)
            cells = tuple(np.flatnonzero(np.asarray(env.targets) > 0))
            initials = budgeted.warm_starts(x, env, cfg, meta["rapid_shared_initializations"])
            outer, noise = independent_draws(belief, cfg, protocol["training_seed"] + job_index, 8)
            inner = jax.random.normal(jax.random.PRNGKey(protocol["inner_seed"] + job_index), (cfg.inner_samples, *belief.sample_shape))
            banks = {n: build(belief, jnp.concatenate((outer[:n], inner))) for n in (2, 4, 8)}
            jax.block_until_ready(banks)
            for variant in protocol["variants"]:
                label, levels, proposals, chains, worlds = variant
                vc = replace(cfg, diffusion_steps=levels, proposals=proposals, worlds=worlds)
                for anticipate in (True, False):
                    key = (repr(vc), cells, anticipate, chains)
                    for seed in protocol["optimizer_seeds"]:
                        if any(r["variant"] == label and r["anticipate"] == anticipate and r["seed"] == seed for r in report["optimization"]):
                            continue
                        inputs = (x, q, belief, banks[worlds], noise[:worlds], env, initials, jax.random.PRNGKey(seed))
                        compile_s = 0.0
                        if key not in compiled:
                            fn = make_optimizer(vc, anticipate, chains, cells)
                            start = time.perf_counter()
                            compiled[key] = fn.lower(*inputs).compile()
                            compile_s = time.perf_counter()-start
                        start = time.perf_counter()
                        results, index, seed_costs = jax.block_until_ready(compiled[key](*inputs))
                        elapsed = time.perf_counter()-start
                        index = int(index)
                        vector = np.asarray(results.vector[index])
                        name = f'{job["name"]}_{label}_{int(anticipate)}_{seed}'
                        np.savez_compressed(out/(name+'.npz'), vector=vector, costs=results.history.costs,
                            candidates=results.history.candidates, weights=results.history.weights, trace=results.trace,
                            seed_costs=seed_costs, iterates=results.iterates)
                        record = {"variant": label, "anticipate": anticipate, "seed": seed, "vector": vector,
                            "train_cost": float(results.cost[index]), "seed_costs": seed_costs,
                            "optimizer_s": elapsed, "compile_s": compile_s,
                            "objective_evaluations": int(results.evaluations.sum()) + len(initials),
                            "outer_rollout_evaluations": (int(results.evaluations.sum()) + len(initials))*worlds,
                            "invalid_fraction": float(np.mean(~np.isfinite(results.history.costs))),
                            "mean_ess": float(np.mean(results.trace[..., 2]))}
                        report["optimization"].append(record)
                        save(result_path, report)
                        print(job["name"], label, anticipate, seed, 'train', record['train_cost'], 's', round(elapsed, 3), flush=True)
            vectors = jnp.asarray(np.array([r["vector"] for r in report["optimization"]]))
            ec = replace(cfg, worlds=4, reject_collisions=False)
            # Selected policies execute with causal earned updates in every arm;
            # frozen refers to planning, not disabling learning during execution.
            key = (repr(ec), cells, len(vectors))
            if key not in evaluators:
                evaluators[key] = make_evaluator(ec, cells, "exact", True)
            for bank in range(protocol["evaluation_banks"]):
                if any(r["bank"] == bank for r in report["evaluation"]):
                    continue
                draws, ns = independent_draws(belief, ec, protocol["evaluation_seed"] + 100*job_index + bank, 4)
                new_inner = jax.random.normal(jax.random.PRNGKey(protocol["evaluation_inner_seed"] + 100*job_index + bank), inner.shape)
                metrics = {}
                for inner_label, ieps in (("fixed_inner", inner), ("fresh_inner", new_inner)):
                    eps = jnp.concatenate((draws, ieps))
                    inputs = (vectors, x, q, belief, banks[2], eps, ns, env)
                    # Cache argument is unused by the original uncached evaluator.
                    start = time.perf_counter()
                    data = jax.block_until_ready(evaluators[key](*inputs))
                    np.savez_compressed(out/f'{job["name"]}_test_{bank}_{inner_label}.npz', **data)
                    metrics[inner_label] = {"cost": data["cost"], "clearance": data["clearance"],
                        "evaluation_wall_s_including_first_compile": time.perf_counter()-start}
                report["evaluation"].append({"bank": bank, "metrics": metrics})
                save(result_path, report)
                print(job["name"], 'independent evaluation', bank, flush=True)
            report["complete"] = True
            save(result_path, report)
    save(out/'status.json', {"complete": True, "jobs": [j['name'] for j in protocol['jobs']]})


if __name__ == "__main__":
    main()
