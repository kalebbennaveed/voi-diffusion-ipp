"""Matched compact-policy feature ablation; no new reward or search budget."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_mbd_reliability import independent_draws, make_optimizer
from diagnose_mbd_reliability import digest, make_evaluator, save

from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.replay_budgeted import snapshots
from voi_diffusion_ipp.planning.budgeted import warm_starts
from voi_diffusion_ipp.wind.cache import build_prior_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    assert source_hashes() == protocol["source_sha256"]
    assert all(digest(p) == h for p, h in protocol["driver_sha256"].items())
    folder = Path(protocol["output"])
    folder.mkdir(parents=True, exist_ok=False)
    build = jax.jit(lambda b, e: build_prior_cache(b, e, 129, True))
    for ji, job in enumerate(protocol["jobs"]):
        path = Path(job["input"])
        assert digest(path) == job["input_sha256"]
        old_path = Path(protocol["reference_output"]) / (job["name"] + ".json")
        old = json.loads(old_path.read_text())
        report = {"job": job, "optimization": [], "evaluation": [], "reference_sha256": digest(old_path), "complete": False}
        output = folder / (job["name"] + ".json")
        for _, base, env, b, x, q, _, meta in snapshots(path, [job["period"]]):
            cfg = replace(base, mbd_latent_bound=6.0, diffusion_reference_steps=4, worlds=2)
            cells = tuple(np.flatnonzero(np.asarray(env.targets) > 0))
            initials = warm_starts(x, env, cfg, meta["rapid_shared_initializations"])
            # Preserve the original eight-draw prefix so the probe arm exactly
            # reproduces its old two-world training objective and random numbers.
            outer, noise = independent_draws(b, cfg, protocol["training_seed"] + ji, 8)
            inner = jax.random.normal(jax.random.PRNGKey(protocol["inner_seed"] + ji), (cfg.inner_samples, *b.sample_shape))
            cache = build(b, jnp.concatenate((outer[:2], inner)))
            jax.block_until_ready(cache)
            for features in ("probe", "route_knots"):
                vc = replace(cfg, budgeted_feedback_features=features)
                for anticipate in (True, False):
                    fn = make_optimizer(vc, anticipate, 1, cells)
                    inputs = (x, q, b, cache, noise[:2], env, initials)
                    start = time.perf_counter()
                    program = fn.lower(*inputs, jax.random.PRNGKey(protocol["optimizer_seeds"][0])).compile()
                    compile_s = time.perf_counter() - start
                    for seed in protocol["optimizer_seeds"]:
                        start = time.perf_counter()
                        result, index, seed_costs = jax.block_until_ready(program(*inputs, jax.random.PRNGKey(seed)))
                        elapsed = time.perf_counter()-start
                        index = int(index)
                        vector, cost = np.asarray(result.vector[index]), float(result.cost[index])
                        if features == "probe":
                            original = next(r for r in old["optimization"] if r["variant"] == "4x8_w2" and r["anticipate"] == anticipate and r["seed"] == seed)
                            np.testing.assert_allclose(vector, original["vector"], atol=1e-6, rtol=0)
                            np.testing.assert_allclose(cost, original["train_cost"], atol=2e-7, rtol=0)
                        report["optimization"].append({"features": features, "anticipate": anticipate, "seed": seed,
                            "vector": vector, "train_cost": cost, "optimizer_s": elapsed, "compile_s": compile_s,
                            "evaluations": int(result.evaluations.sum()) + len(initials),
                            "invalid_fraction": float(np.mean(~np.isfinite(result.history.costs))), "mean_ess": float(np.mean(result.trace[..., 2]))})
                        np.savez_compressed(folder / f'{job["name"]}_{features}_{int(anticipate)}_{seed}.npz',
                            vector=vector, candidates=result.history.candidates, costs=result.history.costs,
                            weights=result.history.weights, iterates=result.iterates, trace=result.trace, seed_costs=seed_costs)
                        save(output, report)
                        print(job["name"], features, anticipate, seed, cost, elapsed, flush=True)
            functions = {}
            for bank in range(protocol["evaluation_banks"]):
                ec = replace(cfg, worlds=4, reject_collisions=False)
                outer_eval, noise_eval = independent_draws(b, ec, protocol["evaluation_seed"] + 100*ji + bank, 4)
                inner_eval = jax.random.normal(jax.random.PRNGKey(protocol["evaluation_inner_seed"] + 100*ji + bank), inner.shape)
                eps = jnp.concatenate((outer_eval, inner_eval))
                arrays = []
                for features in ("probe", "route_knots"):
                    vectors = jnp.asarray([r["vector"] for r in report["optimization"] if r["features"] == features])
                    if features not in functions:
                        functions[features] = make_evaluator(replace(ec, budgeted_feedback_features=features), cells, "exact", True)
                    arrays.append(jax.block_until_ready(functions[features](vectors, x, q, b, cache, eps, noise_eval, env)))
                data = {k: np.concatenate([np.asarray(a[k]) for a in arrays], axis=0) for k in arrays[0]}
                np.savez_compressed(folder / f'{job["name"]}_test_{bank}.npz', **data)
                report["evaluation"].append({"bank": bank, "cost": data["cost"], "clearance": data["clearance"]})
                save(output, report)
                print(job["name"], "test bank", bank, flush=True)
            report["complete"] = True
            save(output, report)
    save(folder / "status.json", {"complete": True})


if __name__ == "__main__":
    main()
