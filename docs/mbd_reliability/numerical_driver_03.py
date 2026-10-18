"""Declared saved-state numerical audit; never modifies a mission or its data."""

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
from voi_diffusion_ipp.planning.budgeted import evaluate
from voi_diffusion_ipp.planning.sampling import couple_outer_samples
from voi_diffusion_ipp.viz.demo_data import serializable
from voi_diffusion_ipp.wind.cache import build_prior_cache
from voi_diffusion_ipp.wind.gp import InnerFields


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(serializable(data), indent=2) + "\n")
    temporary.replace(path)


def recorded_draws(b, cfg, meta, period):
    keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(meta["seed"]), 100 + period), 4)
    eps = jax.random.normal(keys[0], (cfg.worlds + cfg.inner_samples, *b.sample_shape))
    noise = jax.random.normal(keys[1], (cfg.worlds, cfg.stage_steps + cfg.future_steps, 2))
    return couple_outer_samples(eps, noise, cfg.worlds, meta.get("outer_sampling", "iid"))


def shortlist(row, count=4):
    vectors = [np.asarray(row["vector"], np.float32)]
    candidates = np.asarray(row["history"]["candidates"], np.float32).reshape(-1, len(vectors[0]))
    # JSON archives encode non-finite scores as null; retain them as invalid.
    costs = np.asarray(row["history"]["costs"], dtype=float).ravel()
    for index in np.argsort(costs, kind="stable"):
        if np.isfinite(costs[index]) and not any(np.array_equal(candidates[index], v) for v in vectors):
            vectors.append(candidates[index])
        if len(vectors) == count:
            break
    return jnp.asarray(np.stack(vectors))


def make_evaluator(cfg, cells, mode, anticipate):
    """Separate physical interpolation from filter/controller interpolation."""
    @jax.jit
    def run(vectors, x, q, belief, cache, eps, noise, env):
        idx = jnp.asarray(cells)
        ee = env._replace(cells=env.cells[idx], targets=env.targets[idx], process_noise=env.process_noise[idx])
        physical = None
        if mode == "exact_physics":
            fields = belief.inner_fields(eps)
            physical = InnerFields(fields.basis, fields.coefficients[:cfg.worlds])

        def one(vector):
            _, (first, future) = evaluate(
                vector, x, q[idx], belief, cache, noise, ee, cfg, anticipate,
                mode == "exact", eps, physical_fields=physical,
            )
            return {
                "cost": (first.cost + future.cost) * len(idx) / len(q),
                "path": jnp.concatenate((first.x, future.x[:, 1:]), axis=1),
                "clearance": jnp.minimum(first.clearance.min(axis=1), future.clearance.min(axis=1)),
                "correction": jnp.concatenate((first.correction, future.correction), axis=1),
                "slack": jnp.concatenate((first.slack, future.slack), axis=1),
            }
        return jax.lax.map(one, vectors)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    assert protocol["source_sha256"] == source_hashes(), "Frozen numerical source changed"
    assert protocol["driver_sha256"] == digest(__file__), "Frozen audit driver changed"
    out = Path(protocol["output"])
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else {"rows": [], "complete": False}
    compiled = {}
    builders = {n: jax.jit(lambda b, e, n=n: build_prior_cache(b, e, n, True)) for n in (129, 257)}
    for job in protocol["jobs"]:
        path = Path(job["input"])
        assert digest(path) == job["input_sha256"] and digest(path.with_suffix(".json")) == job["json_sha256"]
        needed = [p for p in job["periods"] if not any(r["job"] == job["name"] and r["period"] == p for r in results["rows"])]
        if not needed:
            continue
        for period, cfg, env, b, x, q, row, meta in snapshots(path, needed):
            # All raw finite/post-contact costs are retained, independently of validity.
            cfg = replace(cfg, reject_collisions=False)
            eps, noise = recorded_draws(b, cfg, meta, period)
            vectors = shortlist(row)
            caches, cache_s = {}, {}
            for width, fn in builders.items():
                fn(b, eps).basis.prior_fields.block_until_ready()
                begin = time.perf_counter()
                caches[width] = jax.block_until_ready(fn(b, eps))
                cache_s[width] = time.perf_counter() - begin
            arrays, timings, compilation = {}, {}, {}
            for anticipate in (True, False):
                for label, mode, width in (("cached129", "cached", 129), ("exact_physics129", "exact_physics", 129), ("cached257", "cached", 257), ("exact", "exact", 129)):
                    name = ("learn_" if anticipate else "frozen_") + label
                    key = (repr(cfg), tuple(np.flatnonzero(np.asarray(env.targets) > 0)), len(vectors), mode, width, anticipate)
                    inputs = (vectors, x, q, b, caches[width], eps, noise, env)
                    if key not in compiled:
                        fn = make_evaluator(cfg, key[1], mode, anticipate)
                        begin = time.perf_counter()
                        compiled[key] = fn.lower(*inputs).compile()
                        compilation[name] = time.perf_counter() - begin
                    begin = time.perf_counter()
                    arrays[name] = jax.block_until_ready(compiled[key](*inputs))
                    timings[name] = time.perf_counter() - begin
                    print(job["name"], period, name, np.asarray(arrays[name]["cost"]).mean(axis=1).tolist(), flush=True)
            np.testing.assert_allclose(arrays["learn_exact"]["path"][:, :, :cfg.stage_steps+1], arrays["frozen_exact"]["path"][:, :, :cfg.stage_steps+1], atol=2e-5, rtol=2e-5)
            expected = np.asarray(row["selected_model_path"])
            # Verify archives using their original full-grid, single-vector
            # compilation shape. Report batched/reduced-grid drift separately;
            # changing XLA shapes can amplify roundoff in a switched filter.
            original_cfg = replace(cfg, reject_collisions=meta["config"]["reject_collisions"])
            original = jax.jit(lambda v, xx, qq, bb, ee, nn: evaluate(
                v, xx, qq, bb, None, nn, env, original_cfg, True, True, ee
            ))
            _, (first, future) = original(vectors[0], x, q, b, eps, noise)
            original_path = np.concatenate((first.x[0], future.x[0, 1:]))
            np.testing.assert_allclose(original_path, expected, atol=5e-5, rtol=5e-5)
            replay_error = float(np.max(np.abs(np.asarray(arrays["learn_exact"]["path"][0, 0])-original_path)))
            comparisons = {}
            for name, a in arrays.items():
                prefix = name.split("_")[0]
                ref = arrays[prefix + "_exact"]
                distance = np.linalg.norm(np.asarray(a["path"])[..., :2] - np.asarray(ref["path"])[..., :2], axis=-1)
                cost, refcost = np.asarray(a["cost"]).mean(axis=1), np.asarray(ref["cost"]).mean(axis=1)
                valid = (np.asarray(a["clearance"]) >= 0).all(axis=1) & np.isfinite(cost)
                refvalid = (np.asarray(ref["clearance"]) >= 0).all(axis=1) & np.isfinite(refcost)
                choose = int(np.argmin(np.where(valid, cost, np.inf))) if valid.any() else None
                refchoose = int(np.argmin(np.where(refvalid, refcost, np.inf))) if refvalid.any() else None
                comparisons[name] = {
                    "cost": cost, "clearance": a["clearance"], "valid": valid,
                    "max_distance_m": distance.max(axis=(1, 2)),
                    "first_stage_max_distance_m": distance[..., :cfg.stage_steps+1].max(axis=(1, 2)),
                    "cost_error": cost - refcost,
                    "correction_max_error": np.abs(np.asarray(a["correction"])-np.asarray(ref["correction"])).max(axis=(1, 2)),
                    "selected_index": choose, "exact_selected_index": refchoose,
                    "selection_regret": None if choose is None or refchoose is None or not refvalid[choose] else float(refcost[choose]-refcost[refchoose]),
                }
            np.savez_compressed(out / f'{job["name"]}_{period}.npz', vectors=vectors,
                                **{f"{name}_{field}": np.asarray(value) for name, data in arrays.items() for field, value in data.items()})
            results["rows"].append({"job": job["name"], "period": period, "events": job["events"], "stress_selected": job.get("stress_selected", False), "comparisons": comparisons, "batch_evaluation_s": timings, "compile_s": compilation, "cache_build_s": cache_s, "batched_vs_original_max_state_error": replay_error})
            save(result_path, results)
    results["complete"] = True
    save(result_path, results)


if __name__ == "__main__":
    main()
