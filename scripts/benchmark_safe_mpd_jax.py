"""Same earned initial state, synchronous JAX timing and candidate batching check.

No simulator truth is accessed. An archived survey reconstructs full POAM.
Separate lowering/compilation from blocked warm calls. Thread one worker only.
"""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.planning import safe_mpd
from voi_diffusion_ipp.planning.budgeted import make_planner, warm_starts
from voi_diffusion_ipp.wind.cache import build_prior_cache
from voi_diffusion_ipp.wind.poam import POAM

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path = ROOT / "results/budgeted_missions/safe_mpd_pilot_01_compact_rpcbf/proposed.npz"
    d = np.load(path, allow_pickle=False)
    meta = json.loads(path.with_suffix(".json").read_text())["metadata"]
    cfg = Config(**meta["config"])
    env, template, _, _, _ = make_two_region(meta["task_spec"], cfg)
    poam = POAM.from_data(
        d["initial_observation_positions"],
        d["initial_observations"],
        jax.random.PRNGKey(600000),
        len(template.z),
        noise_std=cfg.observation_std,
        prior_std=cfg.wind_std,
        spectral_count=cfg.poam_spectral_nodes,
        prior_support=template.prior_support,
    )
    belief = poam.planning_belief()
    mu, cov = belief.predict(env.cells)
    np.testing.assert_allclose(mu, d["mean"][0], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(jnp.sqrt(cov[:, 0, 0]), d["std"][0], atol=3e-5, rtol=3e-5)
    keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(meta["seed"]), 100), 4)
    eps = jax.random.normal(keys[0], (cfg.worlds + cfg.inner_samples, *belief.sample_shape))
    noise = jax.random.normal(keys[1], (cfg.worlds, cfg.stage_steps + cfg.future_steps, 2))
    cache = jax.block_until_ready(
        jax.jit(lambda b, e: build_prior_cache(b, e, 129, True))(belief, eps)
    )
    x, q = jnp.asarray(d["x"][0]), jnp.asarray(d["q"][0])
    initials = warm_starts(x, env, cfg, True)
    idx = np.flatnonzero(np.asarray(env.targets) > 0)
    report = {
        "source_sha256": source_hashes(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(x) for x in jax.devices()],
        "note": "Same initial state. Timings here are optimizer only, not full cycle. Three warm calls. All outputs synchronized.",
        "rows": [],
    }
    baseline = {}
    for form, safety, substeps in [
        ("control_mpd", "rpcbf", 4),
        ("safe_mpd", "sampled_mpd_shield", 1),
        ("compact", "sampled_mpd_shield", 1),
    ]:
        for chunk in [1, 2, 8]:
            cc = replace(cfg, candidate_chunk=chunk, safety_filter=safety, shield_substeps=substeps)
            plan = (
                make_planner(cc, True, idx)
                if form == "compact"
                else safe_mpd.make_planner(cc, idx, transformed=form == "safe_mpd")
            )
            inputs = (x, q, belief, cache, noise, env, initials, keys[2])
            t = time.perf_counter()
            compiled = plan.lower(*inputs).compile()
            compile_s = time.perf_counter() - t
            # Warm once before measurement; the lowered executable is reused.
            jax.block_until_ready(compiled(*inputs))
            times = []
            for _ in range(3):
                t = time.perf_counter()
                result = jax.block_until_ready(compiled(*inputs))
                times.append(time.perf_counter() - t)
            if chunk == 1:
                baseline[form] = np.asarray(result.vector)
            row = {
                "form": form,
                "chunk": chunk,
                "substeps": substeps,
                "compile_s": compile_s,
                "warm_s": times,
                "cost": float(result.cost),
                "max_vector_difference_from_chunk1": float(
                    np.max(np.abs(result.vector - baseline[form]))
                ),
            }
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
