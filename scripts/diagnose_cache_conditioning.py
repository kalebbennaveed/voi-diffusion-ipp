"""Test exact prior values at relocated POAM inducing points; offline only."""

import argparse
import json
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_mbd_reliability import digest, make_evaluator, recorded_draws, save, shortlist

from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.replay_budgeted import snapshots
from voi_diffusion_ipp.wind.cache import build_prior_cache


class ExactInducingCache(NamedTuple):
    base: object
    prior_coefficients: jax.Array

    def tabulate(self, belief):
        return self.base.tabulate(belief)

    def relocate(self, belief, new_positions=None):
        cache = self.base.relocate(belief, new_positions)
        # Kernel parameters/support remain fixed during hypothetical planning.
        # Inducing coefficients are zero: this evaluates the unchanged prior
        # field, not any outer physical truth or a posterior observation.
        values = jax.vmap(lambda c: belief.basis.evaluate(belief.basis.centers, c))(
            self.prior_coefficients
        )
        return self._replace(base=cache._replace(prior_at_inducing=values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("protocol", type=Path)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    assert source_hashes() == protocol["source_sha256"]
    assert digest(__file__) == protocol["driver_sha256"]
    out = Path(protocol["output"])
    out.mkdir(parents=True, exist_ok=True)
    results = {"rows": [], "complete": False}
    build = jax.jit(lambda b, e: build_prior_cache(b, e, 129, True))
    for job in protocol["jobs"]:
        path = Path(job["input"])
        assert digest(path) == job["input_sha256"]
        for period, cfg, env, b, x, q, row, meta in snapshots(path, job["periods"]):
            eps, noise = recorded_draws(b, cfg, meta, period)
            cache = build(b, eps)
            prior = b.sample_coefficients(eps).at[:, -len(b.mean):].set(0)
            corrected = ExactInducingCache(cache, prior)
            vectors = shortlist(row)
            cells = tuple(np.flatnonzero(np.asarray(env.targets) > 0))
            fn = make_evaluator(cfg, cells, "cached", True)
            full = make_evaluator(cfg, cells, "exact", True)
            arrays, timings = {}, {}
            for name, evaluator, table in (("original", fn, cache), ("exact_inducing", fn, corrected), ("uncached", full, cache)):
                inputs = (vectors, x, q, b, table, eps, noise, env)
                program = evaluator.lower(*inputs).compile()
                start = time.perf_counter()
                arrays[name] = jax.block_until_ready(program(*inputs))
                timings[name] = time.perf_counter()-start
            # A field-only test at one common relocated posterior separates
            # interpolation from amplification in the inverse-kernel correction.
            points = arrays["uncached"]["path"][0, 0, 1:cfg.stage_steps+1, :2]
            post = b.condition(points, b.mean_at(points), relocate=True)
            query = env.cells
            old = cache.relocate(post).condition(post).at(query)
            new_cache = corrected.relocate(post)
            new = new_cache.base.condition(post).at(query)
            exact = post.inner_fields(eps).at(query)
            original_values = jax.vmap(lambda c, post=post: post.basis.evaluate(post.basis.centers, c))(prior)
            np.testing.assert_allclose(new_cache.base.prior_at_inducing, original_values, atol=1e-7)
            metrics = {}
            for name, a in arrays.items():
                distance = np.linalg.norm(np.asarray(a["path"])[..., :2]-np.asarray(arrays["uncached"]["path"])[..., :2], axis=-1)
                metrics[name] = {"cost": np.asarray(a["cost"]).mean(axis=1), "clearance": a["clearance"], "max_path_error_m": distance.max(axis=(1, 2)), "batch_s": timings[name]}
            results["rows"].append({"job": job["name"], "period": period, "metrics": metrics,
                "same_posterior_field_max_error_original": float(jnp.max(jnp.abs(old-exact))),
                "same_posterior_field_max_error_exact_inducing": float(jnp.max(jnp.abs(new-exact)))})
            np.savez_compressed(out/f'{job["name"]}_{period}.npz', **{f'{n}_{k}':v for n,a in arrays.items() for k,v in a.items()})
            save(out/'results.json', results)
            print(job['name'],period,metrics,flush=True)
    results['complete'] = True
    save(out/'results.json', results)


if __name__ == "__main__":
    main()
