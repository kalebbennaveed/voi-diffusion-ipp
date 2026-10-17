"""Offline horizon/route-family sensitivity on earned reversed-wind states.

Uses actual POAM conditioning and RPCBF evaluations. This is a common-candidate
diagnostic, not an executed mission or replacement for MBD. The timing check at
30 s additionally runs actual MBD and its uncached selected-policy verification.
"""

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_wind_route_choices import round_trip_seeds

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.experiments.wind_cases import physical_wind
from voi_diffusion_ipp.planning.budgeted import evaluate, make_planner, warm_starts
from voi_diffusion_ipp.viz.demo_data import serializable
from voi_diffusion_ipp.wind.cache import build_prior_cache
from voi_diffusion_ipp.wind.oracle import KnownBasis, KnownCache, KnownWind
from voi_diffusion_ipp.wind.poam import POAM

ROOT = Path(__file__).resolve().parents[1]


def main():
    protocol = json.loads((ROOT / "docs/wind_route_study/horizon_protocol_01.json").read_text())
    path = ROOT / protocol["input"]
    output = ROOT / "docs/wind_route_study/horizon_diagnosis_01"
    output.mkdir(exist_ok=False)
    raw = json.loads(path.with_suffix(".json").read_text())
    meta = raw["metadata"]
    assert source_hashes() == meta["source_sha256"]
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    original = Config(**meta["config"])
    env, template, _, _, _ = make_two_region(meta["task_spec"], original)
    model = POAM.from_data(
        data["initial_observation_positions"],
        data["initial_observations"],
        jax.random.PRNGKey(600000),
        len(template.z),
        noise_std=original.observation_std,
        prior_std=original.wind_std,
        spectral_count=original.poam_spectral_nodes,
        prior_support=template.prior_support,
    )
    snapshot = jax.jit(lambda b: b.predict(env.cells))
    build = jax.jit(lambda b, e: build_prior_cache(b, e, 129, True))
    known = KnownWind(KnownBasis(physical_wind(env, meta["physical_wind_spec"])))
    stride = round(original.stage_steps * original.dt / meta["execution_dt"])
    root = jax.random.PRNGKey(meta["seed"])
    result = {
        "label": "Offline development sensitivity, not an executed or globally optimal comparison",
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": protocol,
        "rows": [],
        "timing": [],
    }
    scorers = {}
    for seconds, interval_seconds in protocol["horizons_and_update_intervals_s"]:
        cfg = replace(
            original,
            budgeted_policy="compact",
            worlds=protocol["worlds"],
            future_steps=round((seconds - 2) / original.dt),
            budgeted_update_steps=round(interval_seconds / original.dt),
        )

        def factory(config, anticipate):
            @jax.jit
            def score(vectors, x, q, b, cache, noise):
                def one(y):
                    cost, (first, _future) = evaluate(
                        y, x, q, b, cache, noise, env, config, anticipate
                    )
                    return cost, first.x

                return jax.lax.map(one, vectors)

            return score

        scorers[seconds] = (
            cfg,
            factory(cfg, True),
            factory(cfg, False),
            factory(replace(cfg, worlds=1), False),
        )

    def verifier(config):
        return jax.jit(
            lambda y, x, q, b, e, n: evaluate(y, x, q, b, None, n, env, config, True, True, e)
        )

    last_update_s = 0.0
    for i in range(max(protocol["periods"]) + 1):
        if i in protocol["periods"]:
            b = model.planning_belief()
            mu, cov = snapshot(b)
            np.testing.assert_allclose(mu, data["mean"][i], atol=3e-5, rtol=3e-5)
            np.testing.assert_allclose(jnp.sqrt(cov[:, 0, 0]), data["std"][i], atol=3e-5, rtol=3e-5)
            x, q = jnp.asarray(data["x"][i * stride]), jnp.asarray(data["q"][i * stride])
            for seconds, (cfg, active, frozen, oracle) in scorers.items():
                starts = warm_starts(x, env, cfg, True)
                extra, names = round_trip_seeds(x, env, cfg)
                vectors = jnp.concatenate((starts, extra))
                labels = [f"single_{k}" for k in range(len(starts))] + names
                sa, sf = [], []
                for bank in range(protocol["banks"]):
                    k1, k2 = jax.random.split(jax.random.PRNGKey(971000 + 100 * i + bank))
                    eps = jax.random.normal(k1, (cfg.worlds + cfg.inner_samples, *b.sample_shape))
                    # Use a common prefix across horizons; each comparison is at one fixed horizon.
                    noise = jax.random.normal(k2, (cfg.worlds, 90, 2))[
                        :, : cfg.stage_steps + cfg.future_steps
                    ]
                    cache = build(b, eps)
                    a, fa = active(vectors, x, q, b, cache, noise)
                    f, ff = frozen(vectors, x, q, b, cache, noise)
                    np.testing.assert_allclose(fa, ff, atol=2e-4, rtol=2e-4)
                    sa.append(np.array(a))
                    sf.append(np.array(f))
                    print(
                        "period",
                        i,
                        "horizon",
                        seconds,
                        "bank",
                        bank,
                        "best",
                        int(a.argmin()),
                        int(f.argmin()),
                        flush=True,
                    )
                sa, sf = np.stack(sa), np.stack(sf)
                oc, _ = oracle(
                    vectors,
                    x,
                    q,
                    known,
                    KnownCache(jnp.ones((1 + cfg.inner_samples, 1, 2))),
                    jnp.zeros((1, cfg.stage_steps + cfg.future_steps, 2)),
                )
                np.savez_compressed(
                    output / f"period_{i:03d}_horizon_{seconds:02d}.npz",
                    vectors=np.array(vectors),
                    active=sa,
                    frozen=sf,
                    oracle=np.array(oc),
                )
                row = {
                    "period": i,
                    "horizon_s": seconds,
                    "update_interval_s": cfg.budgeted_update_steps * cfg.dt,
                    "labels": labels,
                    "best_active": labels[int(sa.mean(0).argmin())],
                    "best_frozen": labels[int(sf.mean(0).argmin())],
                    "best_known_wind": labels[int(oc.argmin())],
                    "best_mean_active_cost": float(sa.mean(0).min()),
                    "best_mean_frozen_cost": float(sf.mean(0).min()),
                    "all_banks_finite_active": int(np.isfinite(sa).all(0).sum()),
                    "cross_bank": [],
                }
                for held in range(protocol["banks"]):
                    train = [k for k in range(protocol["banks"]) if k != held]
                    ai, fi = int(sa[train].mean(0).argmin()), int(sf[train].mean(0).argmin())
                    row["cross_bank"].append(
                        {
                            "held": held,
                            "active_choice": labels[ai],
                            "frozen_choice": labels[fi],
                            "active_cost": float(sa[held, ai]),
                            "frozen_choice_active_cost": float(sa[held, fi]),
                        }
                    )
                result["rows"].append(row)
                if i == 15:
                    fast = replace(cfg, worlds=original.worlds)
                    plan = make_planner(fast, True, np.flatnonzero(np.array(env.targets) > 0))
                    verify = verifier(fast)
                    timings = []
                    for repeat in range(4):
                        begin = time.perf_counter()
                        keys = jax.random.split(jax.random.PRNGKey(998000 + repeat), 3)
                        ep = jax.random.normal(
                            keys[0], (fast.worlds + fast.inner_samples, *b.sample_shape)
                        )
                        ns = jax.random.normal(
                            keys[1], (fast.worlds, fast.stage_steps + fast.future_steps, 2)
                        )
                        initial = jnp.concatenate(
                            (warm_starts(x, env, fast, True), round_trip_seeds(x, env, fast)[0])
                        )
                        ca = build(b, ep)
                        selected = plan(x, q, b, ca, ns, env, initial, keys[2])
                        checked = jax.block_until_ready(verify(selected.vector, x, q, b, ep, ns))
                        timings.append(
                            {
                                "seconds": time.perf_counter() - begin + last_update_s,
                                "cached_cost": float(selected.cost),
                                "uncached_cost": float(checked[0]),
                            }
                        )
                    result["timing"].append(
                        {
                            "horizon_s": seconds,
                            "startup": timings[0],
                            "warm_repeats": timings[1:],
                            "scope": "Single saved state; includes seed generation, cache, actual MBD, uncached verification and preceding measured POAM update. No executed mission.",
                        }
                    )
                (output / "summary.json").write_text(
                    json.dumps(serializable(result), indent=2) + "\n"
                )
        sl = slice(i * original.stage_steps, (i + 1) * original.stage_steps)
        begin = time.perf_counter()
        jax.block_until_ready(
            model.update(
                data["observation_positions"][sl],
                data["observations"][sl],
                jax.random.fold_in(root, 500000 + i),
                train_steps=original.poam_train_steps,
            )
        )
        last_update_s = time.perf_counter() - begin


if __name__ == "__main__":
    main()
