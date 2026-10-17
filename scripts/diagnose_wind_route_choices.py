"""Same-state causal/optimizer diagnosis, including geometric round-trip seeds.

No mission is changed. The additional seeds use geometry only and are offered to
both objectives. Independent banks separate objective differences from sampling
error. The explicitly privileged oracle is an offline information benchmark.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.experiments.wind_cases import physical_wind
from voi_diffusion_ipp.planning.budgeted import evaluate, warm_starts
from voi_diffusion_ipp.viz.demo_data import serializable
from voi_diffusion_ipp.wind.cache import build_prior_cache
from voi_diffusion_ipp.wind.oracle import KnownBasis, KnownCache, KnownWind
from voi_diffusion_ipp.wind.poam import POAM

ROOT = Path(__file__).resolve().parents[1]


def round_trip_seeds(x, env, cfg):
    """Additional search initializations, not mandatory routes or alternation."""
    common = warm_starts(x, env, cfg, True)
    vectors = []
    labels = []
    pairs = [(0.0, 0.0), (0.0, -1.5), (0.0, 1.5), (-1.5, 0.0), (1.5, 0.0), (-1.5, 1.5), (1.5, -1.5)]
    scale = 0.5 * (env.bounds[1] - env.bounds[0]) - 0.12
    mid = 0.5 * (env.bounds[0] + env.bounds[1])
    for gi, (goal, other) in enumerate([(env.roi_a, env.roi_b), (env.roi_b, env.roi_a)]):
        forward = (goal - x[:2]) / jnp.maximum(jnp.linalg.norm(goal - x[:2]), 0.01)
        backward = (other - goal) / jnp.linalg.norm(other - goal)
        n1 = jnp.array([-forward[1], forward[0]])
        n2 = jnp.array([-backward[1], backward[0]])
        for outbound, returning in pairs:
            positions = jnp.stack(
                [
                    0.5 * (x[:2] + goal) + outbound * n1,
                    goal,
                    0.5 * (goal + other) + returning * n2,
                    other,
                ]
            )
            latent = jnp.arctanh(jnp.clip((positions - mid) / scale, -0.97, 0.97)).ravel()
            index = 3 * gi + (0 if outbound == 0 else 1 if outbound < 0 else 2)
            vectors.append(common[index].at[4:12].set(latent))
            labels.append(f"roundtrip_goal_{gi}_out_{outbound:g}_back_{returning:g}")
    return jnp.stack(vectors), labels


def diagnose(path, periods, output, worlds=4, banks=4):
    with np.load(path, allow_pickle=False) as source:
        data = {k: source[k] for k in source.files}
    raw = json.loads(path.with_suffix(".json").read_text())
    meta = raw["metadata"]
    assert meta["source_sha256"] == source_hashes()
    cfg = Config(**meta["config"])
    check_cfg = replace(cfg, worlds=worlds)
    env, template, _, _, _ = make_two_region(meta["task_spec"], cfg)
    model = POAM.from_data(
        data["initial_observation_positions"],
        data["initial_observations"],
        jax.random.PRNGKey(600000),
        len(template.z),
        noise_std=cfg.observation_std,
        prior_std=cfg.wind_std,
        spectral_count=cfg.poam_spectral_nodes,
        prior_support=template.prior_support,
    )
    root = jax.random.PRNGKey(meta["seed"])
    build = jax.jit(lambda b, e: build_prior_cache(b, e, 129, True))
    snapshot = jax.jit(lambda b: b.predict(env.cells))

    def scorer(anticipate, config):
        @jax.jit
        def score(vectors, x, q, b, cache, noise):
            def one(y):
                cost, (first, future) = evaluate(y, x, q, b, cache, noise, env, config, anticipate)
                return jnp.array(
                    [
                        cost,
                        first.cost.mean(),
                        future.cost.mean(),
                        future.correction.mean(),
                        future.clearance.min(),
                    ]
                ), first.x

            return jax.lax.map(one, vectors)

        return score

    active, frozen = scorer(True, check_cfg), scorer(False, check_cfg)
    oracle_cfg = replace(cfg, worlds=1)
    oracle = scorer(False, oracle_cfg)
    known = KnownWind(KnownBasis(physical_wind(env, meta["physical_wind_spec"])))
    known_cache = KnownCache(jnp.ones((1 + cfg.inner_samples, 1, 2)))
    stride = round(cfg.stage_steps * cfg.dt / meta["execution_dt"])
    output.mkdir(exist_ok=False, parents=True)
    manifest = {
        "input": str(path.relative_to(ROOT)),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "periods": periods,
        "worlds": worlds,
        "banks": banks,
        "label": "Offline development diagnosis; no new executed mission",
        "rows": [],
    }
    for i in range(max(periods) + 1):
        if i in periods:
            b = model.planning_belief()
            mu, cov = snapshot(b)
            np.testing.assert_allclose(mu, data["mean"][i], atol=3e-5, rtol=3e-5)
            np.testing.assert_allclose(jnp.sqrt(cov[:, 0, 0]), data["std"][i], atol=3e-5, rtol=3e-5)
            x = jnp.asarray(data["x"][i * stride])
            q = jnp.asarray(data["q"][i * stride])
            row = raw["periods"][i]
            initials = warm_starts(x, env, cfg, True)
            dimension = len(row["vector"])
            if cfg.budgeted_policy == "branches":
                # The earlier round-trip-seed screen was negative. Keep this
                # policy-class check restricted to its actual search candidates.
                extra, extra_labels = jnp.empty((0, dimension)), []
            else:
                extra, extra_labels = round_trip_seeds(x, env, cfg)
            vectors = jnp.concatenate(
                [
                    initials,
                    jnp.asarray(row["candidates"]).reshape(-1, dimension),
                    jnp.asarray(row["vector"])[None],
                    extra,
                ]
            )
            existing = len(vectors) - len(extra)
            labels = (
                [f"initial_{j}" for j in range(len(initials))]
                + [f"proposal_{j}" for j in range(cfg.diffusion_steps * cfg.proposals)]
                + ["executed_selection"]
                + extra_labels
            )
            sa = []
            sf = []
            differences = []
            for bank in range(banks):
                k1, k2 = jax.random.split(jax.random.PRNGKey(937100 + 100 * i + bank))
                eps = jax.random.normal(k1, (worlds + cfg.inner_samples, *b.sample_shape))
                noise = jax.random.normal(k2, (worlds, cfg.stage_steps + cfg.future_steps, 2))
                cache = build(b, eps)
                a, fa = active(vectors, x, q, b, cache, noise)
                f, ff = frozen(vectors, x, q, b, cache, noise)
                np.testing.assert_allclose(fa, ff, atol=2e-4, rtol=2e-4)
                sa.append(np.asarray(a))
                sf.append(np.asarray(f))
                differences.append(float(jnp.max(jnp.abs(fa - ff))))
                print(
                    "period",
                    i,
                    "bank",
                    bank,
                    "best",
                    int(jnp.argmin(a[:, 0])),
                    int(jnp.argmin(f[:, 0])),
                    flush=True,
                )
            known_scores, _ = oracle(
                vectors,
                x,
                q,
                known,
                known_cache,
                jnp.zeros((1, cfg.stage_steps + cfg.future_steps, 2)),
            )
            sa = np.stack(sa)
            sf = np.stack(sf)
            known_scores = np.asarray(known_scores)

            def summarize(scores, existing=existing, labels=labels):
                valid = np.isfinite(scores[:, :, 0])
                eligible = np.all(valid, axis=0)
                averages = np.where(eligible, scores[:, :, 0].mean(0), np.inf)
                best = int(np.argmin(averages))
                old = int(np.argmin(averages[:existing]))
                return {
                    "best_label": labels[best],
                    "best_mean": float(averages[best]),
                    "best_existing_label": labels[old],
                    "best_existing_mean": float(averages[old]),
                    "all_banks_finite_candidates": int(eligible.sum()),
                }

            cross = []
            for held in range(banks):
                train = [j for j in range(banks) if j != held]
                ai = int(np.argmin(np.mean(sa[train, :, 0], axis=0)))
                fi = int(np.argmin(np.mean(sf[train, :, 0], axis=0)))
                cross.append(
                    {
                        "held_bank": held,
                        "active_choice": labels[ai],
                        "frozen_choice": labels[fi],
                        "held_active_objective_of_active_choice": float(sa[held, ai, 0]),
                        "held_active_objective_of_frozen_choice": float(sa[held, fi, 0]),
                    }
                )
            np.savez_compressed(
                output / f"period_{i:03d}.npz",
                vectors=np.asarray(vectors),
                active=sa,
                frozen=sf,
                oracle=known_scores,
                first_stage_max_difference=np.asarray(differences),
            )
            oracle_index = int(np.argmin(known_scores[:, 0]))
            oracle_old = int(np.argmin(known_scores[:existing, 0]))
            manifest["rows"].append(
                {
                    "period": i,
                    "time_s": i * cfg.stage_steps * cfg.dt,
                    "labels": labels,
                    "existing_count": existing,
                    "active": summarize(sa),
                    "frozen": summarize(sf),
                    "oracle_best": labels[oracle_index],
                    "oracle_best_cost": float(known_scores[oracle_index, 0]),
                    "oracle_existing_best": labels[oracle_old],
                    "oracle_existing_best_cost": float(known_scores[oracle_old, 0]),
                    "cross_bank_checks": cross,
                    "first_stage_max_difference": max(differences),
                }
            )
            (output / "summary.json").write_text(
                json.dumps(serializable(manifest), indent=2) + "\n"
            )
        sl = slice(i * cfg.stage_steps, (i + 1) * cfg.stage_steps)
        model.update(
            data["observation_positions"][sl],
            data["observations"][sl],
            jax.random.fold_in(root, 500000 + i),
            train_steps=cfg.poam_train_steps,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--periods", nargs="+", type=int, default=[0, 15, 30])
    p.add_argument("--worlds", type=int, default=4)
    p.add_argument("--banks", type=int, default=4)
    a = p.parse_args()
    diagnose(a.path.resolve(), a.periods, a.output.resolve(), a.worlds, a.banks)
