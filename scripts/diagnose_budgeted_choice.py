"""Same-state, common-bank causal learning diagnostic on recorded candidate sets.

Offline evaluation only. It does not change the mission or grant a controller
physical truth. Both objectives start with identical actual state, clarity and
POAM. First-stage execution must agree; only earned future updates differ.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.planning.budgeted import evaluate, warm_starts
from voi_diffusion_ipp.viz.demo_data import serializable
from voi_diffusion_ipp.wind.cache import build_prior_cache
from voi_diffusion_ipp.wind.poam import POAM

ROOT = Path(__file__).resolve().parents[1]


def diagnose(path, periods, worlds, banks, output):
    d = np.load(path, allow_pickle=False)
    raw = json.loads(path.with_suffix(".json").read_text())
    meta = raw["metadata"]
    if meta["source_sha256"] != source_hashes():
        raise ValueError("Use the frozen mission source for a causal replay")
    cfg = Config(**meta["config"])
    audit_cfg = replace(cfg, worlds=worlds)
    env, template, _, _, _ = make_two_region(
        yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text()), cfg
    )
    model = POAM.from_data(
        d["initial_observation_positions"],
        d["initial_observations"],
        jax.random.PRNGKey(600000),
        len(template.z),
        noise_std=cfg.observation_std,
        prior_std=cfg.wind_std,
        spectral_count=cfg.poam_spectral_nodes,
        prior_support=template.prior_support,
    )
    root = jax.random.PRNGKey(meta["seed"])
    snapshot = jax.jit(lambda b: b.predict(env.cells))
    build = jax.jit(lambda b, eps: build_prior_cache(b, eps, 129, True))

    def evaluator(anticipate):
        @jax.jit
        def score(vectors, x, q, b, cache, noise):
            def one(v):
                cost, (first, future) = evaluate(
                    v, x, q, b, cache, noise, env, audit_cfg, anticipate
                )
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

    with_learning, frozen = evaluator(True), evaluator(False)
    results = []
    stride = round(cfg.stage_steps * cfg.dt / meta["execution_dt"])
    for i in range(max(periods) + 1):
        if i in periods:
            b = model.planning_belief()
            mu, cov = snapshot(b)
            np.testing.assert_allclose(mu, d["mean"][i], atol=3e-5, rtol=3e-5)
            np.testing.assert_allclose(jnp.sqrt(cov[:, 0, 0]), d["std"][i], atol=3e-5, rtol=3e-5)
            row = raw["periods"][i]
            x, q = jnp.asarray(d["x"][i * stride]), jnp.asarray(d["q"][i * stride])
            initial = np.asarray(warm_starts(x, env, cfg, meta["rapid_shared_initializations"]))
            vectors = np.concatenate(
                (
                    initial,
                    np.asarray(row["candidates"]).reshape(-1, 18),
                    np.asarray(row["vector"])[None],
                )
            )
            labels = [f"shared_seed_{k}" for k in range(len(initial))]
            labels += [f"recorded_proposal_{k}" for k in range(cfg.diffusion_steps * cfg.proposals)]
            labels += ["executed_selection"]
            scores = []
            for bank in range(banks):
                k1, k2 = jax.random.split(jax.random.PRNGKey(912200 + 100 * i + bank))
                eps = jax.random.normal(k1, (worlds + cfg.inner_samples, *b.sample_shape))
                noise = jax.random.normal(k2, (worlds, cfg.stage_steps + cfg.future_steps, 2))
                cache = build(b, eps)
                active, first_a = with_learning(jnp.asarray(vectors), x, q, b, cache, noise)
                passive, first_b = frozen(jnp.asarray(vectors), x, q, b, cache, noise)
                np.testing.assert_allclose(first_a, first_b, atol=2e-4, rtol=2e-4)
                scores.append(
                    {
                        "bank": bank,
                        "with_anticipated_learning": np.asarray(active),
                        "frozen_future_belief": np.asarray(passive),
                        "first_stage_max_state_difference": float(jnp.abs(first_a - first_b).max()),
                    }
                )
                print(
                    "period",
                    i,
                    "bank",
                    bank,
                    "best indices",
                    int(jnp.argmin(active[:, 0])),
                    int(jnp.argmin(passive[:, 0])),
                    flush=True,
                )
            results.append(
                {
                    "period": i,
                    "time_s": i * cfg.stage_steps * cfg.dt,
                    "labels": labels,
                    "vectors": vectors,
                    "banks": scores,
                }
            )
            output.write_text(
                json.dumps(
                    serializable(
                        {
                            "label": "Offline development diagnostic, not new executed evidence",
                            "input": str(path.relative_to(ROOT)),
                            "worlds": worlds,
                            "independent_banks": banks,
                            "columns": [
                                "total_cost",
                                "first_cost",
                                "future_cost",
                                "future_mean_correction",
                                "future_min_clearance",
                            ],
                            "results": results,
                        }
                    ),
                    indent=2,
                )
                + "\n"
            )
        sl = slice(i * cfg.stage_steps, (i + 1) * cfg.stage_steps)
        model.update(
            d["observation_positions"][sl],
            d["observations"][sl],
            jax.random.fold_in(root, 500000 + i),
            train_steps=cfg.poam_train_steps,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--periods", nargs="+", type=int, default=[18, 26])
    p.add_argument("--worlds", type=int, default=8)
    p.add_argument("--banks", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    diagnose(args.path.resolve(), args.periods, args.worlds, args.banks, args.output)
