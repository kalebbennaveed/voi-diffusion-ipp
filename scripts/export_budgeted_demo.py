"""Replay saved MBD policies for visualization; never rerun policy optimization.

Full POAM is reconstructed from the saved, actually earned measurements. Every
posterior mean is checked against its original snapshot before forecasting.
Candidate paths show cached RPCBF rollouts in planning world 1; selected paths
come from the original uncached verification. Export time is not planner time.
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
import yaml

from voi_diffusion_ipp.config import Config, config_dict
from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.planning.budgeted import evaluate
from voi_diffusion_ipp.viz.demo_data import serializable
from voi_diffusion_ipp.wind.cache import build_prior_cache
from voi_diffusion_ipp.wind.poam import POAM

ROOT = Path(__file__).resolve().parents[1]


def export(path):
    raw = json.loads(path.with_suffix(".json").read_text())
    if not raw["complete"]:
        raise ValueError("Publish completed recordings only")
    d = np.load(path, allow_pickle=False)
    meta = raw["metadata"]
    if meta["source_sha256"] != source_hashes():
        raise ValueError(
            "Numerical source differs from the recorded mission; use its frozen source"
        )
    if (
        meta["task_sha256"]
        != hashlib.sha256((ROOT / "configs/two_region_design.yaml").read_bytes()).hexdigest()
    ):
        raise ValueError("Task definition differs from the recorded mission")
    cfg = Config(**meta["config"])
    method, seed = meta["method"], meta["seed"]
    if method not in ("proposed", "current_belief"):
        raise ValueError("This exporter requires full POAM arms")
    spec = yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text())
    env, model, _, _, _ = make_two_region(spec, cfg)
    root = jax.random.PRNGKey(seed)
    if meta["prior_regime"] == "outer-survey":
        model = POAM.from_data(
            d["initial_observation_positions"],
            d["initial_observations"],
            jax.random.PRNGKey(600000),
            len(model.z),
            noise_std=cfg.observation_std,
            prior_std=cfg.wind_std,
            spectral_count=cfg.poam_spectral_nodes,
            prior_support=model.prior_support,
        )
    else:
        model.update(
            d["initial_observation_positions"], d["initial_observations"], root, train_steps=10
        )
    # Keep the same first physical planning draw and the same independent inner bank.
    # Removing outer draw 2 changes no world-1 control or observation.
    viz_cfg = replace(cfg, worlds=1)
    build = jax.jit(lambda b, e: build_prior_cache(b, e, 129, True))
    snapshot = jax.jit(lambda b: b.predict(env.cells))

    @jax.jit
    def forecast(vectors, x, q, b, cache, noise):
        def one(v):
            _, (first, future) = evaluate(
                v, x, q, b, cache, noise, env, viz_cfg, method == "proposed"
            )
            return jnp.concatenate((first.x[0, :, :2], future.x[0, 1:, :2]))

        return jax.lax.map(one, vectors)

    identity = f"budgeted_{path.parent.name}_{method}_{seed}"
    output = ROOT / "results/demo/live" / identity
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "telemetry_sha256": hashlib.sha256(path.with_suffix(".json").read_bytes()).hexdigest(),
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    manifest_path = output / "forecast_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(
            "Forecast checkpoint provenance differs; preserve it and use a new directory"
        )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    planning = []
    steps = round(cfg.stage_steps * cfg.dt / meta["execution_dt"])
    observation_count = cfg.stage_steps
    snapshots = []
    for i, row in enumerate(raw["periods"]):
        b = model.planning_belief()
        mu, cov = snapshot(b)
        np.testing.assert_allclose(mu, d["mean"][i], atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(jnp.sqrt(cov[:, 0, 0]), d["std"][i], atol=3e-5, rtol=3e-5)
        snapshots.append(
            {"mean": d["mean"][i], "std": d["std"][i], "inducing": np.array(b.basis.centers)}
        )
        keys = jax.random.split(jax.random.fold_in(root, 100 + i), 4)
        all_eps = jax.random.normal(keys[0], (cfg.worlds + cfg.inner_samples, *b.sample_shape))
        eps = jnp.concatenate((all_eps[:1], all_eps[cfg.worlds :]))
        noise = jax.random.normal(keys[1], (cfg.worlds, cfg.stage_steps + cfg.future_steps, 2))[:1]
        hist = row["history"]
        # In reverse order, ab[i-1] = ab[i]/alpha[i], including ab[-1]=1 at last step.
        means = (
            np.array(hist["noisy_after"])
            / np.sqrt(np.array(hist["alphabar"]) / np.array(hist["alpha"]))[:, None]
        )
        vectors = np.concatenate(
            (
                np.array(hist["candidates"]).reshape(-1, 18),
                np.array(hist["incumbent"]),
                means,
                np.array(row["initial_vector"])[None],
                np.array(row["vector"])[None],
            )
        )
        target = output / f"forecast_{i:04d}.npz"
        if target.exists():
            with np.load(target, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["vectors"], vectors)
                paths = saved["paths"]
        else:
            cache = build(b, eps)
            paths = np.array(
                forecast(
                    jnp.asarray(vectors),
                    jnp.asarray(d["x"][i * steps]),
                    jnp.asarray(d["q"][i * steps]),
                    b,
                    cache,
                    noise,
                )
            )
            np.savez_compressed(target, vectors=vectors, paths=paths)
        n = cfg.diffusion_steps * cfg.proposals
        original = np.array(row["selected_model_path"])[:, :2]
        mismatch = float(np.max(np.linalg.norm(paths[-1] - original, axis=-1)))
        planning.append(
            {
                "period": i,
                **hist,
                "trace": row["trace"],
                "candidate_paths": paths[:n].reshape(cfg.diffusion_steps, cfg.proposals, -1, 2),
                "incumbent_paths": paths[n : n + cfg.diffusion_steps],
                "mean_paths": paths[n + cfg.diffusion_steps : -2],
                "initial_path": paths[-2],
                "selected_path": original,
                "selected_cost": row["full_objective"],
                "backup_used": row["backup_used"],
                "policy_type": "continuous posterior feedback",
                "stage_boundary_index": cfg.stage_steps,
                "cached_replay_full_path_max_distance": mismatch,
                "initial_vector": row["initial_vector"],
                "selected_vector": row["vector"],
                "belief_mean": mu,
                "belief_std": jnp.sqrt(cov[:, 0, 0]),
            }
        )
        sl = slice(i * observation_count, (i + 1) * observation_count)
        model.update(
            d["observation_positions"][sl],
            d["observations"][sl],
            jax.random.fold_in(root, 500000 + i),
            train_steps=cfg.poam_train_steps,
        )
        print(method, i + 1, "/", len(raw["periods"]), "replay mismatch", mismatch, flush=True)
    b = model.planning_belief()
    final_mean, final_cov = snapshot(b)
    np.testing.assert_allclose(final_mean, d["mean"][-1], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(jnp.sqrt(final_cov[:, 0, 0]), d["std"][-1], atol=3e-5, rtol=3e-5)
    snapshots.append(
        {"mean": d["mean"][-1], "std": d["std"][-1], "inducing": np.array(b.basis.centers)}
    )
    audit = json.loads(path.with_suffix(".audit.json").read_text())
    # Keep all original 20 Hz execution states and clarity; no synthetic observations.
    ui_cfg = {
        **config_dict(cfg),
        "planning_dt": cfg.dt,
        "dt": meta["execution_dt"],
        "stage_steps": steps,
        "future_steps": round(cfg.future_steps * cfg.dt / meta["execution_dt"]),
        "future_updates": cfg.budgeted_update_steps > 0,
        "mission_horizons": len(planning),
    }
    mission = {
        k: d[v]
        for k, v in {
            "x": "x",
            "q": "q",
            "control": "controls",
            "correction": "correction",
            "slack": "slack",
            "clearance": "clearance",
            "observations": "observations",
            "observation_positions": "observation_positions",
        }.items()
    }
    # UI compression only; full-precision numeric archives remain unchanged.
    for key in ["x", "q"]:
        mission[key] = np.round(mission[key].astype(np.float64), 5 if key == "q" else 6)
    for snapshot in snapshots:
        for key in ["mean", "std"]:
            snapshot[key] = np.round(np.asarray(snapshot[key], dtype=np.float64), 5)
    for plan in planning:
        for key in [
            "candidate_paths",
            "incumbent_paths",
            "mean_paths",
            "initial_path",
            "selected_path",
            "belief_mean",
            "belief_std",
        ]:
            plan[key] = np.round(np.asarray(plan[key], dtype=np.float64), 6)
    # Slack is the recorded soft-QP relaxation, not physical contact.
    mission["observation_times"] = (np.arange(len(d["observations"])) + 1) * cfg.dt
    mission["feasible"] = d["slack"] <= 1e-5
    metrics = {
        "scenario": cfg.scenario,
        "method": method,
        "seed": seed,
        "integrated_deficit": audit["integrated_whole_grid_deficit"],
        "final_deficit": float(np.maximum(d["targets"] - d["q"][-1], 0).mean()),
        "collision": audit["contact_intervals"] > 0,
        "minimum_clearance": audit["minimum_dense_clearance"],
        "max_qp_slack": float(d["slack"].max()),
        "optimizer_failure_count": audit["backup_periods"],
        "mean_plan_seconds": float(np.mean([r["planning_cycle_s"] for r in raw["periods"][2:]])),
        "objective_evaluations_per_plan": cfg.diffusion_steps * cfg.proposals + 9,
        "jax_version": jax.__version__,
    }
    period_s = cfg.stage_steps * cfg.dt
    interval_s = (cfg.budgeted_update_steps or cfg.future_steps) * cfg.dt
    update_times = np.arange(period_s, (cfg.stage_steps + cfg.future_steps) * cfg.dt, interval_s)
    note = (
        "Development experiment, not statistical evidence of learning benefit. Both arms receive the "
        "same archived noisy survey of the two outer routes; the passage is unobserved initially. "
        f"Full POAM real updates at {period_s:g} s boundaries; hypothetical updates at "
        + ", ".join(f"{t:g}" for t in update_times)
        + " s. "
        "All 1,353 cells evolve with Q=0.16; only the two large regions have positive targets. "
        "10 s planning horizon; 20 Hz executed RPCBF and 5 Hz wind observations. "
        "Candidate curves are cached-field RPCBF forecasts in planning world 1; selected curves "
        "are original uncached checks. Export/replay work is separate from planning latency."
    )
    payload = {
        "schema": 2,
        "title": "Two-region persistent monitoring · development",
        "metrics": metrics,
        "config": ui_cfg,
        "mission": mission,
        "environment": {**env._asdict(), "region_size": spec["region_size"]},
        "wind_grid": env.cells,
        "physical_wind": d["physical_wind"],
        "beliefs": snapshots,
        "planning": planning,
        "wind_model": "Full POAM · data-supported inducing initialization",
        "forecast_note": note,
        "research_note": note,
        "source_sha256": meta["source_sha256"],
        "display_precision": "Clarity/field snapshots rounded to 5 decimals, paths to 6; raw archives and reported metrics retain full precision.",
        "research_audit": audit,
        "mission_metadata": meta,
    }
    file = output / "replay.json"
    file.write_text(json.dumps(serializable(payload), separators=(",", ":"), allow_nan=False))
    group = hashlib.sha256(
        json.dumps(
            {k: meta[k] for k in ["config", "prior_regime", "source_sha256"]}, sort_keys=True
        ).encode()
    ).hexdigest()[:16]
    record = {
        "id": identity,
        "kind": "mission",
        "scenario": cfg.scenario,
        "method": method,
        "seed": seed,
        "group": f"budgeted-{group}",
        "label": f"Two large regions · {audit['duration_s']:.0f} s · {period_s:g} s replanning · {method} · development",
        "available_seconds": audit["duration_s"],
        "file": str(file.relative_to(ROOT)),
        "live": False,
        "published_at": time.time(),
    }
    (output / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    return identity


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", type=Path)
    args = p.parse_args()
    for path in args.paths:
        print(export(path.resolve()), flush=True)
