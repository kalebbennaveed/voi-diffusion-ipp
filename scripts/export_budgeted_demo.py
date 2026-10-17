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
from voi_diffusion_ipp.experiments.wind_cases import CASE_LABELS, physical_wind, wind_case_spec
from voi_diffusion_ipp.planning import covariance
from voi_diffusion_ipp.planning.budgeted import evaluate
from voi_diffusion_ipp.planning.sampling import couple_outer_samples
from voi_diffusion_ipp.planning.wind_blind import evaluate_nominal
from voi_diffusion_ipp.viz.demo_data import serializable
from voi_diffusion_ipp.wind.cache import build_prior_cache
from voi_diffusion_ipp.wind.poam import POAM

ROOT = Path(__file__).resolve().parents[1]


def export(path, pair_with=None):
    raw = json.loads(path.with_suffix(".json").read_text())
    if not raw["complete"]:
        raise ValueError("Publish completed recordings only")
    d = np.load(path, allow_pickle=False)
    meta = raw["metadata"]
    if (
        meta.get("planner_form", "compact") != "compact"
        or meta["config"].get("safety_filter", "rpcbf") != "rpcbf"
    ):
        raise ValueError(
            "This exporter labels RPCBF compact-policy forecasts. Safe-MPD experiments "
            "must use their separate comparison report until a correctly labeled replay is added."
        )
    if meta["source_sha256"] != source_hashes():
        raise ValueError(
            "Numerical source differs from the recorded mission; use its frozen source"
        )
    task_path = (ROOT / meta.get("task_file", "configs/two_region_design.yaml")).resolve()
    task_path.relative_to(ROOT)
    if meta["task_sha256"] != hashlib.sha256(task_path.read_bytes()).hexdigest():
        raise ValueError("Task definition differs from the recorded mission")
    cfg = Config(**meta["config"])
    method, seed = meta["method"], meta["seed"]
    if method not in (
        "proposed",
        "current_belief",
        "wind_blind_planner",
        "wind_blind_matched",
        "covariance_anticipated",
        "mean_frozen",
    ):
        raise ValueError("This exporter requires full POAM arms")
    wind_blind = method in ("wind_blind_planner", "wind_blind_matched")
    covariance_mode = method in ("covariance_anticipated", "mean_frozen")
    pairing = None
    if pair_with is not None:
        if not wind_blind:
            raise ValueError("Explicit earlier-source pairing is only for the added baseline")
        previous = json.loads(pair_with.with_suffix(".json").read_text())
        reference_meta = previous["metadata"]
        if not previous["complete"] or reference_meta["method"] != "proposed":
            raise ValueError("Pair with a completed proposed recording")
        for key in [
            "config",
            "seed",
            "prior_regime",
            "execution_dt",
            "observation_dt",
            "task_sha256",
            "rapid_shared_initializations",
        ]:
            if meta[key] != reference_meta[key]:
                raise ValueError(f"Cannot pair different {key}")
        if meta.get("physical_wind_spec") != reference_meta.get("physical_wind_spec"):
            raise ValueError("Cannot pair different physical wind specifications")
        with np.load(pair_with, allow_pickle=False) as reference:
            for key in [
                "physical_wind",
                "cells",
                "targets",
                "obstacles",
                "bounds",
                "initial_observation_positions",
                "initial_observations",
            ]:
                np.testing.assert_array_equal(d[key], reference[key])
            for key in ["x", "q"]:
                np.testing.assert_array_equal(d[key][0], reference[key][0])
                if len(d[key]) != len(reference[key]):
                    raise ValueError("Cannot pair different mission durations")
        reference_id = f"budgeted_{pair_with.parent.name}_proposed_{seed}"
        reference_record = json.loads(
            (ROOT / "results/demo/live" / reference_id / "record.json").read_text()
        )
        pairing = {
            "reference": str(pair_with.relative_to(ROOT)),
            "reference_sha256": hashlib.sha256(pair_with.read_bytes()).hexdigest(),
            "reference_source_sha256": reference_meta["source_sha256"],
            "group": reference_record["group"],
            "scope": "Same task/config/prior/seed/wind/initial states; additive baseline source revision. Not an anticipation-only ablation.",
        }
    spec = meta.get("task_spec") or yaml.safe_load(
        (ROOT / "configs/two_region_design.yaml").read_text()
    )
    env, model, _, _, _ = make_two_region(spec, cfg)
    wind_case = meta.get("wind_case", "original")
    field_spec = meta.get("physical_wind_spec", wind_case_spec("original"))
    np.testing.assert_allclose(
        physical_wind(env, field_spec).at(env.cells), d["physical_wind"], atol=2e-6, rtol=2e-6
    )
    case_label = CASE_LABELS[wind_case]
    if task_path.name == "two_region_open_passage.yaml":
        case_label += " · open passage"
    if cfg.budgeted_policy == "branches":
        case_label += " · posterior-gated routes"
    if meta["prior_regime"] == "routes-survey":
        case_label += " · main route surveyed"
    root = jax.random.PRNGKey(seed)
    if meta["prior_regime"] in ("outer-survey", "routes-survey"):
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
    def forecast(vectors, x, q, b, cache, noise, inner_eps, observation_eps):
        def one(v):
            if wind_blind:
                _, (first, future) = evaluate_nominal(v, x, q, env, viz_cfg)
            elif covariance_mode:
                _, (first, future) = covariance.evaluate(
                    v,
                    x,
                    q,
                    b,
                    cache,
                    inner_eps,
                    observation_eps,
                    env,
                    cfg,
                    method == "covariance_anticipated",
                )
            else:
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
        "pairing": pairing,
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
        all_noise = jax.random.normal(keys[1], (cfg.worlds, cfg.stage_steps + cfg.future_steps, 2))
        all_eps, all_noise = couple_outer_samples(
            all_eps, all_noise, cfg.worlds, meta.get("outer_sampling", "iid")
        )
        eps = jnp.concatenate((all_eps[:1], all_eps[cfg.worlds :]))
        noise = all_noise[:1]
        interval = cfg.budgeted_update_steps or cfg.future_steps
        capacity = cfg.stage_steps + cfg.future_steps - interval
        observation_eps = jax.random.normal(keys[3], (cfg.inner_samples, capacity, 2))
        if covariance_mode:
            eps = all_eps[cfg.worlds :]
        hist = row["history"]
        # In reverse order, ab[i-1] = ab[i]/alpha[i], including ab[-1]=1 at last step.
        means = (
            np.array(hist["noisy_after"])
            / np.sqrt(np.array(hist["alphabar"]) / np.array(hist["alpha"]))[:, None]
        )
        vectors = np.concatenate(
            (
                np.array(hist["candidates"]).reshape(-1, len(row["vector"])),
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
            cache = None if wind_blind else build(b, eps)
            paths = np.array(
                forecast(
                    jnp.asarray(vectors),
                    jnp.asarray(d["x"][i * steps]),
                    jnp.asarray(d["q"][i * steps]),
                    b,
                    cache,
                    noise,
                    all_eps[cfg.worlds :],
                    observation_eps,
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
                "policy_type": "wind-blind nominal"
                if wind_blind
                else "posterior-gated two-route policy"
                if cfg.budgeted_policy == "branches"
                else "continuous posterior feedback",
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
    if meta.get("outer_sampling") == "antithetic":
        note += " Antithetic outer wind and hypothetical measurement-noise pairs; inner RPCBF draws remain independent. Experimental sampling setting, not a changed objective."
    if cfg.budgeted_policy == "branches":
        note += " Two separately optimized future routes share an observation-only posterior gate, reevaluated at causal belief boundaries. Only references switch; physical state and clarity never reset. This is a development policy-class refinement."
    if meta["prior_regime"] == "routes-survey":
        note = note.replace(
            "same archived noisy survey of the two outer routes; the passage is unobserved initially.",
            "same archived noisy survey of the two outer routes plus 21 main-passage locations. This tests low remaining route uncertainty, not a physical-wind-only intervention.",
        )
    if wind_blind:
        note = (
            "Additional development baseline: nominal MBD assumes zero wind and does not include "
            "RPCBF in its objective. Known obstacle collisions are rejected in its nominal model. "
            "Full POAM still learns real wind; only the continuously applied execution RPCBF uses "
            "that belief. Nominal control has no wind compensation. Candidate and selected curves "
            "show unfiltered still-air forecasts, not predicted safety-filtered execution. "
            "Same initial outer-route survey, physical wind, task, 10 s horizon, 2 s replanning, "
            "20 Hz RPCBF and 5 Hz observations as the two existing arms. Comparing against proposed "
            "combines wind-aware planning and anticipated learning; it does not isolate learning value."
        )
    if method == "wind_blind_matched":
        note = note.replace(
            "only the continuously applied execution RPCBF uses "
            "that belief. Nominal control has no wind compensation.",
            "the nominal execution controller AND RPCBF use that belief, "
            "matching the wind-aware arms. Only planning is wind-blind.",
        )
    if covariance_mode:
        note = (
            "Development covariance study. Planning uses deterministic earned-mean wind dynamics, "
            "not sampled outer physical worlds. Full coherent inner RPCBF field samples remain. "
            + (
                "Full predictive GP covariance is conditioned on executed forecast locations at "
                "2,4,6,8 seconds; mean, kernel and inducing inputs are frozen. This differs from "
                "future variational POAM refitting and is not exact expected value of information. "
                if method == "covariance_anticipated"
                else "The forecast belief is frozen; this isolates the covariance-update effect. "
            )
            + "Real execution uses full online POAM, the same wind-aware nominal controller, "
            "20 Hz RPCBF and 5 Hz measurements. Genuine stored MBD candidates are replayed, "
            "not reoptimized. All cells evolve with the archived Q; two large regions have targets. "
            "10-second forecasts, 2-second executed periods. Export time is not planning latency."
        )
    payload = {
        "schema": 2,
        "title": f"{case_label} · two-region persistent monitoring · development",
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
        "research_note": f"{case_label}. " + note,
        "source_sha256": meta["source_sha256"],
        "display_precision": "Clarity/field snapshots rounded to 5 decimals, paths to 6; raw archives and reported metrics retain full precision.",
        "research_audit": audit,
        "mission_metadata": meta,
        "pairing_provenance": pairing,
    }
    file = output / "replay.json"
    file.write_text(json.dumps(serializable(payload), separators=(",", ":"), allow_nan=False))
    group = hashlib.sha256(
        json.dumps(
            {
                **{k: meta[k] for k in ["config", "prior_regime", "source_sha256"]},
                "physical_wind_spec": field_spec,
                "duration_s": audit["duration_s"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    record = {
        "id": identity,
        "kind": "mission",
        "scenario": cfg.scenario,
        "method": method,
        "seed": seed,
        "group": pairing["group"] if pairing else f"budgeted-{group}",
        "label": f"{case_label} · {audit['duration_s']:.0f} s · {method} · development",
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
    p.add_argument("--pair-with", type=Path)
    args = p.parse_args()
    for path in args.paths:
        print(
            export(path.resolve(), args.pair_with.resolve() if args.pair_with else None), flush=True
        )
