"""Export genuine JAX mission and diffusion telemetry for the local research demo."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..config import config_dict, load_config
from ..experiments.run import mission
from ..experiments.scenarios import make_scenario, make_truth
from ..planning.two_stage import evaluate_one, make_buffers
from ..wind import WorldField

ROOT = Path(__file__).resolve().parents[3]


def serializable(value):
    """Strict JSON: infeasible objective values are null, never invalid JSON Infinity."""
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return serializable(np.asarray(value).tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def export_demo(cfg, method, seed, output, status=None):
    """Rerun the actual planner; extra visualization rollouts never affect its decisions."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env, prior, _, _ = make_scenario(cfg)
    xx, yy = np.meshgrid(np.linspace(-2.35, 2.35, 25), np.linspace(-1.8, 1.8, 19))
    grid = env.cells if cfg.whole_field else jnp.asarray(np.stack((xx.ravel(), yy.ravel()), axis=1))
    truth = make_truth(cfg, prior, seed)
    beliefs = []

    def capture_belief(belief):
        mean, covariance = belief.predict(grid)
        error = mean - truth.at(grid)
        return {
            "mean": mean,
            "std": jnp.sqrt(covariance[:, 0, 0]),
            "error": jnp.linalg.norm(error, axis=1),
            "rmse": jnp.sqrt(jnp.mean(error**2)),
            "inducing": belief.basis.centers if cfg.wind_backend == "poam" else [],
        }

    mode = "proposed" if method in ("proposed", "cem", "mppi") else method
    captured = []

    @jax.jit
    def forecast(vectors, x, q, belief, buffers):
        def one(vector):
            result = evaluate_one(
                vector,
                x,
                q,
                belief,
                buffers.world_coefficients[0],
                buffers.observation_noise[0],
                buffers.inner_prior_eps,
                buffers.inner_future_eps,
                env,
                cfg,
                mode,
                buffers.future_noise[0],
            )
            return jnp.concatenate((result.first.x[:, :2], result.future.x[1:, :2])), result.branch

        return jax.lax.map(one, vectors)

    def observer(event, payload):
        if event == "compiling":
            if status:
                status("Compiling the JAX planner")
            return
        if event == "planning" and status:
            status(
                f"Period {payload['horizon'] + 1}/{cfg.mission_horizons}: optimizing full-field clarity cost"
            )
        if event == "planned":
            h, opt, belief = payload["horizon"], payload["opt"], payload["belief"]
            if not beliefs:
                beliefs.append(capture_belief(belief))
            if status:
                status(f"Period {h + 1}/{cfg.mission_horizons}: recording diffusion forecasts")
            ks, _ = jax.random.split(payload["key"])
            buffers = make_buffers(belief, ks, cfg)
            hist = opt.history
            candidate_count = cfg.diffusion_steps * cfg.proposals
            vectors = jnp.concatenate(
                (
                    hist.candidates.reshape(-1, opt.vector.size),
                    hist.incumbent,
                    opt.iterates,
                    opt.initial[None],
                    opt.vector[None],
                )
            )
            paths, branch = forecast(vectors, payload["x"], payload["q"], belief, buffers)
            paths, branch = np.asarray(paths), np.asarray(branch)
            mean, covariance = belief.predict(grid)
            captured.append(
                {
                    "period": h,
                    "trace": opt.trace,
                    "candidates": hist.candidates,
                    "costs": hist.costs,
                    "weights": hist.weights,
                    "noisy_before": hist.noisy_before,
                    "noisy_after": hist.noisy_after,
                    "score": hist.score,
                    "alpha": hist.alpha,
                    "alphabar": hist.alphabar,
                    "candidate_paths": paths[:candidate_count].reshape(
                        cfg.diffusion_steps, cfg.proposals, -1, 2
                    ),
                    "candidate_branches": branch[:candidate_count].reshape(
                        cfg.diffusion_steps, cfg.proposals
                    ),
                    "incumbent_paths": paths[
                        candidate_count : candidate_count + cfg.diffusion_steps
                    ],
                    "mean_paths": paths[candidate_count + cfg.diffusion_steps : -2],
                    "initial_path": paths[-2],
                    "selected_path": paths[-1],
                    "selected_cost": opt.cost,
                    "selected_branch": branch[-1],
                    "belief_mean": mean,
                    "belief_std": jnp.sqrt(covariance[:, 0, 0]),
                    "planning_world_wind": WorldField(
                        belief.basis, buffers.world_coefficients[0]
                    ).at(grid),
                    "initial_vector": opt.initial,
                    "selected_vector": opt.vector,
                }
            )
        elif event == "executed":
            beliefs.append(capture_belief(payload["belief"]))
            if status:
                status(
                    f"Period {payload['horizon'] + 1}/{cfg.mission_horizons}: executed and updated belief"
                )

    metrics = mission(cfg, method, seed, output, observer=observer)
    stem = f"{cfg.scenario}_{method}_{seed}"
    with np.load(output / f"{stem}.npz") as a:
        arrays = {
            name: a[name]
            for name in (
                "x",
                "q",
                "nominal_paths",
                "nominal",
                "control",
                "correction",
                "clearance",
                "feasible",
                "slack",
                "admissible",
                "observations",
                "branches",
            )
        }
    source_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src").rglob("*.py"))
    }
    data = serializable(
        {
            "schema": 2,
            "title": "VoI diffusion · research replay",
            "metrics": metrics,
            "config": config_dict(cfg),
            "mission": arrays,
            "environment": {name: getattr(env, name) for name in env._fields},
            "wind_grid": grid,
            "beliefs": beliefs,
            "physical_wind": truth.at(grid),
            "planning": captured,
            "source_sha256": source_hashes,
            "wind_model": "Full POAM predictive covariance; decoupled posterior samples with tested spectral quadrature"
            if cfg.wind_backend == "poam"
            else "Legacy finite-feature GP",
            "forecast_note": "Paths show outer planning world 1. Candidate costs and weights use all outer worlds. Forecasts are safety-filtered simulations, not executed trajectories. The denoised mean is not guaranteed feasible; execution uses the best evaluated policy.",
        }
    )
    path = output / f"{stem}.demo.json"
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    if status:
        status("Complete")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/field_poam.yaml")
    p.add_argument("--seed", type=int, default=102)
    p.add_argument("--method", default="proposed")
    p.add_argument("--output", default="results/demo")
    p.add_argument("--horizons", type=int)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.horizons:
        cfg = replace(cfg, mission_horizons=args.horizons)
    print(export_demo(cfg, args.method, args.seed, args.output, print))


if __name__ == "__main__":
    main()
