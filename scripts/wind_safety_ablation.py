"""Matched policy replay isolating wind in RPCBF, with online full POAM in both arms.

This is a controller-level counterfactual, NOT a replanned MBD comparison. Recorded
waypoint policy vectors are fixed in both arms. Nominal feedback still uses each
arm's own POAM mean; only the ablated safety filter assumes zero wind. The physical
wind, initial conditions, observation-noise stream, and clarity dynamics are shared.
No future information from the source recording is given to either controller.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp import clarity
from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.dynamics import step
from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.scenarios import make_scenario, make_truth, new_poam
from voi_diffusion_ipp.planning.rollout import Rollout, rollout, waypoint_controller
from voi_diffusion_ipp.planning.two_stage import decode
from voi_diffusion_ipp.safety import filter_control
from voi_diffusion_ipp.viz.demo_data import serializable
from voi_diffusion_ipp.wind import InnerFields

ROOT = Path(__file__).resolve().parents[1]


class WindIgnoredByFilter(NamedTuple):
    """Explicit zero-wind safety ablation; never used as the POAM estimator."""

    marker: jax.Array

    def mean_at(self, locations):
        return jnp.zeros((len(jnp.atleast_2d(locations)), 2), dtype=self.marker.dtype)


def replay_rollout(x0, q0, controls, belief, inner, world, noise, env, cfg, wind_aware):
    if wind_aware:
        return rollout(x0, q0, controls, belief, inner, world, noise, env, cfg)
    # Keep the actual RPCBF spline, derivatives and QP. Remove wind only from
    # its backup dynamics, backup compensation and derivative constraints.
    filter_belief = WindIgnoredByFilter(jnp.array(0.0))
    filter_fields = InnerFields(inner.basis, jnp.zeros_like(inner.coefficients))
    physical_wind = lambda p: world.at(p)[0]

    def body(carry, inputs):
        x, q = carry
        waypoint, eps = inputs
        nominal = waypoint_controller(x, waypoint, belief, cfg)
        control, info = filter_control(x, nominal, filter_belief, filter_fields, env, cfg)
        xn = step(x, control, physical_wind, cfg.dt, cfg.linear_drag)
        qn = clarity.step(
            q,
            x[:2],
            xn[:2],
            env.cells,
            cfg.sensor_sigma,
            cfg.sensor_R,
            cfg.clarity_decay if env.process_noise is None else env.process_noise,
            cfg.dt,
        )
        cost = 0.5 * cfg.dt * (clarity.deficit(q, env.targets) + clarity.deficit(qn, env.targets))
        observation = world.at(xn[:2])[0] + belief.noise_std * eps
        mids = jax.vmap(lambda t: step(x, control, physical_wind, t, cfg.linear_drag))(
            cfg.dt * jnp.array([0.25, 0.5, 0.75, 1.0])
        )
        clearance = jnp.min(jax.vmap(env.clearance)(mids))
        return (xn, qn), (
            xn,
            qn,
            nominal,
            control,
            observation,
            cost,
            info.correction,
            info.feasible,
            clearance,
            info.admissible_fraction,
            info.slack,
        )

    _, arrays = jax.lax.scan(body, (x0, q0), (controls, noise))
    x, q, nominal, control, observation, cost, correction, feasible, clearance, area, slack = arrays
    return Rollout(
        jnp.concatenate((x0[None], x)),
        jnp.concatenate((q0[None], q)),
        nominal,
        control,
        observation,
        cost.sum(),
        correction,
        feasible,
        clearance,
        area,
        slack,
    )


def dense_clearance(states, controls, truth, env, dt, subdivisions=32, linear_drag=0.0):
    """Independently integrate each held control to resolve near-grazing contact."""
    wind = lambda p: truth.at(p)[0]

    def segment(x, u):
        def tick(x, _):
            xn = step(x, u, wind, dt / subdivisions, linear_drag)
            return xn, env.clearance(xn)

        return jax.lax.scan(tick, x, None, length=subdivisions)

    ends, clearances = jax.vmap(segment)(states[:-1], controls)
    return jnp.min(clearances, axis=1), jnp.max(jnp.abs(ends - states[1:]))


def run_arm(data, cfg, wind_aware, seed=None):
    env, prior, x, q = make_scenario(cfg)
    seed = data["metrics"]["seed"] if seed is None else seed
    root = jax.random.PRNGKey(seed)
    truth = make_truth(cfg, prior, seed)
    if seed == data["metrics"]["seed"]:
        np.testing.assert_allclose(truth.at(env.cells), data["physical_wind"], atol=1e-6)
    poam = new_poam(cfg, prior.basis.centers)
    belief = poam.planning_belief()
    execute = jax.jit(
        lambda x, q, u, b, eps, noise: replay_rollout(
            x, q, u, b, b.inner_fields(eps), truth, noise, env, cfg, wind_aware
        )
    )
    results, beliefs = [], []

    def snapshot():
        mean, cov = belief.predict(env.cells)
        return serializable(
            {
                "mean": mean,
                "std": jnp.sqrt(cov[:, 0, 0]),
                "rmse": jnp.sqrt(jnp.mean((mean - truth.at(env.cells)) ** 2)),
            }
        )

    beliefs.append(snapshot())
    start = time.perf_counter()
    for h, plan in enumerate(data["planning"]):
        controls, _ = decode(jnp.asarray(plan["selected_vector"]), cfg)
        eps = jax.random.normal(
            jax.random.fold_in(root, 300000 + h), (cfg.inner_samples, *belief.sample_shape)
        )
        noise = jax.random.normal(jax.random.fold_in(root, 400000 + h), (cfg.stage_steps, 2))
        result = execute(x, q, controls, belief, eps, noise)
        result.cost.block_until_ready()
        if cfg.measurements:
            poam.update(
                result.x[1:, :2],
                result.observations,
                jax.random.fold_in(root, 600001 + h),
                train_steps=cfg.poam_train_steps,
            )
            belief = poam.planning_belief()
        x, q = result.x[-1], result.q[-1]
        results.append(jax.tree.map(np.asarray, result))
        beliefs.append(snapshot())
        print(
            f"{cfg.scenario} {'wind-aware' if wind_aware else 'wind-ignored'} "
            f"{h + 1}/{len(data['planning'])} clearance={float(result.clearance.min()):.4f}",
            flush=True,
        )
    mission = {
        name: np.concatenate([getattr(r, name) for r in results])
        for name in (
            "nominal",
            "controls",
            "observations",
            "correction",
            "feasible",
            "clearance",
            "admissible",
            "slack",
        )
    }
    for name in ("x", "q"):
        mission[name] = np.concatenate(
            [getattr(results[0], name)[:1]] + [getattr(r, name)[1:] for r in results]
        )
    deficit = np.maximum(np.asarray(env.targets)[None] - mission["q"], 0).mean(axis=1)
    dense, discrepancy = jax.jit(
        lambda x, u: dense_clearance(x, u, truth, env, cfg.dt, linear_drag=cfg.linear_drag)
    )(jnp.asarray(mission["x"]), jnp.asarray(mission["controls"]))
    mission["dense_clearance"] = np.asarray(dense)
    collision_steps = np.flatnonzero(mission["dense_clearance"] < 0)
    metrics = {
        "integrated_deficit": float(np.trapezoid(deficit, dx=cfg.dt)),
        "minimum_clearance": float(mission["clearance"].min()),
        "dense_minimum_clearance": float(dense.min()),
        "dense_substeps": 32,
        "integration_endpoint_max_discrepancy": float(discrepancy),
        "collision": bool(len(collision_steps)),
        "first_collision_time": float((collision_steps[0] + 1) * cfg.dt)
        if len(collision_steps)
        else None,
        "collision_step_fraction": float(np.mean(mission["dense_clearance"] < 0)),
        "mean_correction": float(mission["correction"].mean()),
        "max_slack": float(mission["slack"].max()),
        "wind_rmse": beliefs[-1]["rmse"],
        "wall_seconds": time.perf_counter() - start,
    }
    if wind_aware and seed == data["metrics"]["seed"]:
        # Positive control: changing the evaluation harness must reproduce the
        # original executed path. A mismatch invalidates the ablation report.
        np.testing.assert_allclose(mission["x"], data["mission"]["x"], atol=2e-4, rtol=1e-4)
        np.testing.assert_allclose(mission["q"], data["mission"]["q"], atol=2e-4, rtol=1e-4)
        np.testing.assert_allclose(
            metrics["integrated_deficit"], data["metrics"]["integrated_deficit"], atol=2e-4
        )
    return serializable({"metrics": metrics, "mission": mission, "beliefs": beliefs})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/wind_safety")
    parser.add_argument(
        "--seed", type=int, help="Independent physical/noise seed for fixed-policy stress testing"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("All outputs must stay inside voi-diffusion-ipp")
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.recording.read_text())
    seed = data["metrics"]["seed"] if args.seed is None else args.seed
    cfg = Config(
        **{k: tuple(v) if k in ("methods", "seeds") else v for k, v in data["config"].items()}
    )
    if (
        not cfg.whole_field
        or cfg.wind_backend != "poam"
        or cfg.control_representation != "waypoint"
    ):
        raise ValueError("Use a full POAM whole-field waypoint recording")
    if data["metrics"]["optimizer_failure_count"]:
        raise ValueError("Recorded fallbacks need explicit policy replay, not selected vectors")
    result = {
        "schema": 1,
        "kind": "matched recorded-policy safety ablation; no replanning",
        "scenario": cfg.scenario,
        "seed": seed,
        "policy_seed": data["metrics"]["seed"],
        "config": data["config"],
        "source_recording": str(args.recording.resolve().relative_to(ROOT)),
        "recording_sha256": hashlib.sha256(args.recording.read_bytes()).hexdigest(),
        "source_sha256": source_hashes(),
        "ablation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": data["environment"],
        "physical_wind": data["physical_wind"],
        "wind_grid": data["wind_grid"],
        "comparison": "Identical recorded waypoint schedule and hidden physical wind; each arm collects its own measurements and runs genuine online POAM. Only RPCBF wind modeling changes.",
        "limitation": "Fixed policy replay isolates the execution filter. It does not establish the outcome of independently replanning with a wind-ignorant planner.",
    }
    if seed != data["metrics"]["seed"]:
        env, prior, _, _ = make_scenario(cfg)
        result["physical_wind"] = serializable(make_truth(cfg, prior, seed).at(env.cells))
    result["wind_aware"] = run_arm(data, cfg, True, seed)
    result["wind_ignored"] = run_arm(data, cfg, False, seed)
    path = output / f"{cfg.scenario}_{seed}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
    temporary.replace(path)
    print(
        json.dumps(
            {name: result[name]["metrics"] for name in ("wind_aware", "wind_ignored")}, indent=2
        )
    )
    print(path)


if __name__ == "__main__":
    main()
