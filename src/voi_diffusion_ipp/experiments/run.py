"""Reproducible paired held-out missions; JSON metrics and NPZ replay data."""

import argparse
import json
import platform
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..clarity import deficit
from ..config import config_dict, load_config
from ..dynamics import step
from ..planning.mpc import initial_policy, next_initial, planner
from ..planning.rollout import rollout
from ..planning.two_stage import decode, route
from ..wind import WorldField
from .scenarios import make_scenario


def mission(cfg, method, seed, out, progress=True):
    env, belief, x, q = make_scenario(cfg)
    # World key namespace differs from optimization/observation/filter namespaces.
    root = jax.random.PRNGKey(seed)
    truth_belief = belief._replace(basis=belief.basis._replace(independent=jnp.array(False)))
    truth = WorldField(
        truth_belief.basis, truth_belief.sample_field(jax.random.fold_in(root, 100000), 1)[0]
    )
    poam = None
    if cfg.wind_backend == "poam" and method != "oracle":
        from ..wind.poam import POAM

        anchors = belief.basis.centers
        poam = POAM(anchors, jax.random.fold_in(root, 600000), float(belief.noise_std))
        belief = poam.planning_belief(anchors)
    if method == "oracle":
        belief = truth_belief._replace(
            mean=truth.coefficients, covariance=jnp.zeros_like(belief.covariance)
        )
    plan = planner(cfg, method)
    execute = jax.jit(
        lambda x, q, u, b, eps, noise: rollout(
            x, q, u, b, b.inner_fields(eps), truth, noise, env, cfg
        )
    )
    execute_fallback = jax.jit(
        lambda x, q, u, b, eps, noise: rollout(
            x, q, u, b, b.inner_fields(eps), truth, noise, env, cfg, feedback=True
        )
    )
    future_exec = jax.jit(
        lambda x, q, u, b, eps: rollout(
            x, q, u, b, b.inner_fields(eps), truth, jnp.zeros((cfg.future_steps, 2)), env, cfg
        )
    )
    nominal_step = jax.jit(
        lambda x, u, b: step(x, u, lambda p: (b.basis.features(p) @ b.mean)[0], cfg.dt)
    )
    initial = jax.jit(lambda x, q, b: initial_policy(x, q, b, env, cfg))(x, q, belief)
    compile_start = time.perf_counter()
    compiled = plan.lower(x, q, belief, jax.random.fold_in(root, 200000), initial, env).compile()
    compile_s = time.perf_counter() - compile_start
    start = time.perf_counter()
    xs = [np.asarray(x)[None]]
    qs = [np.asarray(q)[None]]
    nom_paths = []
    us = []
    safe = []
    corr = []
    feas = []
    clear = []
    areas = []
    slacks = []
    obs = []
    means = []
    variances = []
    branches = []
    timings = []
    costs = []
    traces = []
    diag = []
    optimizer_failures = []
    query = env.cells
    for h in range(cfg.mission_horizons):
        kp = jax.random.fold_in(root, 200000 + h)
        ki = jax.random.fold_in(root, 300000 + h)
        kn = jax.random.fold_in(root, 400000 + h)
        eps = jax.random.normal(ki, (cfg.inner_samples, *belief.mean.shape))
        noise = jax.random.normal(kn, (cfg.stage_steps, 2))
        t = time.perf_counter()
        opt = compiled(x, q, belief, kp, initial, env)
        opt.cost.block_until_ready()
        timings.append(time.perf_counter() - t)
        optimizer_failed = not np.isfinite(opt.cost)
        optimizer_failures.append(optimizer_failed)
        if optimizer_failed:
            print(
                f"OPTIMIZER FAILURE: {method} seed={seed} horizon={h}; executing labeled clarity-feedback fallback through RPCBF",
                flush=True,
            )
        u, fu = decode(opt.vector, cfg)
        result = (execute_fallback if optimizer_failed else execute)(x, q, u, belief, eps, noise)
        result.cost.block_until_ready()
        old = belief
        if cfg.measurements and method != "oracle":
            if poam is None:
                belief = belief.condition(result.x[1:, :2], result.observations)
            else:
                poam.update(
                    result.x[1:, :2],
                    result.observations,
                    jax.random.fold_in(root, 600001 + h),
                    train_steps=cfg.poam_train_steps,
                )
                belief = poam.planning_belief(anchors)
        branch = route(result.x[-1], result.q[-1], belief, env, cfg)
        future_eps = jax.random.normal(
            jax.random.fold_in(root, 500000 + h), (cfg.inner_samples, *belief.mean.shape)
        )
        learned = future_exec(result.x[-1], result.q[-1], fu[branch], belief, future_eps)
        old_branch = route(result.x[-1], result.q[-1], old, env, cfg)
        noinfo = future_exec(result.x[-1], result.q[-1], fu[old_branch], old, future_eps)
        diag.append(
            {
                "learned_cost": float(learned.cost),
                "no_information_cost": float(noinfo.cost),
                "net_benefit": float(noinfo.cost - learned.cost),
                "correction_change": float(
                    jnp.mean(learned.correction) - jnp.mean(noinfo.correction)
                ),
                "admissible_change": float(
                    jnp.mean(learned.admissible) - jnp.mean(noinfo.admissible)
                ),
            }
        )
        nominal = [np.asarray(x)]
        nx = x
        for uu in u:
            nx = nominal_step(nx, uu, old)
            nominal.append(np.asarray(nx))
        nom_paths.append(np.array(nominal))
        x, q = result.x[-1], result.q[-1]
        initial = next_initial(opt.vector, x, q, belief, env, cfg)
        xs.append(np.asarray(result.x[1:]))
        qs.append(np.asarray(result.q[1:]))
        for dest, value in [
            (us, result.nominal),
            (safe, result.controls),
            (corr, result.correction),
            (feas, result.feasible),
            (clear, result.clearance),
            (areas, result.admissible),
            (slacks, result.slack),
            (obs, result.observations),
        ]:
            dest.append(np.asarray(value))
        mean, cov = belief.predict(query)
        means.append(np.asarray(mean))
        variances.append(np.asarray(cov[:, 0, 0]))
        branches.append(int(branch))
        costs.append(float(result.cost))
        traces.append(np.asarray(opt.trace))
        if progress:
            print(
                f"{cfg.scenario} {method} seed={seed} horizon={h + 1}/{cfg.mission_horizons} cost={costs[-1]:.4f} plan={timings[-1]:.2f}s clearance={float(jnp.min(result.clearance)):.3f}",
                flush=True,
            )
    X = np.concatenate(xs)
    Q = np.concatenate(qs)
    C = np.concatenate(clear)
    F = np.concatenate(feas)
    correction = np.concatenate(corr)
    probe = np.linalg.norm(X[1:, :2] - np.asarray(env.probe), axis=1) < 0.4
    access = np.flatnonzero(np.linalg.norm(X[:, :2] - np.asarray(env.roi_a), axis=1) < 0.4)
    mu, var = belief.predict(query)
    err = np.asarray(mu - truth.at(query))
    v = np.asarray(var[:, 0, 0])
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    metrics = {
        "scenario": cfg.scenario,
        "method": method,
        "seed": seed,
        "integrated_deficit": sum(costs),
        "final_deficit": float(deficit(q, env.targets)),
        "roi_a_access": bool(len(access)),
        "roi_a_time": float(access[0] * cfg.dt) if len(access) else None,
        "probe_seconds": float(probe.sum() * cfg.dt),
        "collision": bool(np.min(C) < 0),
        "minimum_clearance": float(np.min(C)),
        "qp_infeasible_fraction": float(np.mean(~F)),
        "correction_frequency": float(np.mean(correction > 1e-3)),
        "mean_correction": float(correction.mean()),
        "max_qp_slack": float(np.max(np.concatenate(slacks))),
        "wind_rmse": float(np.sqrt(np.mean(err**2))),
        "wind_95_coverage": float(np.mean(np.abs(err) <= 1.96 * np.sqrt(v[:, None] + 1e-12))),
        "wind_mean_std": float(np.sqrt(v).mean()),
        "compile_seconds": compile_s,
        "planning_seconds": timings,
        "mean_plan_seconds": float(np.mean(timings)),
        "objective_evaluations_per_plan": int(opt.evaluations),
        "optimizer_failure_count": sum(optimizer_failures),
        "seconds_per_objective": float(np.mean(timings) / int(opt.evaluations)),
        "mission_wall_seconds": time.perf_counter() - start,
        "peak_process_bytes": peak_bytes,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "jax_version": jax.__version__,
        "platform": platform.platform(),
        "matched_continuations": diag,
        "config": config_dict(cfg),
    }
    if poam is not None:
        metrics["poam_snapshot_max_omitted_variance"] = float(
            jnp.max(poam.snapshot_residual_variance(query, anchors))
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.scenario}_{method}_{seed}"
    (out / f"{stem}.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        out / f"{stem}.npz",
        x=X,
        q=Q,
        nominal_paths=np.array(nom_paths),
        nominal=np.concatenate(us),
        control=np.concatenate(safe),
        correction=correction,
        clearance=C,
        feasible=F,
        slack=np.concatenate(slacks),
        admissible=np.concatenate(areas),
        observations=np.concatenate(obs),
        belief_mean=np.array(means),
        belief_variance=np.array(variances),
        branches=np.array(branches),
        traces=np.array(traces),
        cells=np.asarray(env.cells),
        targets=np.asarray(env.targets),
        optimizer_failed=np.array(optimizer_failures),
        obstacles=np.asarray(env.obstacles),
        bounds=np.asarray(env.bounds),
        roi_a=np.asarray(env.roi_a),
        roi_b=np.asarray(env.roi_b),
        probe=np.asarray(env.probe),
        true_wind=np.asarray(truth.at(query)),
    )
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--output", default="results/runs")
    p.add_argument("--horizons", type=int)
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.horizons:
        cfg = replace(cfg, mission_horizons=a.horizons)
    rows = []
    for method in a.methods or cfg.methods:
        for seed in a.seeds or cfg.seeds:
            rows.append(mission(cfg, method, seed, a.output))
    (Path(a.output) / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
