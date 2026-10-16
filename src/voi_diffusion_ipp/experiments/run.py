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
from ..planning.rollout import continuation, rollout
from ..planning.two_stage import decode, route
from .scenarios import make_scenario, make_truth, new_poam


def mission(cfg, method, seed, out, progress=True, observer=None):
    env, belief, x, q = make_scenario(cfg)
    # World key namespace differs from optimization/observation/filter namespaces.
    root = jax.random.PRNGKey(seed)
    truth = make_truth(cfg, belief, seed)
    poam = None
    if cfg.wind_backend == "poam" and method != "oracle":
        poam = new_poam(cfg, belief.basis.centers)
        belief = poam.planning_belief()
    if method == "oracle":
        if cfg.whole_field:
            raise ValueError(
                "Analytic-field oracle is not yet implemented; do not substitute a GP fit"
            )
        belief = belief._replace(
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
        lambda x, q, u, b, eps, noise, updates: continuation(
            x,
            q,
            u,
            b,
            eps,
            truth,
            noise,
            env,
            cfg,
            updates=updates,
            feedback=(method == "fixed"),
        ),
        static_argnames=("updates",),
    )
    planning_updates = cfg.future_updates and method in ("proposed", "cem", "mppi", "fixed")
    choose = jax.jit(
        lambda x, q, b, fu, eps, updates: route(x, q, b, env, cfg, fu, eps, updates),
        static_argnames=("updates",),
    )
    warm_next = jax.jit(
        lambda previous, x, q, b: next_initial(previous, x, q, b, env, cfg, planning_updates)
    )
    nominal_step = jax.jit(lambda x, u, b: step(x, u, lambda p: b.mean_at(p)[0], cfg.dt))
    initial = jax.jit(lambda x, q, b: initial_policy(x, q, b, env, cfg))(x, q, belief)
    if observer:
        observer("compiling", {"config": cfg, "method": method, "seed": seed})
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
    history = []
    policies = []
    belief_means = [np.asarray(belief.mean)]
    belief_covariances = [np.asarray(belief.covariance)]
    wind_mean_history = [np.asarray(belief.predict(env.cells)[0])]
    wind_variance_history = [np.asarray(belief.predict(env.cells)[1][:, 0, 0])]
    poam_losses = []
    poam_states = []
    for h in range(cfg.mission_horizons):
        kp = jax.random.fold_in(root, 200000 + h)
        ki = jax.random.fold_in(root, 300000 + h)
        kn = jax.random.fold_in(root, 400000 + h)
        eps = jax.random.normal(ki, (cfg.inner_samples, *belief.sample_shape))
        noise = jax.random.normal(kn, (cfg.stage_steps, 2))
        if observer:
            observer("planning", {"horizon": h, "compile_seconds": compile_s})
        t = time.perf_counter()
        opt = compiled(x, q, belief, kp, initial, env)
        opt.cost.block_until_ready()
        timings.append(time.perf_counter() - t)
        history.append(jax.tree.map(np.asarray, opt.history))
        policies.append(np.asarray(opt.vector))
        if observer:
            observer(
                "planned",
                {
                    "horizon": h,
                    "x": x,
                    "q": q,
                    "belief": belief,
                    "opt": opt,
                    "key": kp,
                    "environment": env,
                },
            )
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
                losses = poam.update(
                    result.x[1:, :2],
                    result.observations,
                    jax.random.fold_in(root, 600001 + h),
                    train_steps=cfg.poam_train_steps,
                )
                poam_losses.append(losses)
                belief = poam.planning_belief()
        future_eps = jax.random.normal(
            jax.random.fold_in(root, 500000 + h), (cfg.inner_samples, *belief.sample_shape)
        )
        branch = choose(result.x[-1], result.q[-1], belief, fu, future_eps, planning_updates)
        future_noise = jax.random.normal(
            jax.random.fold_in(root, 510000 + h), (cfg.future_steps, 2)
        )
        learned = future_exec(
            result.x[-1],
            result.q[-1],
            fu[branch],
            belief,
            future_eps,
            future_noise,
            cfg.future_updates,
        )
        old_branch = choose(result.x[-1], result.q[-1], old, fu, future_eps, False)
        noinfo = future_exec(
            result.x[-1], result.q[-1], fu[old_branch], old, future_eps, future_noise, False
        )
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
                "optimized_policy_valid": not optimizer_failed,
            }
        )
        nominal = [np.asarray(x)]
        nx = x
        for uu in result.nominal:
            nx = nominal_step(nx, uu, old)
            nominal.append(np.asarray(nx))
        nom_paths.append(np.array(nominal))
        x, q = result.x[-1], result.q[-1]
        initial = warm_next(opt.vector, x, q, belief)
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
        wind_mean_history.append(np.asarray(mean))
        wind_variance_history.append(np.asarray(cov[:, 0, 0]))
        if poam is not None:
            poam_states.append(
                {
                    "inducing": np.asarray(poam.z).tolist(),
                    "parameters": {k: np.asarray(v).tolist() for k, v in poam.parameters.items()},
                }
            )
        belief_means.append(np.asarray(belief.mean))
        belief_covariances.append(np.asarray(belief.covariance))
        costs.append(float(result.cost))
        traces.append(np.asarray(opt.trace))
        if progress:
            print(
                f"{cfg.scenario} {method} seed={seed} horizon={h + 1}/{cfg.mission_horizons} cost={costs[-1]:.4f} plan={timings[-1]:.2f}s clearance={float(jnp.min(result.clearance)):.3f}",
                flush=True,
            )
        if observer:
            observer(
                "executed",
                {
                    "horizon": h,
                    "result": result,
                    "belief": belief,
                    "branch": int(branch),
                    "cost": costs[-1],
                },
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
        exact = np.asarray(belief.joint(query)[1])
        sampled = np.asarray(belief.sample_moments(query)[1])
        metrics["poam"] = {
            "predictive_residual_retained": True,
            "inducing_count": len(poam.z),
            "spectral_nodes": cfg.poam_spectral_nodes,
            "sampler_covariance_relative_error": float(
                np.linalg.norm(sampled - exact) / np.linalg.norm(exact)
            ),
            "training_losses": poam_losses,
            "states": poam_states,
        }
    metrics["monitoring_cells"] = len(env.cells)
    metrics["monitoring_domain"] = "whole_field" if cfg.whole_field else "legacy_rois"
    if cfg.whole_field:
        for key in ("roi_a_access", "roi_a_time", "probe_seconds"):
            metrics.pop(key)
    truth_grid = np.asarray(truth.at(query))
    metrics["wind_history"] = [
        {
            "time": h * cfg.stage_steps * cfg.dt,
            "rmse": float(np.sqrt(np.mean((mu - truth_grid) ** 2))),
            "mean_std": float(np.sqrt(np.maximum(v, 0)).mean()),
            "coverage_95": float(
                np.mean(np.abs(mu - truth_grid) <= 1.96 * np.sqrt(np.maximum(v, 0))[:, None])
            ),
        }
        for h, (mu, v) in enumerate(zip(wind_mean_history, wind_variance_history))
    ]
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
        wind_mean_history=np.asarray(wind_mean_history),
        wind_variance_history=np.asarray(wind_variance_history),
        policy_vectors=np.asarray(policies),
        belief_coefficients=np.asarray(belief_means),
        belief_covariances=np.asarray(belief_covariances),
        **{
            f"diffusion_{name}": np.stack([getattr(item, name) for item in history])
            for name in opt.history._fields
        },
    )
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/field_poam.yaml")
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
