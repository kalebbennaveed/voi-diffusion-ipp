"""Isolate one earned POAM update in matched, uncached continuation rollouts.

This diagnostic freezes the t=2s posterior for the remaining eight seconds to
separate the policy-reference and controller/RPCBF effects of that ONE update.
It is not the deployed planner's four-update forecast or a new task objective.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_mbd_reliability import digest, save

from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.replay_budgeted import snapshots
from voi_diffusion_ipp.planning import budgeted
from voi_diffusion_ipp.wind import InnerFields, WorldField


def make_evaluator(cfg, cells):
    @jax.jit
    def run(vector, x, q, belief, eps, noise, env):
        idx = jnp.asarray(cells)
        ee = env._replace(cells=env.cells[idx], targets=env.targets[idx], process_noise=env.process_noise[idx])
        scale = len(idx) / len(q)
        bank = belief.inner_fields(eps)
        inner = InnerFields(bank.basis, bank.coefficients[cfg.worlds:])
        controls, _, _ = budgeted.split_vector(vector, cfg)
        matrices = budgeted.reference_matrices(cfg.future_steps * cfg.dt, cfg.future_steps)

        def world_run(item):
            coefficients, ns = item
            world = WorldField(bank.basis, coefficients)
            first = budgeted.segment(x, q[idx], controls, belief, inner, world, ns[:cfg.stage_steps], ee, cfg, False)
            posterior = belief.condition(first.x[1:, :2], first.observations, relocate=cfg.hypothetical_relocation)
            updated_bank = posterior.inner_fields(eps)
            updated_inner = InnerFields(updated_bank.basis, updated_bank.coefficients[cfg.worlds:])
            references = [budgeted.future_reference(vector, first.x[-1], b, env, matrices, cfg) for b in (belief, posterior)]
            paths, metrics = [], []
            # no update; controller/filter only; reference only; both
            for update_control, update_reference in ((False, False), (True, False), (False, True), (True, True)):
                future = budgeted.segment(first.x[-1], first.q[-1], references[int(update_reference)],
                    posterior if update_control else belief, updated_inner if update_control else inner,
                    world, ns[cfg.stage_steps:], ee, cfg, True)
                paths.append(future.x)
                metrics.append(jnp.array([first.cost * scale, future.cost * scale,
                    (first.cost + future.cost) * scale, future.correction.mean(),
                    jnp.minimum(first.clearance.min(), future.clearance.min()), future.slack.max()]))
            # Query the SAME future reference locations before/after learning.
            route = references[0][:, 0]
            old_mu, old_cov = belief.predict(route)
            new_mu, new_cov = posterior.predict(route)
            old_probe_mu, old_probe_cov = belief.predict(env.probe)
            new_probe_mu, new_probe_cov = posterior.predict(env.probe)
            return {"first_path": first.x, "path": jnp.stack(paths), "metrics": jnp.stack(metrics),
                    "reference_position_shift_max": jnp.linalg.norm(references[1][:, 0] - references[0][:, 0], axis=-1).max(),
                    "route_mean_change_rms": jnp.sqrt(jnp.mean((new_mu - old_mu)**2)),
                    "route_std_before": jnp.sqrt(old_cov[:, 0, 0]).mean(),
                    "route_std_after": jnp.sqrt(new_cov[:, 0, 0]).mean(),
                    "probe_mean_before": old_probe_mu[0], "probe_mean_after": new_probe_mu[0],
                    "probe_std_before": jnp.sqrt(old_probe_cov[0, 0, 0]),
                    "probe_std_after": jnp.sqrt(new_probe_cov[0, 0, 0])}
        return jax.lax.map(world_run, (bank.coefficients[:cfg.worlds], noise))
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    assert protocol["source_sha256"] == source_hashes()
    assert digest(__file__) == protocol["driver_sha256"]
    folder = Path(protocol["output"])
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    for ji, job in enumerate(protocol["jobs"]):
        path = Path(job["input"])
        assert digest(path) == job["input_sha256"]
        search_path = Path(protocol["search_output"]) / (job["name"] + ".json")
        search = json.loads(search_path.read_text())
        policies = [r for r in search["optimization"] if r["variant"] == "4x8_w2"]
        assert len(policies) == 4 and search["complete"]
        for period, base, env, b, x, q, _, _ in snapshots(path, [job["period"]]):
            cfg = replace(base, worlds=4, reject_collisions=False, budgeted_update_steps=0)
            cells = tuple(np.flatnonzero(np.asarray(env.targets) > 0))
            evaluator = make_evaluator(cfg, cells)
            key1, key2 = jax.random.split(jax.random.PRNGKey(protocol["seed"] + ji))
            eps = jax.random.normal(key1, (cfg.worlds + cfg.inner_samples, *b.sample_shape))
            noise = jax.random.normal(key2, (cfg.worlds, cfg.stage_steps + cfg.future_steps, 2))
            checks = [jax.jit(lambda v, anticipate=anticipate, x=x, q=q, b=b, noise=noise, env=env, cfg=cfg, eps=eps:
                budgeted.evaluate(v, x, q, b, None, noise, env, cfg, anticipate, True, eps))
                for anticipate in (False, True)]
            for pi, row in enumerate(policies):
                vector = jnp.asarray(row["vector"])
                data = jax.block_until_ready(evaluator(vector, x, q, b, eps, noise, env))
                # Independent call to the production evaluator with exactly one
                # update. This checks the diagnostic endpoints, not its premise.
                parity = []
                for evaluate, arm in zip(checks, (0, 3), strict=True):
                    _, (first, future) = jax.block_until_ready(evaluate(vector))
                    np.testing.assert_allclose(first.x, data["first_path"], atol=5e-5, rtol=2e-5)
                    np.testing.assert_allclose(future.x, data["path"][:, arm], atol=5e-5, rtol=2e-5)
                    np.testing.assert_allclose(first.cost + future.cost, data["metrics"][:, arm, 2], atol=2e-6, rtol=2e-5)
                    parity.append(float(jnp.max(jnp.abs(future.x-data["path"][:, arm]))))
                output = folder / f'{job["name"]}_{pi}.npz'
                np.savez_compressed(output, **data)
                record = {"job": job["name"], "period": period, "anticipate": row["anticipate"], "optimizer_seed": row["seed"],
                          "source_policy_file": str(search_path), "source_policy_sha256": digest(search_path),
                          "source_policy_vector": row["vector"], "data_sha256": digest(output), "parity_path_error": parity,
                          **{k:v for k,v in data.items() if k not in ("path", "first_path")}}
                rows.append(record)
                save(folder/'results.json', {"complete": False, "rows": rows})
                print(job["name"], row["anticipate"], row["seed"], np.asarray(data["metrics"][:, :, 2]).mean(axis=0), flush=True)
    save(folder/'results.json', {"complete": True, "rows": rows})


if __name__ == "__main__":
    main()
