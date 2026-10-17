"""Causal execution-filter latency ablation around an observed contact.

Both modes start at the same earned POAM/state at 74 s and follow the same
archived nominal policies for six seconds. Kernel adaptation remains at 2 s.
The additional mode conditions on actually earned 5 Hz observations between
those boundaries. No trajectory optimization is rerun; this is not a new MBD
mission or proof of safety. Simulator truth is kept outside the filter.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from audit_budgeted_mission import physical_wind_numpy

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.experiments.wind_cases import physical_wind
from voi_diffusion_ipp.planning.budgeted import segment, split_vector
from voi_diffusion_ipp.wind.poam import POAM

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/wind_route_study/observation_latency_01"
PATH = ROOT / "results/budgeted_missions/crosswind_wind_01/current_belief.npz"


def dense_check(states, controls, meta, obstacles, bounds):
    x = states[:-1].astype(float).copy()
    u = controls.astype(float)
    minimum = np.full(len(x), np.inf)

    def f(x):
        return np.c_[
            x[:, 2:],
            u
            - meta["config"]["linear_drag"] * x[:, 2:]
            + physical_wind_numpy(x[:, :2], meta["task_spec"], meta["physical_wind_spec"]),
        ]

    h = 0.05 / 32
    for _ in range(32):
        k1 = f(x)
        k2 = f(x + h * k1 / 2)
        k3 = f(x + h * k2 / 2)
        k4 = f(x + h * k3)
        x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        gap = (
            np.linalg.norm(x[:, None, :2] - obstacles[None, :, :2], axis=-1) - obstacles[None, :, 2]
        )
        edge = np.minimum(x[:, :2] - bounds[0], bounds[1] - x[:, :2]).min(1)
        minimum = np.minimum(minimum, np.minimum(gap.min(1), edge))
    return {
        "minimum_clearance": float(minimum.min()),
        "contact_intervals": int(np.sum(minimum < 0)),
        "first_contact_time_s": 74 + 0.05 * int(np.flatnonzero(minimum < 0)[0])
        if np.any(minimum < 0)
        else None,
        "max_local_state_error": float(np.max(np.abs(x - states[1:]))),
        "substeps": 32,
    }


def main():
    OUT.mkdir(exist_ok=False)
    raw = json.loads(PATH.with_suffix(".json").read_text())
    meta = raw["metadata"]
    cfg = Config(**meta["config"])
    assert source_hashes() == meta["source_sha256"]
    with np.load(PATH, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    env, template, _, _, _ = make_two_region(meta["task_spec"], cfg)
    truth = physical_wind(env, meta["physical_wind_spec"])
    root = jax.random.PRNGKey(meta["seed"])
    period_start = 37
    periods = 3
    stride = 40
    protocol = {
        "label": "Fixed-policy, same-state observation-availability diagnostic; not a replanned comparison",
        "start_s": 74,
        "duration_s": 6,
        "kernel_adaptation_seconds": 2,
        "conditions": [
            "observations applied at 2 s boundaries",
            "additional 5 Hz earned-data conditioning, fixed kernel and inducing positions within each 2 s period",
        ],
        "nominal_policies": "Archived current-belief selected vectors and fallback choices at the same timestamps",
        "source_sha256": meta["source_sha256"],
        "input_sha256": hashlib.sha256(PATH.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement_noise": "Same PRNG innovations; measurement positions follow each counterfactual executed path",
        "assertion": "The 2 s condition must reproduce the original stored path before interpreting the additional-update condition.",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    part_cfg = replace(cfg, dt=0.05, stage_steps=4)
    actual_cfg = replace(cfg, dt=0.05, stage_steps=40)
    execute = jax.jit(
        lambda x, q, refs, b, eps, noise, backup: segment(
            x, q, refs, b, b.inner_fields(eps), truth, noise, env, part_cfg, False, backup
        )
    )
    condition = jax.jit(lambda b, p, y: b.condition(p, y, relocate=False))
    results = {}
    for cadence in [2.0, 0.2]:
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
        for i in range(period_start):
            sl = slice(i * cfg.stage_steps, (i + 1) * cfg.stage_steps)
            model.update(
                data["observation_positions"][sl],
                data["observations"][sl],
                jax.random.fold_in(root, 500000 + i),
                train_steps=cfg.poam_train_steps,
            )
        b = model.planning_belief()
        mu, cov = b.predict(env.cells)
        np.testing.assert_allclose(mu, data["mean"][period_start], atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(
            jnp.sqrt(cov[:, 0, 0]), data["std"][period_start], atol=3e-5, rtol=3e-5
        )
        x = jnp.asarray(data["x"][period_start * stride])
        q = jnp.asarray(data["q"][period_start * stride])
        xs = [np.asarray(x)[None]]
        qs = [np.asarray(q)[None]]
        us = []
        slack = []
        earned = []
        earned_positions = []
        for i in range(period_start, period_start + periods):
            keys = jax.random.split(jax.random.fold_in(root, 100 + i), 4)
            eps = jax.random.normal(keys[0], (cfg.worlds + cfg.inner_samples, *b.sample_shape))[
                cfg.worlds :
            ]
            noises = jax.random.normal(jax.random.fold_in(root, 400000 + i), (stride, 2))
            refs = split_vector(jnp.asarray(raw["periods"][i]["vector"]), actual_cfg)[0]
            positions = []
            observations = []
            for j in range(10):
                sl = slice(4 * j, 4 * j + 4)
                part = execute(
                    x, q, refs[sl], b, eps, noises[sl], jnp.array(raw["periods"][i]["backup_used"])
                )
                x, q = part.x[-1], part.q[-1]
                positions.append(x[:2])
                observations.append(part.observations[-1])
                xs.append(np.asarray(part.x[1:]))
                qs.append(np.asarray(part.q[1:]))
                us.append(np.asarray(part.controls))
                slack.append(np.asarray(part.slack))
                if cadence < 2.0 and j < 9:
                    b = condition(b, x[None, :2], part.observations[-1:])
            pp = jnp.stack(positions)
            yy = jnp.stack(observations)
            earned.append(np.asarray(yy))
            earned_positions.append(np.asarray(pp))
            model.update(
                pp, yy, jax.random.fold_in(root, 500000 + i), train_steps=cfg.poam_train_steps
            )
            b = model.planning_belief()
        states = np.concatenate(xs)
        controls = np.concatenate(us)
        clarity = np.concatenate(qs)
        check = dense_check(states, controls, meta, data["obstacles"], data["bounds"])
        if cadence == 2.0:
            reference = data["x"][period_start * stride : (period_start + periods) * stride + 1]
            check["original_state_max_error"] = float(np.max(np.abs(states - reference)))
            np.testing.assert_allclose(states, reference, atol=8e-5, rtol=8e-5)
        name = "boundary_2s" if cadence == 2.0 else "earned_5hz"
        np.savez_compressed(
            OUT / f"{name}.npz",
            x=states,
            controls=controls,
            q=clarity,
            slack=np.concatenate(slack),
            observations=np.concatenate(earned),
            observation_positions=np.concatenate(earned_positions),
        )
        results[name] = check
        (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(name, check, flush=True)


if __name__ == "__main__":
    main()
