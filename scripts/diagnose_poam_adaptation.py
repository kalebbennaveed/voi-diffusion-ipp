"""Predeclared full-POAM adaptation sensitivity on frozen earned observations.

No trajectory is changed and hidden wind is never supplied to inference. Compare
slower learning and larger minibatches while retaining all POAM components.
"""

import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_wind_calibration import spatial_masks, summary

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.wind.poam import POAM, attention_weights

ROOT = Path(__file__).resolve().parents[1]


def replay(job, variants, output):
    path = ROOT / job["input"]
    raw = json.loads(path.with_suffix(".json").read_text())
    meta = raw["metadata"]
    for name, digest in source_hashes().items():
        if name.startswith("src/voi_diffusion_ipp/wind/"):
            assert digest == meta["source_sha256"][name], name
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    cfg = Config(**meta["config"])
    env, template, _, _, _ = make_two_region(meta["task_spec"], cfg)
    masks = spatial_masks(data, meta["task_spec"])
    predict_grid = jax.jit(lambda b: b.predict(env.cells))
    predict_observations = jax.jit(lambda b, x: b.predict(x))
    count = min(job["periods"], len(data["mean"]) - 1)
    report = {
        "input": job["input"],
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "periods": count,
        "variants": {},
    }
    for variant in variants:
        name = variant["name"]
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
        errors, obs_stds, means, stds, records, times = [], [], [], [], [], []
        for i in range(count + 1):
            belief = model.planning_belief()
            mean, cov = predict_grid(belief)
            mean, std = np.asarray(mean), np.sqrt(np.asarray(cov[:, 0, 0]))
            assert np.isfinite(mean).all() and np.isfinite(std).all()
            if name == "baseline":
                np.testing.assert_allclose(mean, data["mean"][i], atol=3e-5, rtol=3e-5)
                np.testing.assert_allclose(std, data["std"][i], atol=3e-5, rtol=3e-5)
            means.append(mean)
            stds.append(std)
            weights = np.asarray(attention_weights(model.parameters, env.cells)) ** 2
            lengths = weights @ np.asarray(model.lengths)
            records.append(
                {
                    "period": i,
                    "amplitude": float(jnp.exp(model.parameters["log_amplitude"])),
                    "regional": {
                        k: summary((mean - data["physical_wind"])[m], std[m])
                        | {"mean_kernel_length_weight": float(lengths[m].mean())}
                        for k, m in masks.items()
                    },
                }
            )
            if i == count:
                break
            sl = slice(i * cfg.stage_steps, (i + 1) * cfg.stage_steps)
            x, y = data["observation_positions"][sl], data["observations"][sl]
            mu, co = predict_observations(belief, jnp.asarray(x))
            errors.append(y - np.asarray(mu))
            obs_stds.append(np.sqrt(np.asarray(co[:, 0, 0]) + cfg.observation_std**2))
            begin = time.perf_counter()
            jax.block_until_ready(
                model.update(
                    x,
                    y,
                    jax.random.fold_in(jax.random.PRNGKey(meta["seed"]), 500000 + i),
                    train_steps=cfg.poam_train_steps,
                    batch_size=variant["batch_size"],
                    lr=variant["lr"],
                )
            )
            times.append(time.perf_counter() - begin)
        np.testing.assert_array_equal(
            model.data_y,
            np.concatenate(
                (data["initial_observations"], data["observations"][: count * cfg.stage_steps])
            ),
        )
        innovation, sigma = np.concatenate(errors), np.concatenate(obs_stds)
        nll = (
            0.5 * (innovation / sigma[:, None]) ** 2
            + np.log(sigma[:, None])
            + 0.5 * np.log(2 * np.pi)
        )
        report["variants"][name] = {
            "settings": variant,
            "prequential": summary(innovation, sigma),
            "mean_marginal_prequential_nll": float(nll.mean()),
            "real_update_median_s": float(np.median(times[2:])),
            "real_update_p95_s": float(np.quantile(times[2:], 0.95)),
            "update_after_ten_periods_median_s": float(np.median(times[10:])),
            "update_after_ten_periods_p95_s": float(np.quantile(times[10:], 0.95)),
            "timing_scope": "Raw timings retained. A larger minibatch compiles new shapes while history grows; the after-ten-periods figures exclude that transient. No complete planning-cycle timing is claimed here.",
            "records": records,
        }
        np.savez_compressed(
            output / f"{job['label']}_{name}.npz",
            mean=means,
            std=stds,
            innovations=innovation,
            observation_std=sigma,
            update_seconds=times,
        )
        (output / f"{job['label']}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            job["label"],
            name,
            "NLL",
            float(nll.mean()),
            "passage",
            records[-1]["regional"]["passage"],
            flush=True,
        )


def main():
    protocol = json.loads((ROOT / "docs/wind_route_study/adaptation_protocol_01.json").read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == protocol["driver_sha256"]
    output = ROOT / "docs/wind_route_study/adaptation_diagnosis_01"
    output.mkdir(exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for job in protocol["jobs"]:
        replay(job, protocol["variants"], output)


if __name__ == "__main__":
    main()
