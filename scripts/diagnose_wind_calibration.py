"""Offline spatial/prequential POAM diagnostic; never a planner input or a new mission.

Replay the earned observations, comparing the production 24-point model with
40 inducing points and a diagnostic frozen-kernel variant. All variants retain
the complete POAM predictive residual and transported statistics. Physical wind
is used ONLY for evaluation. Spatial coverage is descriptive, not an independent
sample confidence interval or a guarantee for this adaptively collected data.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.wind.poam import POAM, attention_weights

ROOT = Path(__file__).resolve().parents[1]


def summary(error, std):
    if not error.size:
        return {"components": 0}
    sigma = np.broadcast_to(np.asarray(std)[..., None], error.shape)
    z = error / np.maximum(sigma, 1e-8)
    return {
        "components": int(error.size),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_sigma": float(sigma.mean()),
        "coverage_1sigma": float(np.mean(np.abs(z) <= 1)),
        "coverage_1p96sigma": float(np.mean(np.abs(z) <= 1.959963984540054)),
        "standardized_rmse": float(np.sqrt(np.mean(z**2))),
        "max_absolute_z": float(np.max(np.abs(z))),
    }


def spatial_masks(data, spec):
    cells = data["cells"]
    direction = np.asarray(spec["region_b"]) - np.asarray(spec["region_a"])
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    free = np.all(
        np.linalg.norm(cells[:, None] - data["obstacles"][None, :, :2], axis=-1)
        > data["obstacles"][None, :, 2],
        axis=1,
    )
    return {
        "free_field": free,
        "passage": free & (np.abs(cells @ direction) <= 1.1) & (np.abs(cells @ normal) <= 0.45),
        "outer": free & (np.abs(cells @ normal) >= 0.9),
        "monitored_regions": data["targets"] > 0,
    }


def replay(path, output):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(path.with_suffix(".json").read_text())["metadata"]
    cfg = Config(**meta["config"])
    spec = yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text())
    masks = spatial_masks(data, spec)
    prior_x, prior_y = data["initial_observation_positions"], data["initial_observations"]
    periods = len(data["mean"]) - 1
    outputs = {}
    predict = jax.jit(lambda b, x: b.predict(x))
    for name, inducing, training in [
        ("production_24", 24, cfg.poam_train_steps),
        ("capacity_40", 40, cfg.poam_train_steps),
        ("frozen_after_survey_24_diagnostic", 24, 0),
    ]:
        # Prior sample support affects draws only; no draws are used in this diagnostic.
        model = POAM.from_data(
            prior_x,
            prior_y,
            jax.random.PRNGKey(600000),
            inducing,
            noise_std=cfg.observation_std,
            prior_std=cfg.wind_std,
            spectral_count=cfg.poam_spectral_nodes,
        )
        records, all_mean, all_std, innovations, innovation_std = [], [], [], [], []
        for i in range(periods + 1):
            belief = model.planning_belief()
            mean, cov = predict(belief, jnp.asarray(data["cells"]))
            mean, std = np.asarray(mean), np.sqrt(np.maximum(np.asarray(cov[:, 0, 0]), 0))
            if name == "production_24":
                np.testing.assert_allclose(mean, data["mean"][i], atol=3e-5, rtol=3e-5)
                np.testing.assert_allclose(std, data["std"][i], atol=3e-5, rtol=3e-5)
            seen = np.concatenate((prior_x, data["observation_positions"][: i * cfg.stage_steps]))
            distance = np.linalg.norm(data["cells"][:, None] - seen[None], axis=-1).min(axis=1)
            regions = dict(masks)
            regions["near_observed_free"] = masks["free_field"] & (distance <= 0.3)
            regions["far_unobserved_free"] = masks["free_field"] & (distance > 0.3)
            row = {
                "period": i,
                "time_s": i * cfg.stage_steps * cfg.dt,
                "kernel_amplitude": float(np.exp(np.asarray(model.parameters["log_amplitude"]))),
                "regional": {
                    k: summary((mean - data["physical_wind"])[m], std[m])
                    for k, m in regions.items()
                },
                "inducing_positions": np.asarray(model.z).tolist(),
            }
            if i < periods:
                sl = slice(i * cfg.stage_steps, (i + 1) * cfg.stage_steps)
                positions, observations = (
                    data["observation_positions"][sl],
                    data["observations"][sl],
                )
                mu, covariance = predict(belief, jnp.asarray(positions))
                innovation = observations - np.asarray(mu)
                obs_std = np.sqrt(np.asarray(covariance[:, 0, 0]) + cfg.observation_std**2)
                innovations.append(innovation)
                innovation_std.append(obs_std)
                row["next_batch_prediction_before_assimilation"] = summary(innovation, obs_std)
            records.append(row)
            all_mean.append(mean)
            all_std.append(std)
            if i < periods:
                model.update(
                    positions,
                    observations,
                    jax.random.fold_in(jax.random.PRNGKey(meta["seed"]), 500000 + i),
                    train_steps=training,
                )
        weights = np.asarray(attention_weights(model.parameters, jnp.asarray(data["cells"]))) ** 2
        effective_length = weights @ np.asarray(model.lengths)
        outputs[name] = {
            "inducing_count": inducing,
            "real_train_steps_after_survey": training,
            "prequential_all_batches": summary(
                np.concatenate(innovations), np.concatenate(innovation_std)
            ),
            "final_effective_length_by_region": {
                k: float(effective_length[m].mean()) for k, m in masks.items()
            },
            "records": records,
        }
        np.savez_compressed(
            output / f"{path.stem}_{name}.npz",
            mean=all_mean,
            std=all_std,
            innovations=innovations,
            observation_std=innovation_std,
        )
        print(path.stem, name, json.dumps(records[-1]["regional"]["passage"]), flush=True)
        report = {
            "label": "Offline posterior diagnosis, not a replanned comparison or held-out validation",
            "input": str(path.relative_to(ROOT)),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_sha256": source_hashes(),
            "regions": {k: int(v.sum()) for k, v in masks.items()},
            "variants": outputs,
        }
        (output / f"{path.stem}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "input": str(args.directory),
                "arms": ["proposed", "current_belief"],
                "variants": ["production_24", "capacity_40", "frozen_after_survey_24_diagnostic"],
                "interpretation": "Saved data reused, no new observations, truth evaluation only; no selected winner",
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    for arm in ["proposed", "current_belief"]:
        replay((args.directory / f"{arm}.npz").resolve(), args.output)
