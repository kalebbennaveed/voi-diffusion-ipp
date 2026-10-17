"""One declared shorter-scale full-POAM replay; not a new mission.

The existing learned kernel over-smooths the passage even with dense inference.
Halving all primitive length scales is an exploratory prior-range sensitivity,
not a fitted optimum or a claimed safety guarantee. Keep neural/amplitude
learning, all transported data, inducing relocation and the full residual.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from diagnose_wind_calibration import spatial_masks, summary

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.wind.poam import POAM, attention_weights

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (0.15, 0.4, 0.65, 0.9)


def replay(path, output):
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(path.with_suffix(".json").read_text())["metadata"]
    cfg = Config(**metadata["config"])
    spec = yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text())
    masks = spatial_masks(data, spec)
    model = POAM.from_data(
        data["initial_observation_positions"],
        data["initial_observations"],
        jax.random.PRNGKey(600000),
        24,
        lengths=LENGTHS,
        noise_std=cfg.observation_std,
        prior_std=cfg.wind_std,
        spectral_count=cfg.poam_spectral_nodes,
    )
    predict = jax.jit(lambda b, x: b.predict(x))
    means, stds, innovations, noise_stds, snapshots, parameters = [], [], [], [], [], []
    periods = len(data["mean"]) - 1
    for i in range(periods + 1):
        belief = model.planning_belief()
        mean, covariance = predict(belief, jnp.asarray(data["cells"]))
        mean, std = np.asarray(mean), np.sqrt(np.asarray(covariance[:, 0, 0]))
        assert np.isfinite(mean).all() and np.isfinite(std).all()
        means.append(mean)
        stds.append(std)
        parameters.append({k: np.asarray(v) for k, v in model.parameters.items()})
        weights = np.asarray(attention_weights(model.parameters, jnp.asarray(data["cells"]))) ** 2
        weighted_lengths = weights @ np.asarray(model.lengths)
        snapshots.append(
            {
                "period": i,
                "time_s": i * cfg.stage_steps * cfg.dt,
                "observations": len(model.data_x),
                "amplitude": float(jnp.exp(model.parameters["log_amplitude"])),
                "regional": {
                    name: {
                        **summary((mean - data["physical_wind"])[mask], std[mask]),
                        "mean_kernel_length_weight": float(weighted_lengths[mask].mean()),
                    }
                    for name, mask in masks.items()
                },
            }
        )
        if i in (0, 15, 30, periods):
            print(path.stem, i, snapshots[-1]["regional"]["passage"], flush=True)
        if i == periods:
            break
        sl = slice(i * cfg.stage_steps, (i + 1) * cfg.stage_steps)
        x, y = data["observation_positions"][sl], data["observations"][sl]
        mu, cov = predict(belief, jnp.asarray(x))
        innovations.append(y - np.asarray(mu))
        noise_stds.append(np.sqrt(np.asarray(cov[:, 0, 0]) + cfg.observation_std**2))
        model.update(
            x,
            y,
            jax.random.fold_in(jax.random.PRNGKey(metadata["seed"]), 500000 + i),
            train_steps=cfg.poam_train_steps,
        )
    errors, sigma = np.concatenate(innovations), np.concatenate(noise_stds)
    np.testing.assert_array_equal(
        model.data_y, np.concatenate((data["initial_observations"], data["observations"]))
    )
    report = {
        "input": str(path.relative_to(ROOT)),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256": source_hashes(),
        "lengths": LENGTHS,
        "prequential": summary(errors, sigma),
        "mean_marginal_prequential_nll": float(
            np.mean(
                0.5 * (errors / sigma[:, None]) ** 2
                + np.log(sigma[:, None])
                + 0.5 * np.log(2 * np.pi)
            )
        ),
        "snapshots": snapshots,
    }
    (output / f"{path.stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    arrays = {
        "cells": data["cells"],
        "physical_wind": data["physical_wind"],
        "mean": np.asarray(means),
        "std": np.asarray(stds),
        "observation_error": errors,
        "observation_std": sigma,
        "lengths": np.asarray(model.lengths),
    }
    arrays.update({"parameter_" + k: np.stack([p[k] for p in parameters]) for k in parameters[0]})
    np.savez_compressed(output / f"{path.stem}.npz", **arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "arms": ["proposed", "current_belief"],
                "lengths": LENGTHS,
                "baseline_lengths": [0.3, 0.8, 1.3, 1.8],
                "hypothesis": "Long correlation inferred from adaptive narrow flight-line data may underestimate transverse uncertainty; test exactly one shorter prior range, retaining full POAM training.",
                "design_status": "Exploratory sensitivity declared after the original calibration and dense-kernel findings, not held-out model selection.",
                "scope": "Offline inference only on exactly the same earned observations. Neither optimizer nor physical trajectory changes. Hidden truth used only in scoring.",
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    for arm in ["proposed", "current_belief"]:
        replay((args.directory / f"{arm}.npz").resolve(), args.output)
