"""Reproducible whole-field POAM mapping and posterior-sampling diagnostics.

A prescribed survey isolates the estimator. This is not a planner comparison.
All targets come from the independent analytic physical wind, with sensor noise.
"""

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.experiments.scenarios import make_scenario, make_truth, new_poam
from voi_diffusion_ipp.wind.poam_belief import spectral_nodes

out = Path("docs/field_poam_validation")
out.mkdir(parents=True, exist_ok=True)
cfg = load_config("configs/field_poam.yaml")
rows = []
for scenario, seed in [("field_vortices", 300), ("field_shear", 301), ("field_mixed", 302)]:
    c = replace(cfg, scenario=scenario)
    env, initial, _, _ = make_scenario(c)
    truth = make_truth(c, initial, seed)
    p = new_poam(c, initial.basis.centers)
    truth_grid = np.asarray(truth.at(env.cells))
    # 12 alternating survey legs, 12 measurements per leg; no held-out grid labels.
    observations = []
    history = []
    training = []
    for h, y in enumerate(np.linspace(-1.65, 1.65, 12)):
        x = np.linspace(-2.2, 2.2, 12)
        if h % 2:
            x = x[::-1]
        pos = jnp.asarray(np.column_stack((x, np.full_like(x, y))))
        noise = jax.random.normal(
            jax.random.fold_in(jax.random.PRNGKey(seed), h + 800000), (len(pos), 2)
        )
        training.append(
            p.update(
                pos,
                truth.at(pos) + cfg.observation_std * noise,
                jax.random.PRNGKey(900000 + h),
                train_steps=cfg.poam_train_steps,
            )
        )
        observations.extend(np.asarray(pos).tolist())
        mean, variance = p.predict(env.cells)
        err = np.asarray(mean) - truth_grid
        std = np.sqrt(np.asarray(variance[:, 0, 0]))
        history.append(
            {
                "observations": len(observations),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "std": float(std.mean()),
                "coverage95": float(np.mean(np.abs(err) <= 1.96 * std[:, None])),
            }
        )
    b = p.planning_belief()
    exact = np.asarray(b.joint(env.cells)[1])
    convergence = []
    for nodes in [16, 32, 64, 128, 256]:
        candidate = b._replace(basis=b.basis._replace(frequencies=spectral_nodes(nodes)))
        cov = np.asarray(candidate.sample_moments(env.cells)[1])
        convergence.append(
            {
                "spectral_nodes": nodes,
                "relative_covariance_error": float(
                    np.linalg.norm(cov - exact) / np.linalg.norm(exact)
                ),
                "max_variance_error": float(np.max(np.abs(np.diag(cov - exact)))),
            }
        )
    row = {
        "scenario": scenario,
        "seed": seed,
        "prior_rmse": float(np.sqrt(np.mean(truth_grid**2))),
        "history": history,
        "training_losses": training,
        "sampler_convergence": convergence,
    }
    rows.append(row)
    print(scenario, "prior", row["prior_rmse"], "final", history[-1], flush=True)
    (out / "mapping.json").write_text(json.dumps(rows, indent=2) + "\n")
    # Shared scales across truth and estimate; error is actual held-out field error.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    grid = np.asarray(env.cells)
    shape = (cfg.grid_ny, cfg.grid_nx)
    mean = np.asarray(b.predict(env.cells)[0])
    std = np.sqrt(np.asarray(b.predict(env.cells)[1][:, 0, 0]))
    for ax, title, vectors, background in [
        (axes[0], "Physical wind", truth_grid, np.linalg.norm(truth_grid, axis=1)),
        (axes[1], "POAM posterior mean", mean, np.linalg.norm(mean, axis=1)),
        (axes[2], "POAM uncertainty", mean, std),
    ]:
        im = ax.pcolormesh(
            grid[:, 0].reshape(shape),
            grid[:, 1].reshape(shape),
            background.reshape(shape),
            shading="nearest",
            vmin=0,
            vmax=1.2 if ax != axes[2] else 0.6,
            cmap="viridis",
        )
        ii = np.arange(len(grid)).reshape(shape)[::3, ::3].ravel()
        ax.quiver(grid[ii, 0], grid[ii, 1], vectors[ii, 0], vectors[ii, 1], scale=5, color="white")
        ax.set(title=title, aspect="equal", xlabel="x (m)", ylabel="y (m)")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(
        f"{scenario} · independent survey diagnostic · final RMSE {history[-1]['rmse']:.3f}"
    )
    fig.savefig(out / f"{scenario}_mapping.png", dpi=160)
    plt.close(fig)
