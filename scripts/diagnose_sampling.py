"""Post-hoc coverage of realized localized fields by execution's inner samples.

Truth is used only for this offline diagnostic, never to choose a controller.
"""

import json
from pathlib import Path

import jax
import numpy as np
from scipy.spatial import ConvexHull

from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.experiments.scenarios import make_scenario


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/recourse_multistage.yaml")
    _, prior, _, _ = make_scenario(cfg)
    records = []
    for seed in (204, 206):
        truth = np.asarray(
            prior.sample_field(jax.random.fold_in(jax.random.PRNGKey(seed), 100000), 1)[0]
        )[0]
        with np.load(
            root
            / f"results/multistage_validation/repeated_updates/localized_corridor_proposed_{seed}.npz"
        ) as a:
            for h in sorted(set(np.flatnonzero(~a["feasible"]) // cfg.stage_steps)):
                mean = a["belief_coefficients"][h][0]
                sigma = float(np.sqrt(a["belief_covariances"][h][0, 0]))
                z = (truth - mean) / sigma
                eps = np.asarray(
                    jax.random.normal(
                        jax.random.fold_in(jax.random.PRNGKey(seed), 300000 + int(h)),
                        (cfg.inner_samples, *prior.mean.shape),
                    )
                )[:, 0, :]
                hull = ConvexHull(eps)
                residual = float(np.max(hull.equations[:, :2] @ z + hull.equations[:, 2]))
                records.append(
                    {
                        "seed": seed,
                        "period": int(h),
                        "truth_coefficients": truth.tolist(),
                        "posterior_mean": mean.tolist(),
                        "posterior_std": sigma,
                        "standardized_truth": z.tolist(),
                        "inner_standardized_draws": eps.tolist(),
                        "truth_inside_inner_convex_hull": residual <= 0,
                        "convex_hull_residual": residual,
                        "interpretation": "Post-hoc diagnostic only. Coverage by four posterior draws is limited; this does not identify the sole cause of QP infeasibility.",
                    }
                )
    path = root / "results/multistage_sampling_diagnosis.json"
    path.write_text(json.dumps(records, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
