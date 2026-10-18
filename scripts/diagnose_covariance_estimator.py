"""Compare full predictive conditioning with POAM projection at identical locations."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.replay_budgeted import snapshots
from voi_diffusion_ipp.wind.covariance_forecast import CovarianceForecast

ROOT = Path(__file__).resolve().parents[1]


@jax.jit
def compare(b, locations, cells):
    observations = b.mean_at(locations)
    gp = CovarianceForecast.empty(b, len(locations)).condition_locations(locations)
    fixed = b.condition(locations, observations, relocate=False)
    relocated = b.condition(locations, observations, relocate=True)
    return [belief.predict(cells) for belief in [b, gp, fixed, relocated]]


def main():
    path = ROOT / "results/budgeted_missions/covariance_mission_01_8101/covariance_anticipated.npz"
    out = ROOT / "docs/covariance_study/estimator_consistency_01"
    out.mkdir(exist_ok=False)
    protocol = {
        "input": str(path.relative_to(ROOT)),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256": source_hashes(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "periods": [0, 5, 18],
        "locations": "Actual earned next-two-second positions, identical across all update operators. Post-hoc diagnostic, never passed to online planning.",
        "values": "Old predicted mean (zero innovation); frozen kernel for all forecast operators. Actual next POAM snapshot additionally includes real values, relocation and kernel training.",
        "claim": "Separates full GP conditioning from inducing projection and actual estimator changes even with perfectly known measurement locations. Not a calibration or performance trial.",
        "status": "running",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    data = np.load(path, allow_pickle=False)
    rows = []
    for period, cfg, env, b, _, _, _, _ in snapshots(path, protocol["periods"]):
        z = jnp.array(
            data["observation_positions"][period * cfg.stage_steps : (period + 1) * cfg.stage_steps]
        )
        results = jax.block_until_ready(compare(b, z, env.cells))
        means = np.array([r[0] for r in results])
        variances = np.array([r[1][:, 0, 0] for r in results])
        actual = data["std"][period + 1] ** 2
        initial = variances[0]
        assert np.max(variances[1] - initial) < 2e-5
        np.testing.assert_array_equal(means[0], means[1])
        row = {"period": period, "forecast": {}}
        for i, name in enumerate(["before", "full_gp", "poam_fixed_inducing", "poam_relocated"]):
            row["forecast"][name] = {
                "mean_grid_variance": float(variances[i].mean()),
                "mean_variance_reduction": float((initial - variances[i]).mean()),
                "max_mean_change": float(np.max(np.abs(means[i] - means[0]))),
                "mean_absolute_variance_gap_to_gp": float(
                    np.abs(variances[i] - variances[1]).mean()
                ),
                "max_absolute_variance_gap_to_gp": float(np.abs(variances[i] - variances[1]).max()),
            }
        row["actual_next_poam"] = {
            "mean_grid_variance": float(actual.mean()),
            "mean_absolute_variance_gap_to_gp": float(np.abs(actual - variances[1]).mean()),
            "max_absolute_variance_gap_to_gp": float(np.abs(actual - variances[1]).max()),
            "fraction_cells_more_uncertain_than_gp": float(np.mean(actual > variances[1] + 1e-5)),
        }
        np.savez_compressed(
            out / f"maps_{period}.npz",
            cells=np.array(env.cells),
            locations=np.array(z),
            means=means,
            variances=variances,
            actual_next_variance=actual,
        )
        rows.append(row)
        (out / "results.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    protocol["status"] = "complete"
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
