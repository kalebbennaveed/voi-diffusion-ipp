"""Summarize retained inference diagnostics without changing any experiment."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/wind_dense_diagnosis_01"


def nll(errors, sigma):
    errors = np.asarray(errors).reshape(-1, 2)
    sigma = np.asarray(sigma).reshape(-1, 1)
    return float(np.mean(0.5 * (errors / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2 * np.pi)))


def main():
    fig, axs = plt.subplots(2, 3, figsize=(13.2, 7.0), layout="constrained", sharex=True)
    colors = ["#1d669e", "#ab6419", "#45856b"]
    summary = {
        "interpretation": "Same-earned-data diagnostics in one development wind field. Grid components are correlated; descriptive coverage is not a formal confidence interval. No planner changed.",
        "arms": {},
    }
    for row, arm in enumerate(["proposed", "current_belief"]):
        original = json.loads((ROOT / f"docs/wind_calibration_01/{arm}.json").read_text())[
            "variants"
        ]["production_24"]
        forecast = json.loads(
            (ROOT / f"docs/forecast_residual_diagnosis_01/{arm}.json").read_text()
        )
        shorter = json.loads((ROOT / f"docs/kernel_range_diagnosis_01/{arm}.json").read_text())
        dense = json.loads((OUT / f"{arm}.json").read_text())
        reports = [original["records"], forecast["snapshots"], shorter["snapshots"]]
        labels = ["Original POAM", "Coarse forecast + POAM residual", "Half-length-scale POAM"]
        metrics = ["rmse", "mean_sigma", "coverage_1p96sigma"]
        for rows, label, color in zip(reports, labels, colors, strict=True):
            ts = [item["time_s"] for item in rows]
            for col, metric in enumerate(metrics):
                values = [item["regional"]["passage"][metric] for item in rows]
                axs[row, col].plot(ts, values, color=color, label=label, linewidth=1.7)
        for col, metric in enumerate(metrics):
            times = [s["time_s"] for s in dense["snapshots"]]
            values = [
                s["variants"]["dense_same_learned_kernel"]["regional"]["passage"][metric]
                for s in dense["snapshots"]
            ]
            axs[row, col].scatter(
                times,
                values,
                s=35,
                marker="D",
                color="#383c42",
                label="Dense GP, same learned kernel",
                zorder=5,
            )
            axs[row, col].grid(alpha=0.18)
            axs[row, col].set_xlim(0, 120)
            if row:
                axs[row, col].set_xlabel("Mission time of the saved observation history [s]")
        axs[row, 0].set_ylabel(
            "Proposed history" if arm == "proposed" else "Current-belief history"
        )
        axs[row, 2].axhline(0.95, color="#777777", linestyle=":", linewidth=1)
        axs[row, 2].set_ylim(0, 1.05)
        old = np.load(ROOT / f"docs/wind_calibration_01/{arm}_production_24.npz")
        coarse = np.load(ROOT / f"docs/forecast_residual_diagnosis_01/{arm}.npz")
        half = np.load(ROOT / f"docs/kernel_range_diagnosis_01/{arm}.npz")
        summary["arms"][arm] = {
            "final_passage": {
                label: rows[-1]["regional"]["passage"]
                for label, rows in zip(labels, reports, strict=True)
            },
            "prequential": {
                "original": original["prequential_all_batches"],
                "forecast_residual": forecast["prequential"],
                "half_length_scale": shorter["prequential"],
            },
            "mean_marginal_prequential_nll": {
                "original": nll(old["innovations"], old["observation_std"]),
                "forecast_residual": nll(coarse["observation_error"], coarse["observation_std"]),
                "half_length_scale": nll(half["observation_error"], half["observation_std"]),
            },
            "dense_final": dense["snapshots"][-1]["variants"],
        }
    for ax, title in zip(
        axs[0],
        [
            "Passage RMSE [m/s²]",
            "Passage mean posterior σ [m/s²]",
            "Fraction within ±1.96σ (correlated grid)",
        ],
        strict=True,
    ):
        ax.set_title(title, fontsize=11)
    axs[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Wind-model diagnosis: identical earned observations, unchanged physical trajectories",
        fontsize=13,
    )
    fig.savefig(OUT / "calibration_diagnosis.png", dpi=180)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
