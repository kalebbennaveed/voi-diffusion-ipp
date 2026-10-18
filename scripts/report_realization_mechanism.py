"""Report every arm of the single-earned-update mechanism diagnostic."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    folder = Path("docs/mbd_reliability")
    data = json.loads((folder / "mechanism_02/results.json").read_text())
    assert data["complete"] and len(data["rows"]) == 12
    rows = []
    for row in data["rows"]:
        m = np.asarray(row["metrics"])
        assert m.shape == (4, 4, 6) and np.isfinite(m).all()
        assert max(row["parity_path_error"]) == 0
        # Cost and correction averages share all four physical worlds. Raw
        # post-contact values remain included and contacts are counted separately.
        cost = m[:, :, 2].mean(axis=0)
        rows.append({"job": row["job"], "anticipate": row["anticipate"], "seed": row["optimizer_seed"],
            "mean_cost": cost.tolist(), "gain_percent_of_mean_cost": (100 * (cost[0]-cost)/cost[0]).tolist(),
            "contact_exposures": (m[:, :, 4] < 0).sum(axis=0).tolist(),
            "mean_rpcbf_correction": m[:, :, 3].mean(axis=0).tolist(),
            "mean_reference_position_shift": float(np.mean(row["reference_position_shift_max"])),
            "maximum_reference_position_shift": max(row["reference_position_shift_max"]),
            "route_std_before_after": [float(np.mean(row[k])) for k in ("route_std_before", "route_std_after")],
            "probe_std_before_after": [float(np.mean(row[k])) for k in ("probe_std_before", "probe_std_after")]})
    summary = {"complete": True, "arm_order": ["old", "controller_and_filter", "reference", "both"],
        "rows": rows, "interpretation": "Same first terminal state/clarity and physical wind per comparison; one update held fixed for eight seconds. All production endpoint state parity errors are zero. Raw cost includes contacts; descriptive posterior-model mechanism diagnostic, not a mission or optimal probing demonstration."}
    (folder/"mechanism_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for col, job in enumerate(("original", "reversed", "low_value")):
        rs = [r for r in rows if r["job"] == job]
        x = np.arange(len(rs))
        for arm, (name, color) in enumerate((("Controller + filter", "#126f87"), ("Reference only", "#b55b2b"), ("Both", "#765494")), start=1):
            heights = [r["gain_percent_of_mean_cost"][arm] for r in rs]
            bars = axes[0, col].bar(x + (arm-2)*.25, heights, width=.25, color=color, label=name)
            for bar, r in zip(bars, rs, strict=True):
                if r["contact_exposures"][0] or r["contact_exposures"][arm]:
                    bar.set_hatch("///")
        axes[0, col].axhline(0, color="0.4", lw=1)
        axes[0, col].set_title(job.replace("_", " ").title())
        axes[0, col].set_ylabel("Raw cost improvement from old belief (%)")
        axes[1, col].plot(x, [r["route_std_before_after"][0]-r["route_std_before_after"][1] for r in rs], "o-", color="#126f87", label="Route reduction")
        axes[1, col].plot(x, [r["probe_std_before_after"][0]-r["probe_std_before_after"][1] for r in rs], "s-", color="#b55b2b", label="Central probe reduction")
        axes[1, col].axhline(0, color="0.4", lw=1)
        axes[1, col].set_ylabel("Change in mean marginal wind std")
        for ax in axes[:, col]:
            ax.set_xticks(x, ["Ant. 01", "Ant. 02", "Frozen 01", "Frozen 02"], fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("One earned POAM update • four new wind worlds • same first two seconds\nHatched bars include contact in an arm: raw cost is not a successful-monitoring score.", fontsize=12)
    fig.savefig(folder/"mechanism_summary.png", dpi=170)
    plt.close(fig)
    for row in rows:
        print(row["job"], row["anticipate"], row["seed"], "gains", row["gain_percent_of_mean_cost"], "contacts", row["contact_exposures"])


if __name__ == "__main__":
    main()
