"""Report every declared persistent covariance comparison, including contacts."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "proposed": "Realization anticipation",
    "current_belief": "Wind-aware frozen",
    "covariance_anticipated": "Covariance anticipation",
    "mean_frozen": "Mean-dynamics frozen",
    "wind_blind_matched": "Wind-blind matched",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("protocol", type=Path)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    status = json.loads((ROOT / protocol["status"]).read_text())
    if status["state"] != "complete":
        raise RuntimeError("Do not silently omit incomplete declared missions")
    out = args.protocol.parent
    rows = []
    paths = {}
    for job in protocol["jobs"]:
        for method in job["methods"]:
            path = ROOT / job["output"] / f"{method}.npz"
            doc = json.loads(path.with_suffix(".json").read_text())
            audit = json.loads(path.with_suffix(".audit.json").read_text())
            assert audit["complete"] and abs(audit["duration_s"] - 120) < 1e-8
            assert audit["recorded_cost_disagreement"] < 2e-5
            seed = doc["metadata"]["seed"]
            paths[seed, method] = path
            raw = np.load(path, allow_pickle=False)
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "cost": audit["integrated_whole_grid_deficit"],
                    "contact_intervals": audit["contact_intervals"],
                    "clearance": audit["minimum_dense_clearance"],
                    "transfers": len(audit["transfers"]),
                    "directional_routes": audit["directional_routes"],
                    "backup_periods": audit["backup_periods"],
                    "cycle_median_s": audit["steady_cycle_s"]["median"],
                    "cycle_p95_s": audit["steady_cycle_s"]["p95"],
                    "cycle_2s_miss_fraction": audit["cycle_2s_miss_fraction"],
                    "startup_cycles_s": audit["startup_cycle_s"],
                    "mean_control_correction": float(raw["correction"].mean()),
                    "max_qp_slack": float(raw["slack"].max()),
                    "passage_rmse_initial_final": audit["passage_rmse_initial_final"],
                    "passage_std_initial_final": audit["passage_std_initial_final"],
                }
            )
    comparisons = []
    for seed in sorted({r["seed"] for r in rows}):
        by = {r["method"]: r for r in rows if r["seed"] == seed}
        for a, b in [
            ("covariance_anticipated", "mean_frozen"),
            ("proposed", "current_belief"),
            ("covariance_anticipated", "proposed"),
            ("proposed", "wind_blind_matched"),
            ("current_belief", "wind_blind_matched"),
            ("covariance_anticipated", "wind_blind_matched"),
        ]:
            if a not in by or b not in by:
                continue
            comparisons.append(
                {
                    "seed": seed,
                    "method": a,
                    "comparator": b,
                    "cost_reduction_percent": 100 * (by[b]["cost"] - by[a]["cost"]) / by[b]["cost"],
                    "both_contact_free": by[a]["contact_intervals"]
                    == by[b]["contact_intervals"]
                    == 0,
                }
            )
    summary = {
        "scope": "Three development seeds with fixed original physical wind. No held-out method-selection claim, no confidence interval or universal safety assertion. Post-contact cost is raw cost, not successful monitoring.",
        "rows": rows,
        "comparisons": comparisons,
    }
    (out / "mission_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    methods = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for i, m in enumerate(methods):
        selected = [r for r in rows if r["method"] == m]
        for j, row in enumerate(selected):
            x = i + (j - 1) * 0.13
            marker = "x" if row["contact_intervals"] else "o"
            axes[0].scatter(x, row["cost"], marker=marker, color=f"C{j}")
            axes[1].scatter(x, row["cycle_median_s"], marker="o", color=f"C{j}")
            axes[1].plot([x, x], [row["cycle_median_s"], row["cycle_p95_s"]], color=f"C{j}")
    for ax in axes:
        ax.set_xticks(range(5), [LABELS[m].replace(" ", "\n", 1) for m in methods], fontsize=9)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set(
        title="120s monitoring cost · all declared seeds",
        ylabel="Integrated whole-grid clarity deficit",
    )
    axes[1].set(
        title="Full cycle latency · median to p95", ylabel="Seconds (startup reported separately)"
    )
    axes[1].axhline(2, ls="--", color="black", lw=1, label="2s soft budget")
    axes[1].legend()
    fig.savefig(out / "mission_comparison.png", dpi=160)
    plt.close(fig)
    seed = protocol["seeds"][0]
    fig, axes = plt.subplots(5, 2, figsize=(11, 14), layout="constrained")
    for i, m in enumerate(methods):
        path = paths[seed, m]
        d = np.load(path, allow_pickle=False)
        meta = json.loads(path.with_suffix(".json").read_text())["metadata"]
        ax, aq = axes[i]
        ax.plot(d["x"][:, 0], d["x"][:, 1], lw=0.8, color=f"C{i}")
        for ob in d["obstacles"]:
            ax.add_patch(plt.Circle(ob[:2], ob[2], color="#677980"))
        spec = meta["task_spec"]
        t = np.arange(len(d["q"])) * meta["execution_dt"]
        for region, color in [("a", "#287f8d"), ("b", "#a25b90")]:
            center = np.array(spec["region_" + region])
            size = np.array(spec["region_size"])
            mask = np.all(np.abs(d["cells"] - center) <= size / 2, axis=1)
            aq.plot(t, d["q"][:, mask].mean(axis=1), color=color, lw=0.9, label=region.upper())
            ax.add_patch(plt.Rectangle(center - size / 2, *size, fill=False, edgecolor=color))
        aq.set(ylim=(0, 1), xlim=(0, 120), ylabel="Regional mean clarity", xlabel="Mission seconds")
        aq.legend(fontsize=8)
        aq.grid(alpha=0.2)
        ax.set(title=LABELS[m], xlim=d["bounds"][:, 0], ylim=d["bounds"][:, 1], aspect="equal")
    fig.suptitle(f"Original physical wind · development seed {seed} · no forced alternation")
    fig.savefig(out / "persistent_execution.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
