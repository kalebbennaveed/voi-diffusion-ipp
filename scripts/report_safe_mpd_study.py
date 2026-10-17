"""Summarize every declared outcome, including contacts and stalled missions."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("protocol", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in protocol["jobs"]:
        for method in job["methods"]:
            path = ROOT / job["output"] / method
            if not path.with_suffix(".audit.json").exists():
                rows.append({"name": job["name"], "method": method, "complete": False})
                continue
            audit = json.loads(path.with_suffix(".audit.json").read_text())
            raw = json.loads(path.with_suffix(".json").read_text())
            meta, periods = raw["metadata"], raw["periods"]
            execution_ticks = round(
                meta["config"]["stage_steps"] * meta["config"]["dt"] / meta["execution_dt"]
            )
            row = {
                "name": job["name"],
                "method": method,
                "complete": raw["complete"],
                "planner_form": meta.get("planner_form", "compact"),
                "safety_filter": meta["config"].get("safety_filter", "rpcbf"),
                "seed": meta["seed"],
                "wind_case": meta["wind_case"],
                "audit": audit,
                "shield_active_fraction": float(
                    np.mean([v.get("shield_active_fraction", 0) for v in periods])
                ),
                "execution_tick_ms": float(
                    np.median([1000 * v["execution_s"] / execution_ticks for v in periods[2:]])
                ),
                "steady_components_s": {
                    k: float(np.median([v[k] for v in periods[2:]]))
                    for k in ["cache_s", "plan_s", "verify_s", "poam_update_s", "snapshot_s"]
                },
            }
            rows.append(row)
    (args.output / "results.json").write_text(
        json.dumps({"protocol": protocol, "rows": rows}, indent=2) + "\n"
    )
    ready = [r for r in rows if r["complete"]]
    fig, axes = plt.subplots(
        1, 3, figsize=(15, max(3, 0.45 * len(ready) + 1)), layout="constrained"
    )
    labels = [r["name"].replace("safe_mpd_mission_01_", "") for r in ready]
    fields = [
        ("integrated_whole_grid_deficit", "Integrated clarity deficit (lower is better)"),
        ("steady_cycle_s", "Median full planning cycle (s)"),
        ("minimum_dense_clearance", "Minimum physical clearance (m)"),
    ]
    for ax, (key, title) in zip(axes, fields):
        values = [
            r["audit"][key]["median"] if key == "steady_cycle_s" else r["audit"][key] for r in ready
        ]
        colors = ["#ae4a40" if r["audit"]["contact_intervals"] else "#366d83" for r in ready]
        ax.barh(labels, values, color=colors)
        ax.set_xlabel(title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        if key == "steady_cycle_s":
            ax.axvline(2, color="black", ls="--", lw=1)
        if key == "minimum_dense_clearance":
            ax.axvline(0, color="black", lw=1)
    fig.suptitle(
        "Safe-MPD development comparison — 120s missions, 10s forecasts\nRed marks physical contact; post-contact cost is not successful monitoring performance"
    )
    fig.savefig(args.output / "comparison.png", dpi=160)
    plt.close(fig)
    # Full physical trajectories, no smooth replacement paths.
    fig, axes = plt.subplots(
        1, len(ready), figsize=(4 * len(ready), 4), layout="constrained", squeeze=False
    )
    for ax, row in zip(axes[0], ready):
        job = next(j for j in protocol["jobs"] if j["name"] == row["name"])
        d = np.load(ROOT / job["output"] / (row["method"] + ".npz"), allow_pickle=False)
        ax.plot(d["x"][:, 0], d["x"][:, 1], lw=0.8, color="#286e89")
        ax.scatter(*d["x"][0, :2], color="green", s=25)
        for ox, oy, r in d["obstacles"]:
            ax.add_patch(plt.Circle((ox, oy), r, color="#777777"))
        ax.set(
            xlim=d["bounds"][:, 0],
            ylim=d["bounds"][:, 1],
            aspect="equal",
            title=row["name"].replace("safe_mpd_mission_01_", ""),
        )
    fig.savefig(args.output / "trajectories.png", dpi=140)
    plt.close(fig)
    for r in ready:
        a = r["audit"]
        print(
            r["name"],
            json.dumps(
                {
                    "cost": a["integrated_whole_grid_deficit"],
                    "median_s": a["steady_cycle_s"]["median"],
                    "p95_s": a["steady_cycle_s"]["p95"],
                    "contacts": a["contact_intervals"],
                    "transfers": len(a["transfers"]),
                    "shield_fraction": r["shield_active_fraction"],
                    "execution_tick_ms": r["execution_tick_ms"],
                }
            ),
        )


if __name__ == "__main__":
    main()
