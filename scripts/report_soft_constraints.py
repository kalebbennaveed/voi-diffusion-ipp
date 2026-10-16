"""Reinterpret existing validation with allowed slack; do not rerun or change the method."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for arm in ("frozen_belief", "two_stage", "repeated_updates"):
        for file in sorted((ROOT / "results/multistage_validation" / arm).glob("*.npz")):
            metrics = json.loads(file.with_suffix(".json").read_text())
            with np.load(file) as a:
                slack = a["slack"].astype(float)
                dt = metrics["config"]["dt"]
                rows.append(
                    {
                        "arm": arm,
                        "seed": metrics["seed"],
                        "task_cost": metrics["integrated_deficit"],
                        "peak_slack": float(slack.max()),
                        "integrated_slack": float(np.maximum(slack, 0).sum() * dt),
                        "slack_steps": int((slack > 0).sum()),
                        "slack_seconds": float((slack > 0).sum() * dt),
                        "minimum_clearance": metrics["minimum_clearance"],
                        "collision": metrics["collision"],
                    }
                )
    assert len(rows) == 30
    aggregate = []
    for arm in ("frozen_belief", "two_stage", "repeated_updates"):
        rs = [r for r in rows if r["arm"] == arm]
        aggregate.append(
            {
                "arm": arm,
                "mean_task_cost": float(np.mean([r["task_cost"] for r in rs])),
                "maximum_slack": max(r["peak_slack"] for r in rs),
                "total_integrated_slack": sum(r["integrated_slack"] for r in rs),
                "total_slack_seconds": sum(r["slack_seconds"] for r in rs),
                "total_slack_steps": sum(r["slack_steps"] for r in rs),
                "collisions": sum(r["collision"] for r in rs),
            }
        )
    result = {
        "interpretation": "Barrier slack is allowed. A positive slack is not a physical collision or an optimizer failure. All archived numerical outputs are unchanged.",
        "slack_units": "common RPCBF derivative-constraint units; not geometric distance",
        "missions": rows,
        "aggregate": aggregate,
    }
    output = ROOT / "docs/multistage_validation/soft_constraint_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
