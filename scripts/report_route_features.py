"""Verify and summarize the complete route-feature ablation, including failures."""

import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    folder = Path("docs/mbd_reliability/route_features_01")
    rows, comparisons, evidence, baseline_errors = [], [], {}, []
    for job in ("original", "reversed", "low_value"):
        file = folder / (job + ".json")
        data = json.loads(file.read_text())
        assert data["complete"] and len(data["optimization"]) == 8 and len(data["evaluation"]) == 4
        evidence[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        old = json.loads(Path(f"docs/mbd_reliability/search_01/{job}.json").read_text())
        costs, gaps = [], []
        for item in data["evaluation"]:
            path = folder / f'{job}_test_{item["bank"]}.npz'
            with np.load(path) as arrays:
                np.testing.assert_array_equal(arrays["cost"], item["cost"])
                np.testing.assert_array_equal(arrays["clearance"], item["clearance"])
                costs.append(arrays["cost"])
                gaps.append(arrays["clearance"])
            evidence[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        costs, gaps = np.concatenate(costs, axis=1), np.concatenate(gaps, axis=1)
        assert costs.shape == gaps.shape == (8, 16) and np.isfinite(costs).all() and np.isfinite(gaps).all()
        for i, row in enumerate(data["optimization"]):
            if row["features"] == "probe":
                original = next(r for r in old["optimization"] if r["variant"] == "4x8_w2" and r["anticipate"] == row["anticipate"] and r["seed"] == row["seed"])
                baseline_errors.append({"job": job, "anticipate": row["anticipate"], "seed": row["seed"],
                    "vector_max_error": float(np.max(np.abs(np.asarray(row["vector"])-original["vector"]))),
                    "cost_error": abs(row["train_cost"]-original["train_cost"])})
            rows.append({k: v for k, v in row.items() if k != "vector"} | {
                "job": job, "test_raw_mean_cost": float(costs[i].mean()), "test_contacts": int((gaps[i] < 0).sum()),
                "minimum_clearance": float(gaps[i].min())})
        for anticipate in (True, False):
            groups = {}
            for features in ("probe", "route_knots"):
                rr = [r for r in rows if r["job"] == job and r["anticipate"] == anticipate and r["features"] == features]
                groups[features] = {"train_cost": float(np.mean([r["train_cost"] for r in rr])),
                    "test_raw_mean_cost": float(np.mean([r["test_raw_mean_cost"] for r in rr])),
                    "optimizer_s": float(np.mean([r["optimizer_s"] for r in rr])),
                    "contact_exposures": sum(r["test_contacts"] for r in rr), "exposures": 32}
            comparisons.append({"job": job, "anticipate": anticipate, **groups,
                "route_feature_regret_percent": 100*(groups["route_knots"]["test_raw_mean_cost"]/groups["probe"]["test_raw_mean_cost"]-1)})
    result = {"complete": True, "rows": rows, "comparisons": comparisons, "baseline_reproduction": baseline_errors,
        "evidence_sha256": evidence, "claim": "Development feature ablation. Shared posterior test worlds; two optimizer seeds. Raw cost includes contacts. New fields never used for optimization or policy selection; no physical safety or mission-level superiority claim."}
    (folder.parent / "route_features_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    for r in comparisons:
        print(r["job"], r["anticipate"], "probe", r["probe"], "route", r["route_knots"], "regret%", r["route_feature_regret_percent"])
    print("Largest baseline vector/cost errors", max(r["vector_max_error"] for r in baseline_errors), max(r["cost_error"] for r in baseline_errors))


if __name__ == "__main__":
    main()
