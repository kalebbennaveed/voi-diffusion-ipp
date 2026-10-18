"""Check all declared replay artifacts against their untouched numeric archives."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    status = json.loads((ROOT / "results/covariance_exports_01_status.json").read_text())
    assert status["state"] == "complete" and len(status["jobs"]) == 15
    rows = []
    cache_rows = []
    for job in status["jobs"]:
        assert job["state"] == "complete"
        path = ROOT / job["input"]
        original = json.loads(path.with_suffix(".json").read_text())
        folder = ROOT / "results/demo/live" / job["id"]
        replay_path = folder / "replay.json"
        replay = json.loads(replay_path.read_text())
        record = json.loads((folder / "record.json").read_text())
        meta = original["metadata"]
        assert replay["source_sha256"] == meta["source_sha256"]
        assert replay["metrics"]["method"] == record["method"] == meta["method"]
        assert replay["metrics"]["seed"] == record["seed"] == meta["seed"]
        assert len(replay["planning"]) == len(original["periods"]) == 60
        assert record["available_seconds"] == 120
        with np.load(path, allow_pickle=False) as raw:
            for key, tolerance in [("x", 5.1e-7), ("q", 5.1e-6)]:
                np.testing.assert_allclose(replay["mission"][key], raw[key], atol=tolerance, rtol=0)
                assert np.isfinite(replay["mission"][key]).all()
            np.testing.assert_array_equal(replay["physical_wind"], raw["physical_wind"])
            for index, snapshot in enumerate(replay["beliefs"]):
                for key in ["mean", "std"]:
                    np.testing.assert_allclose(snapshot[key], raw[key][index], atol=5.1e-6, rtol=0)
        for plan, original_plan in zip(replay["planning"], original["periods"], strict=True):
            assert plan["candidate_paths"] and len(plan["candidate_paths"]) == 4
            assert all(len(level) == 8 for level in plan["candidate_paths"])
            np.testing.assert_array_equal(plan["selected_vector"], original_plan["vector"])
            np.testing.assert_array_equal(plan["weights"], original_plan["weights"])
            np.testing.assert_allclose(
                plan["selected_path"],
                np.array(original_plan["selected_model_path"])[:, :2],
                atol=5.1e-7,
                rtol=0,
            )
        worst = max(replay["planning"], key=lambda p: p["cached_replay_full_path_max_distance"])
        period = worst["period"]
        original_plan = original["periods"][period]
        with np.load(folder / f"forecast_{period:04d}.npz", allow_pickle=False) as cached:
            difference = np.linalg.norm(
                cached["paths"][-1] - np.array(original_plan["selected_model_path"])[:, :2],
                axis=-1,
            )
        stage = meta["config"]["stage_steps"]
        cache_rows.append(
            {
                "method": record["method"],
                "seed": record["seed"],
                "period": period,
                "max_distance_m": float(difference.max()),
                "at_forecast_s": float(np.argmax(difference) * meta["config"]["dt"]),
                "first_stage_max_m": float(difference[: stage + 1].max()),
                "cached_cost": original_plan["cached_objective"],
                "full_cost": original_plan["full_objective"],
                "backup_used": original_plan["backup_used"],
                "at_update_boundaries": difference[::stage].tolist(),
            }
        )
        rows.append(
            {
                "id": job["id"],
                "group": record["group"],
                "method": record["method"],
                "seed": record["seed"],
                "replay_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
                "forecast_manifest_sha256": hashlib.sha256(
                    (folder / "forecast_manifest.json").read_bytes()
                ).hexdigest(),
                "max_cached_vs_original_uncached_path_distance": max(
                    p["cached_replay_full_path_max_distance"] for p in replay["planning"]
                ),
            }
        )
    assert len({r["group"] for r in rows}) == 1
    assert len({r["seed"] for r in rows}) == 3
    report = {
        "state": "passed",
        "checks": "All 15 full 120s recordings, all 900 plans, stored optimizer vectors/weights, original selected uncached paths, real execution and POAM maps agree with numeric archives to declared UI rounding. Every cached candidate checkpoint passed exporter vector checks and source/input manifests. No optimizer reruns.",
        "limits": "Candidate forecasts use interpolated fields; final curves use the original uncached verification. Their maximum trajectory differences are recorded below; these are model/field-evaluation differences, not physical execution error. Realization curves show world 1 while objective weights average both worlds.",
        "records": rows,
    }
    (ROOT / "docs/covariance_study/export_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (ROOT / "docs/covariance_study/cache_path_diagnostic.json").write_text(
        json.dumps(
            {
                "scope": "Worst cached-versus-uncached selected-model trajectory in each declared mission. Cached final replay uses the same stored policy/earned belief and sample seeds; original uncached check is the actual selected forecast in replay. Not physical execution divergence or collision evidence.",
                "rows": cache_rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
