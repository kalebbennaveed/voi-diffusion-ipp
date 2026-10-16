"""Preserve compact, auditable evidence from the completed local experiment suite.

Raw trajectories remain in results/. This deliberately excludes exploratory and
superseded attempts, but includes every fallback and collision in the final suite.
"""

import csv
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "docs/validation"
    out.mkdir(parents=True, exist_ok=True)
    folders = {
        "evaluation": 63,
        "additional": 27,
        **{
            f"ablations/{v}": 3
            for v in (
                "no_measurements",
                "shared_future",
                "independent_cells",
                "terminal",
                "short_probe",
                "fast_spline",
            )
        },
        "extended": 2,
        "extended_no_measurements": 1,
        "long_mission": 6,
        "long_no_measurements": 3,
    }
    rows, hashes = [], {}
    for folder, expected in folders.items():
        found = []
        for path in sorted((root / "results" / folder).glob("*.json")):
            record = json.loads(path.read_text())
            if not isinstance(record, dict) or "integrated_deficit" not in record:
                continue
            trajectory = path.with_suffix(".npz")
            with np.load(trajectory) as arrays:
                n = record["config"]["stage_steps"]
                dt = record["config"]["dt"]
                deficit = np.maximum(arrays["targets"] - arrays["q"], 0).mean(axis=1)
                steps = dt * (deficit[1:] + deficit[:-1]) * 0.5
                record["period_costs"] = steps.reshape((-1, n)).sum(axis=1).tolist()
                np.testing.assert_allclose(
                    sum(record["period_costs"]), record["integrated_deficit"], atol=2e-5
                )
                record["executed_branches"] = arrays["branches"].tolist()
                record["mean_invalid_proposal_fraction"] = float(arrays["traces"][:, :, 3].mean())
                if "optimizer_failed" in arrays:
                    failed = arrays["optimizer_failed"].tolist()
                else:
                    # The first runner aborted all-invalid missions. Only its
                    # successful outputs remain; failed ones were rerun later.
                    assert np.all(np.isfinite(arrays["traces"][:, -1, 0]))
                    failed = [False] * record["config"]["mission_horizons"]
                record["optimizer_failure_count"] = sum(failed)
                for index, diagnostic in enumerate(record["matched_continuations"]):
                    diagnostic["optimized_policy_valid"] = not failed[index]
            record["suite"] = folder
            found.append(record)
            for file in (path, trajectory):
                hashes[str(file.relative_to(root))] = hashlib.sha256(file.read_bytes()).hexdigest()
        if len(found) != expected:
            raise ValueError(f"Incomplete {folder}: {len(found)} / {expected}")
        rows.extend(found)
    (out / "missions.json").write_text(json.dumps(rows, indent=2) + "\n")
    columns = [
        "suite",
        "scenario",
        "method",
        "seed",
        "integrated_deficit",
        "final_deficit",
        "roi_a_access",
        "roi_a_time",
        "probe_seconds",
        "collision",
        "minimum_clearance",
        "qp_infeasible_fraction",
        "optimizer_failure_count",
        "mean_correction",
        "max_qp_slack",
        "wind_rmse",
        "wind_95_coverage",
        "wind_mean_std",
        "compile_seconds",
        "mean_plan_seconds",
        "objective_evaluations_per_plan",
        "peak_process_bytes",
        "mean_invalid_proposal_fraction",
    ]
    with (out / "missions.csv").open("w") as file:
        writer = csv.DictWriter(file, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        groups.setdefault((row["suite"], row["scenario"], row["method"]), []).append(row)
    aggregate, table = (
        [],
        [
            "# Archived measured results",
            "",
            f"{len(rows)} missions. Costs are actual integrated clarity deficits; lower is better. ± is sample SD, not a confidence interval.",
            "",
            "| Suite | Scenario | Method | n | Cost mean ± SD | ROI access | Collisions | QP infeasible | Optimizer fallbacks | Mean plan s |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ],
    )
    for (suite, scenario, method), rs in sorted(groups.items()):
        costs = [r["integrated_deficit"] for r in rs]
        r = {
            "suite": suite,
            "scenario": scenario,
            "method": method,
            "n": len(rs),
            "cost_mean": float(np.mean(costs)),
            "cost_sd": float(np.std(costs, ddof=1)) if len(rs) > 1 else None,
            "access": sum(r["roi_a_access"] for r in rs),
            "collisions": sum(r["collision"] for r in rs),
            "qp_infeasible": float(np.mean([r["qp_infeasible_fraction"] for r in rs])),
            "fallbacks": sum(r["optimizer_failure_count"] for r in rs),
            "plan_seconds": float(np.mean([r["mean_plan_seconds"] for r in rs])),
        }
        aggregate.append(r)
        sd = f" ± {r['cost_sd']:.4f}" if r["cost_sd"] is not None else ""
        table.append(
            f"| {suite} | {scenario} | {method} | {r['n']} | {r['cost_mean']:.4f}{sd} | {r['access']}/{r['n']} | {r['collisions']}/{r['n']} | {r['qp_infeasible']:.2%} | {r['fallbacks']} | {r['plan_seconds']:.2f} |"
        )
    (out / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    (out / "summary.md").write_text("\n".join(table) + "\n")
    for name in (
        "environment_cpu.json",
        "backend_benchmark.json",
        "learning_mechanism.json",
        "optimizer_benchmark.json",
        "rpcbf_variants.json",
        "environment_metal_unsandboxed.log",
        "environment_metaljax.log",
        "tests_cpu.log",
        "tests_routing.log",
        "smoke_test.log",
    ):
        shutil.copy2(root / "results" / name, out / name)
    shutil.copy2(root / "results/poam_smoke/corridor_proposed_0.json", out / "poam_smoke.json")
    source_paths = [root / "pyproject.toml", root / "uv.lock", root / "uv.toml"]
    for folder in ("src", "tests", "scripts", "configs"):
        source_paths.extend(
            p for p in (root / folder).rglob("*") if p.suffix in (".py", ".yaml", ".sh", ".txt")
        )
    source_hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source_paths)
    }
    manifest = {
        "captured_utc": datetime.now(UTC).isoformat(),
        "base_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "note": "Source hashes capture the reviewed tree at archive time, including uncommitted changes. Successful early runs predate diagnostic-only fixes; see experimental_findings.md. Timing includes concurrent host load.",
        "mission_count": len(rows),
        "source_sha256": source_hashes,
        "raw_result_sha256": hashes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Archived {len(rows)} missions and {len(groups)} aggregate groups to {out}")


if __name__ == "__main__":
    main()
