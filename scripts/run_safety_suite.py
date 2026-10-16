"""Declare and execute a paired, independent-seed wind-in-RPCBF stress test."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/wind_safety"
REPORT = ROOT / "docs/wind_safety_validation"
CASES = [
    ("field_vortices", 300, range(610, 618)),
    ("field_shear", 301, range(620, 628)),
    ("field_mixed", 302, range(630, 638)),
]


def atomic(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def protocol():
    REPORT.mkdir(parents=True, exist_ok=True)
    script = ROOT / "scripts/wind_safety_ablation.py"
    files = list((ROOT / "src").rglob("*.py")) + [script, Path(__file__)]
    manifest = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)
    }
    policies = {
        case: hashlib.sha256(
            (ROOT / f"demo/data/{case}_proposed_{seed}.json").read_bytes()
        ).hexdigest()
        for case, seed, _ in CASES
    }
    path = REPORT / "protocol.json"
    if path.exists():
        declared = json.loads(path.read_text())
        if declared["source_sha256"] != manifest or declared["policy_sha256"] != policies:
            raise ValueError(
                "Safety study source/policy changed; preserve the old protocol and results"
            )
        return declared
    declared = {
        "declared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Isolate wind in the genuine spline RPCBF while retaining POAM-based nominal compensation and actual online learning in both arms.",
        "type": "Paired fixed-policy safety stress test, not independent MBD replanning.",
        "cases": [
            {"scenario": case, "policy_seed": seed, "held_out_seeds": list(seeds)}
            for case, seed, seeds in CASES
        ],
        "source_sha256": manifest,
        "policy_sha256": policies,
        "pilot_disclosure": "Mixed-flow policy/physical seed 302 was examined before this declaration. The wind-ignored arm contacted an obstacle; all 24 new physical seeds are declared before execution. Original seeds 300/301/302 are positive-control replays, excluded from held-out inference.",
        "invariants": "Same starting robot/clarity, physical wind, waypoint schedule and noise stream per pair. Both arms adapt their own full POAM on their own executed measurements. Only RPCBF assumes zero wind in the ablation. All core spline/QP routines are retained.",
        "primary_outcome": "Any physical collision with clearance <0, checked using 32 RK4 substeps per control; finite sampled RPCBF is not a formal guarantee.",
        "secondary_outcomes": [
            "minimum clearance",
            "time to first collision",
            "corrections",
            "soft slack",
            "whole-field clarity cost",
            "wind RMSE",
        ],
        "inference": "24 independent physical seeds, eight per layout. Paired exact binomial test of discordant collision outcomes (McNemar), two-sided. Report all cases; source-seed positive controls excluded. No generality claim outside these fixed policy schedules and wind distributions.",
        "integration_check": "Compare RK4 step endpoints against 32 smaller steps, and retain discrepancy. Near-grazing collisions must exceed integration error before interpretation.",
    }
    atomic(path, declared)
    return declared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args()
    declared = protocol()
    if args.initialize_only:
        print(REPORT / "protocol.json")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "scenario": case,
            "policy_seed": original,
            "seed": seed,
            "split": "positive-control" if seed == original else "held-out",
            "status": "queued",
        }
        for case, original, seeds in CASES
        for seed in [original, *seeds]
    ]
    for job in jobs:
        path = OUT / f"{job['scenario']}_{job['seed']}.json"
        if path.exists():
            data = json.loads(path.read_text())
            expected = declared["source_sha256"]["scripts/wind_safety_ablation.py"]
            if data.get("ablation_script_sha256") == expected:
                job["status"] = "complete"
                continue
        job["status"] = "running"
        atomic(OUT / "status.json", {"pid": os.getpid(), "updated_at": time.time(), "jobs": jobs})
        recording = ROOT / f"demo/data/{job['scenario']}_proposed_{job['policy_seed']}.json"
        command = [
            sys.executable,
            str(ROOT / "scripts/wind_safety_ablation.py"),
            str(recording),
            "--seed",
            str(job["seed"]),
        ]
        with (OUT / f"{job['scenario']}_{job['seed']}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        job["status"] = "complete" if result.returncode == 0 and path.exists() else "failed"
        print(job["status"], job["scenario"], job["seed"], flush=True)
    atomic(OUT / "status.json", {"pid": os.getpid(), "updated_at": time.time(), "jobs": jobs})
    if any(j["status"] == "failed" for j in jobs):
        raise SystemExit("Failures retained; inspect per-case logs")


if __name__ == "__main__":
    main()
