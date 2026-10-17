"""Run declared follow-ups sequentially; no scheduler, retries, or partial overwrites."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "results/wind_route_followup_status.json"


def main():
    assert (
        json.loads((ROOT / "results/wind_route_study_status.json").read_text())["state"]
        == "complete"
    )
    fields = [
        "scripts/run_budgeted_mission.py",
        "scripts/export_budgeted_demo.py",
        "scripts/diagnose_observation_latency.py",
        "scripts/diagnose_wind_route_choices.py",
        "configs/two_region_budgeted.yaml",
        "configs/two_region_design.yaml",
    ]
    files = [ROOT / name for name in fields] + list(ROOT.glob("src/**/*.py"))
    frozen = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT / "docs/wind_route_study/followup_source_manifest.json").open("x").write(
        json.dumps(frozen, indent=2) + "\n"
    )
    common = json.loads((ROOT / "docs/wind_route_study/protocol_01.json").read_text())[
        "common_arguments"
    ]
    common = [("routes-survey" if v == "outer-survey" else v) for v in common]
    paths = [
        f"results/budgeted_missions/offroute_surveyed_wind_01/{m}.npz"
        for m in ["proposed", "current_belief"]
    ]
    jobs = [
        ("observation_latency_02", ["scripts/diagnose_observation_latency.py"]),
        (
            "offroute_surveyed_wind_01",
            [
                "scripts/run_budgeted_mission.py",
                *common,
                "--wind-case",
                "offroute",
                "--methods",
                "proposed",
                "current_belief",
                "--output",
                "results/budgeted_missions/offroute_surveyed_wind_01",
            ],
        ),
        ("offroute_surveyed_audit_01", ["scripts/audit_budgeted_mission.py", *paths]),
        ("offroute_surveyed_export_01", ["scripts/export_budgeted_demo.py", *paths]),
        ("wind_route_report_followup_01", ["scripts/report_wind_route_study.py"]),
    ]
    for case in ["reversed", "offroute"]:
        jobs.append(
            (
                case + "_causal_01",
                [
                    "scripts/diagnose_wind_route_choices.py",
                    f"results/budgeted_missions/{case}_wind_01/proposed.npz",
                    "--output",
                    f"docs/wind_route_study/{case}_causal_01",
                ],
            )
        )
    status = {"state": "running", "phases": [], "jobs": [name for name, _ in jobs]}
    for name, args in jobs:
        for file, digest in frozen.items():
            assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == digest, file
        row = {"label": name, "state": "running", "started_at": time.time()}
        status["phases"].append(row)
        STATUS.write_text(json.dumps(status, indent=2) + "\n")
        with (ROOT / "results" / f"{name}.log").open("a") as log:
            code = subprocess.call(
                [sys.executable, *args],
                cwd=ROOT,
                env=os.environ,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        row.update(
            ended_at=time.time(), returncode=code, state="complete" if code == 0 else "failed"
        )
        # Preserve a failed diagnostic without blocking the independent declared mission.
        STATUS.write_text(json.dumps(status, indent=2) + "\n")
    status["state"] = (
        "complete"
        if all(r["returncode"] == 0 for r in status["phases"])
        else "complete_with_failures"
    )
    STATUS.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
