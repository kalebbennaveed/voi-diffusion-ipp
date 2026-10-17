"""Sequential coordinator for the declared wind study; never overwrites partial runs."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/wind_route_study/protocol_01.json"
STATUS = ROOT / "results/wind_route_study_status.json"


def main():
    protocol = json.loads(PROTOCOL.read_text())
    status = {"protocol": str(PROTOCOL.relative_to(ROOT)), "phases": [], "state": "running"}

    def frozen():
        for name, digest in protocol["source_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name

    def run(label, args):
        frozen()
        row = {"label": label, "started_at": time.time(), "state": "running"}
        status["phases"].append(row)
        STATUS.write_text(json.dumps(status, indent=2) + "\n")
        with (ROOT / "results" / f"{label}.log").open("a") as log:
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
        if code:
            status["state"] = "failed"
        STATUS.write_text(json.dumps(status, indent=2) + "\n")
        if code:
            raise RuntimeError(label)

    for job in protocol["jobs"]:
        directory = ROOT / job["output"]
        paths = [directory / f"{m}.npz" for m in job["methods"]]
        complete = all(
            p.exists() and json.loads(p.with_suffix(".json").read_text())["complete"] for p in paths
        )
        if not complete:
            if directory.exists():
                raise FileExistsError(
                    f"Partial data retained; explicit recovery required: {directory}"
                )
            run(
                job["case"] + "_wind_01",
                [
                    "scripts/run_budgeted_mission.py",
                    *protocol["common_arguments"],
                    "--wind-case",
                    job["case"],
                    "--methods",
                    *job["methods"],
                    "--output",
                    job["output"],
                ],
            )
        run(
            job["case"] + "_audit_01",
            ["scripts/audit_budgeted_mission.py", *[str(p) for p in paths]],
        )
        for method, path in zip(job["methods"], paths, strict=True):
            record = (
                ROOT
                / "results/demo/live"
                / f"budgeted_{directory.name}_{method}_{job['seed']}"
                / "record.json"
            )
            if not record.exists():
                run(
                    job["case"] + "_" + method + "_export_01",
                    ["scripts/export_budgeted_demo.py", str(path)],
                )
    run("wind_route_report_final_01", ["scripts/report_wind_route_study.py"])
    status["state"] = "complete"
    STATUS.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
