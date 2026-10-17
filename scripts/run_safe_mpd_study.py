"""Run a declared comparison serially; fail closed on source drift or duplicates."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("protocol", type=Path)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    status_path = ROOT / protocol["status"]
    if status_path.exists():
        raise FileExistsError("Inspect existing study; do not duplicate it")
    status = {"state": "running", "jobs": [], "protocol": str(args.protocol)}
    env = os.environ | {"JAX_PLATFORMS": "cpu"}
    for job in protocol["jobs"]:
        for file, digest in protocol["frozen_sha256"].items():
            assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == digest, file
        row = {"name": job["name"], "state": "running", "started": time.time()}
        status["jobs"].append(row)
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print("START", job["name"], flush=True)
        try:
            with (ROOT / "results" / (job["name"] + ".log")).open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/run_budgeted_mission.py",
                        *job["arguments"],
                        "--output",
                        job["output"],
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                for method in job["methods"]:
                    subprocess.run(
                        [
                            sys.executable,
                            "scripts/audit_budgeted_mission.py",
                            str(ROOT / job["output"] / (method + ".npz")),
                        ],
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            row["state"] = "complete"
        except Exception as exc:
            row.update(state="failed", error=repr(exc))
            status["state"] = "failed"
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            raise
        row["ended"] = time.time()
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print("COMPLETE", job["name"], flush=True)
    status["state"] = "complete"
    status_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
