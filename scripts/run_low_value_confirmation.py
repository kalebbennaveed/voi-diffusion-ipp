"""Run the prospectively declared eight-seed low-value study, sequentially."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    protocol = json.loads(
        (ROOT / "docs/wind_route_study/low_value_confirmation_protocol_01.json").read_text()
    )
    status_path = ROOT / "results/low_value_confirmation_status_01.json"
    if status_path.exists():
        raise RuntimeError("Study already started; inspect checkpoints instead of duplicating it")
    assert (ROOT / "results/horizon_coordinator_01.log").read_text().rstrip().endswith("COMPLETE")
    state = {"state": "running", "protocol": protocol, "seeds": []}
    status_path.write_text(json.dumps(state, indent=2) + "\n")
    for job in protocol["jobs"]:
        for file, digest in protocol["source_sha256"].items():
            assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == digest, file
        item = {"seed": job["seed"], "state": "running", "started": time.time()}
        state["seeds"].append(item)
        status_path.write_text(json.dumps(state, indent=2) + "\n")
        with (ROOT / "results" / f"low_value_seed_{job['seed']}.log").open("x") as log:
            code = subprocess.call(
                [sys.executable, "scripts/run_budgeted_mission.py", *job["arguments"]],
                cwd=ROOT,
                env=os.environ,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if code == 0:
                files = [f"{job['output']}/{m}.npz" for m in ["proposed", "current_belief"]]
                code = subprocess.call(
                    [sys.executable, "scripts/audit_budgeted_mission.py", *files],
                    cwd=ROOT,
                    env=os.environ,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        item.update(state="complete" if code == 0 else "failed", returncode=code, ended=time.time())
        status_path.write_text(json.dumps(state, indent=2) + "\n")
        print(item, flush=True)
        if code:
            state["state"] = "failed"
            status_path.write_text(json.dumps(state, indent=2) + "\n")
            raise RuntimeError(f"Seed {job['seed']} failed; all partial data retained")
    subprocess.run(
        [sys.executable, "scripts/report_low_value_confirmation.py"], cwd=ROOT, check=True
    )
    state["state"] = "complete"
    status_path.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
