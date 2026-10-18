"""Serial publication of the declared completed study, with forecast checkpoints."""

import json
import time
from pathlib import Path

from export_budgeted_demo import export

ROOT = Path(__file__).resolve().parents[1]


def main():
    protocol = json.loads((ROOT / "docs/covariance_study/persistent_protocol_01.json").read_text())
    assert json.loads((ROOT / protocol["status"]).read_text())["state"] == "complete"
    status_path = ROOT / "results/covariance_exports_01_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"jobs": []}
    status["state"] = "running"
    for job in protocol["jobs"]:
        for method in job["methods"]:
            path = ROOT / job["output"] / (method + ".npz")
            key = str(path.relative_to(ROOT))
            previous = next((r for r in status["jobs"] if r["input"] == key), None)
            if previous and previous["state"] == "complete":
                continue
            row = previous or {"input": key}
            if previous is None:
                status["jobs"].append(row)
            row.update(state="running", started=time.time())
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            try:
                row["id"] = export(path)
                row.update(state="complete", ended=time.time())
            except Exception as exc:
                row.update(state="failed", error=repr(exc))
                status["state"] = "failed"
                status_path.write_text(json.dumps(status, indent=2) + "\n")
                raise
            status_path.write_text(json.dumps(status, indent=2) + "\n")
    status["state"] = "complete"
    status_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
