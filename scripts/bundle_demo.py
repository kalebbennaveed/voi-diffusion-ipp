"""Bundle a recorded replay with an explicit scientific label, without changing its data."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("replay", type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--group", required=True)
    args = p.parse_args()
    data = json.loads(args.replay.read_text())
    cfg, metrics = data["config"], data["metrics"]
    identity = hashlib.sha256(args.replay.read_bytes()).hexdigest()[:16]
    out = ROOT / "demo/data"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{identity}.json"
    shutil.copy2(args.replay, dest)
    index = out / "catalog.json"
    records = json.loads(index.read_text()) if index.exists() else []
    record = {
        "id": identity,
        "label": args.label,
        "method": metrics["method"],
        "seed": metrics["seed"],
        "variant": "multistage" if cfg["future_updates"] else "two_stage",
        "group": args.group,
        "file": str(dest.relative_to(ROOT)),
        "live": False,
    }
    records = [r for r in records if r["id"] != identity]
    records.append(record)
    index.write_text(json.dumps(records, indent=2) + "\n")
    print(identity)


if __name__ == "__main__":
    main()
