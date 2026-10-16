"""Capture source/configuration before evaluation; never replace an existing study."""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from voi_diffusion_ipp.config import config_dict, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/recourse_validation"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.relative_to(root)
    if out.exists() and any(out.iterdir()):
        raise ValueError("The study directory must be empty before freezing its protocol")
    cfg = load_config(root / "configs/recourse_validation.yaml")
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "configuration": config_dict(cfg),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src").rglob("*.py"))
        },
        "primary_metric": "Paired actual integrated clarity deficit: comparator minus proposed",
        "secondary_metrics": [
            "collisions",
            "QP infeasibility",
            "optimizer fallbacks",
            "ROI access",
            "initial cost and subsequent recovery",
        ],
        "development_seeds": [20, 21, 22],
        "validation_seeds": list(cfg.seeds),
        "selection_policy": (
            "Run and retain all pairs. No configuration change or seed removal after observing these results."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Frozen protocol: {out / 'protocol.json'}")


if __name__ == "__main__":
    main()
