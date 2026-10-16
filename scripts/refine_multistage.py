"""Frozen fresh-seed comparison of repeated belief updates and both controls."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from voi_diffusion_ipp.config import config_dict, load_config
from voi_diffusion_ipp.experiments.run import mission

ROOT = Path(__file__).resolve().parents[1]


def core_hashes():
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src").rglob("*.py"))
        if "/viz/" not in str(p)
    }


def main():
    out = ROOT / "results/multistage_validation"
    out.mkdir(parents=True, exist_ok=True)
    base = replace(
        load_config(ROOT / "configs/recourse_multistage.yaml"), seeds=tuple(range(200, 210))
    )
    variants = {
        "repeated_updates": (base, "proposed"),
        "two_stage": (replace(base, future_updates=False), "proposed"),
        "frozen_belief": (replace(base, future_updates=False), "current_belief"),
    }
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "core_sha256": core_hashes(),
        "seeds": list(base.seeds),
        "development_seeds": [20, 21, 22],
        "arms": {
            name: {"config": config_dict(cfg), "method": method}
            for name, (cfg, method) in variants.items()
        },
        "primary_comparison": "Actual integrated monitoring deficit, repeated updates versus frozen future belief",
        "secondary_comparison": "Repeated updates versus the previous two-stage proposed method",
        "policy": "Run and retain every predeclared seed and arm; no changes after viewing results. Record all failures and solver fallbacks.",
    }
    protocol_file = out / "protocol.json"
    if protocol_file.exists():
        saved = json.loads(protocol_file.read_text())
        if saved["core_sha256"] != protocol["core_sha256"] or saved["seeds"] != protocol["seeds"]:
            raise ValueError("Frozen numerical source or seed set changed")
    else:
        protocol_file.write_text(json.dumps(protocol, indent=2) + "\n")
    for name, (cfg, method) in variants.items():
        for seed in cfg.seeds:
            if core_hashes() != protocol["core_sha256"]:
                raise ValueError("Numerical source changed during frozen evaluation")
            file = out / name / f"{cfg.scenario}_{method}_{seed}.json"
            if file.exists() and file.with_suffix(".npz").exists():
                r = json.loads(file.read_text())
                if r["config"] != json.loads(json.dumps(config_dict(cfg))):
                    raise ValueError("Saved configuration mismatch")
                continue
            mission(cfg, method, seed, out / name)


if __name__ == "__main__":
    main()
