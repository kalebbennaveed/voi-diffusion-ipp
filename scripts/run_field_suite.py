"""Run the predeclared full-POAM development pairs and capture real diffusion telemetry."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.viz.demo_data import export_demo

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenario", choices=["field_vortices", "field_shear", "field_mixed"])
args = parser.parse_args()
cfg = load_config("configs/field_poam.yaml")
root = Path("results/field_poam/recordings")
root.mkdir(parents=True, exist_ok=True)
rows = []
for scenario, seed in [("field_vortices", 300), ("field_shear", 301), ("field_mixed", 302)]:
    if args.scenario and scenario != args.scenario:
        continue
    for method in ["proposed", "current_belief"]:
        path = root / scenario / f"{scenario}_{method}_{seed}.demo.json"
        if not path.exists():
            print(time.strftime("%H:%M:%S"), "START", scenario, method, seed, flush=True)
            path = export_demo(
                replace(cfg, scenario=scenario),
                method,
                seed,
                root / scenario,
                lambda s: print(time.strftime("%H:%M:%S"), s, flush=True),
            )
        rows.append(str(path))
        (root / f"completed_{args.scenario or 'all'}.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )
