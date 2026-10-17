"""Privileged fixed-family physical transit diagnostic; not a global optimum."""

import json
from pathlib import Path

import yaml
from audit_budgeted_mission import physical_wind_numpy
from audit_two_region_physics import clearance, fastest_duration, path_kinematics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/wind_route_study"


def main():
    study = json.loads((OUT / "protocol_01.json").read_text())
    spec = yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text())
    protocol = {
        "label": "Secondary geometry/physics diagnostic. Hidden physical field used only here, never supplied to deployable planners.",
        "paths": "Rest-to-rest quintic timing on middle and symmetric lateral-offset paths; optimize duration only.",
        "declared_sweep": {"duration_range_seconds": [2.0, 30.0], "duration_scan_step": 0.1},
        "offsets": [-1.5, 0.0, 1.5],
        "sample_counts": [4001, 8001],
        "drag": 1.0,
        "component_control_limit": 1.6,
        "claim_limit": "Fastest feasible duration found in this fixed family; no global optimality or learned-planner claim.",
    }
    protocol_path = OUT / "route_physics_protocol_01.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    cases = {}
    for case, field in study["physical_fields"].items():
        rows = []
        for offset in protocol["offsets"]:
            for reverse in [False, True]:
                times = []
                for count in protocol["sample_counts"]:
                    k = path_kinematics(spec, offset, reverse, count)
                    t, status = fastest_duration(
                        k,
                        physical_wind_numpy(k[0], spec, field),
                        protocol["drag"],
                        protocol["component_control_limit"],
                        protocol,
                    )
                    times.append(t)
                if all(t is not None for t in times):
                    assert abs(times[1] - times[0]) < 0.002
                rows.append(
                    {
                        "offset": offset,
                        "direction": "B_to_A" if reverse else "A_to_B",
                        "seconds": times[-1],
                        "refinement_seconds": times,
                        "status": status,
                        "geometric_clearance": clearance(k[0], spec),
                    }
                )
        best = {
            direction: min(
                [
                    r
                    for r in rows
                    if r["direction"] == direction
                    and r["seconds"] is not None
                    and r["geometric_clearance"] > 0.06
                ],
                key=lambda r: r["seconds"],
            )
            for direction in ["A_to_B", "B_to_A"]
        }
        cases[case] = {"routes": rows, "fastest_in_fixed_family": best}
    (OUT / "route_physics_01.json").write_text(
        json.dumps({"protocol": protocol, "cases": cases}, indent=2) + "\n"
    )
    print(json.dumps({case: r["fastest_in_fixed_family"] for case, r in cases.items()}, indent=2))


if __name__ == "__main__":
    main()
