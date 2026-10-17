"""Independent NumPy dense physical checks and measured persistent-task summaries."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def physical_wind_numpy(positions, task_spec, field_spec=None):
    """Independent float64 field equation, including archived legacy recordings."""
    field_spec = field_spec or {
        "family": "localized_passage_jet_v1",
        "amplitude": 1.8,
        "axial_fraction": 1.0,
        "cross_fraction": 0.2,
        "center": [0.0, 0.0],
        "longitudinal_width": 1.1,
        "transverse_width": 0.35,
    }
    if field_spec["family"] != "localized_passage_jet_v1":
        raise ValueError("Unknown archived physical field")
    tangent = np.array(task_spec["region_b"], dtype=float) - task_spec["region_a"]
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    p = np.atleast_2d(np.asarray(positions, dtype=float)) - field_spec["center"]
    envelope = np.exp(
        -0.5
        * (
            (p @ tangent / field_spec["longitudinal_width"]) ** 4
            + (p @ normal / field_spec["transverse_width"]) ** 2
        )
    )
    return (
        field_spec["amplitude"]
        * envelope[:, None]
        * (field_spec["axial_fraction"] * tangent + field_spec["cross_fraction"] * normal)
    )


def audit(path, substeps=32):
    doc = json.loads(path.with_suffix(".json").read_text())
    meta = doc["metadata"]
    d = np.load(path, allow_pickle=False)
    spec = meta.get("task_spec") or yaml.safe_load(
        (ROOT / "configs/two_region_design.yaml").read_text()
    )
    a, b = np.array(spec["region_a"]), np.array(spec["region_b"])
    tangent = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-tangent[1], tangent[0]])
    dt = meta["execution_dt"]
    controls = np.array(d["controls"], dtype=np.float64)
    recorded = np.array(d["x"], dtype=np.float64)
    # All intervals independently reintegrated from their recorded initial states.
    # No JAX dynamics, RPCBF outputs or stored clearance used for this check.
    x = recorded[:-1].copy()
    drag = meta["config"]["linear_drag"]
    minimum = np.full(len(x), np.inf)
    obstacles, bounds = d["obstacles"], d["bounds"]

    # Refuse to silently apply this study-specific field to another recording.
    cells = np.asarray(d["cells"], dtype=float)
    expected_wind = physical_wind_numpy(cells, spec, meta.get("physical_wind_spec"))
    np.testing.assert_allclose(d["physical_wind"], expected_wind, atol=2e-6, rtol=2e-6)

    def derivative(x):
        wind = physical_wind_numpy(x[:, :2], spec, meta.get("physical_wind_spec"))
        return np.concatenate((x[:, 2:], controls - drag * x[:, 2:] + wind), axis=1)

    h = dt / substeps
    for _ in range(substeps):
        k1 = derivative(x)
        k2 = derivative(x + h * k1 / 2)
        k3 = derivative(x + h * k2 / 2)
        k4 = derivative(x + h * k3)
        x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        gap = (
            np.linalg.norm(x[:, None, :2] - obstacles[None, :, :2], axis=-1) - obstacles[None, :, 2]
        )
        boundary = np.minimum(x[:, :2] - bounds[0], bounds[1] - x[:, :2]).min(axis=1)
        minimum = np.minimum(minimum, np.minimum(gap.min(axis=1), boundary))
    q, cells, targets = d["q"], d["cells"], d["targets"]
    mask = np.all(
        np.abs(cells[:, None] - np.stack((a, b))[None]) <= np.array(spec["region_size"]) / 2,
        axis=-1,
    )
    costs = np.maximum(targets - q, 0).mean(axis=1)
    integrated = float(np.trapezoid(costs, dx=dt))
    visits = []
    for name, center in [("A", a), ("B", b)]:
        inside = np.all(
            np.abs(recorded[:, :2] - center) <= np.array(spec["region_size"]) / 2, axis=1
        )
        changes = np.diff(np.r_[False, inside, False].astype(int))
        for start, end in zip(
            np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True
        ):
            duration = (end - start - 1) * dt
            if duration >= 0.5:
                visits.append({"region": name, "entry_s": start * dt, "duration_s": duration})
    visits.sort(key=lambda v: v["entry_s"])
    sequence = []
    for v in visits:
        if not sequence or sequence[-1] != v["region"]:
            sequence.append(v["region"])
    transfers = []
    previous = None
    last_inside = 0
    for i, state in enumerate(recorded):
        region = next(
            (
                name
                for name, center in [("A", a), ("B", b)]
                if np.all(np.abs(state[:2] - center) <= np.array(spec["region_size"]) / 2)
            ),
            None,
        )
        if region is not None:
            if previous is not None and region != previous:
                section = recorded[last_inside : i + 1, :2]
                crossing = int(np.argmin(np.abs(section @ tangent)))
                lateral = float(section[crossing] @ normal)
                transfers.append(
                    {
                        "direction": previous + "_to_" + region,
                        "departure_s": last_inside * dt,
                        "arrival_s": i * dt,
                        "seconds": (i - last_inside) * dt,
                        "midplane_lateral_m": lateral,
                        "route": "middle"
                        if abs(lateral) < 0.22
                        else "outer"
                        if abs(lateral) > 1.22
                        else "other",
                    }
                )
            previous = region
            last_inside = i
    passage = (np.abs(cells @ tangent) < 1.1) & (np.abs(cells @ normal) < 0.22)
    passage_rmse = np.sqrt(
        np.mean((d["mean"][:, passage] - d["physical_wind"][passage]) ** 2, axis=(1, 2))
    )
    passage_std = np.mean(d["std"][:, passage], axis=1)
    periods = doc["periods"]
    times = [p["planning_cycle_s"] for p in periods[2:]]
    result = {
        "input": str(path.relative_to(ROOT)),
        "method": meta["method"],
        "wind_case": meta.get("wind_case", "original"),
        "complete": doc["complete"],
        "duration_s": (len(recorded) - 1) * dt,
        "integrated_whole_grid_deficit": integrated,
        "integrated_roi_deficit": integrated * len(cells) / np.any(mask, axis=1).sum(),
        "recorded_cost_disagreement": abs(integrated - periods[-1]["integrated_deficit"]),
        "dense_substeps": substeps,
        "minimum_dense_clearance": float(minimum.min()),
        "contact_intervals": int(np.sum(minimum < 0)),
        "first_contact_interval_start_s": float(np.flatnonzero(minimum < 0)[0] * dt)
        if np.any(minimum < 0)
        else None,
        "max_local_state_error": float(np.max(np.abs(x - recorded[1:]))),
        "peak_control": float(np.abs(controls).max()),
        "backup_periods": sum(p["backup_used"] for p in periods),
        "visits": visits,
        "transfers": transfers,
        "directional_routes": {
            direction: {
                "count": len(selected := [v for v in transfers if v["direction"] == direction]),
                "route_counts": {
                    route: sum(v["route"] == route for v in selected)
                    for route in ["middle", "outer", "other"]
                },
                "mean_seconds": float(np.mean([v["seconds"] for v in selected]))
                if selected
                else None,
            }
            for direction in ["A_to_B", "B_to_A"]
        },
        "passage_rmse_initial_final": [float(passage_rmse[0]), float(passage_rmse[-1])],
        "passage_std_initial_final": [float(passage_std[0]), float(passage_std[-1])],
        "visit_sequence": sequence,
        "startup_cycle_s": [r["planning_cycle_s"] for r in periods[:2]],
        "timing_definition": "First two cycles reported separately; remaining cycles include input preparation, previous real POAM update and full selected-trajectory check.",
        "steady_cycle_s": {
            k: float(np.quantile(times, q)) for k, q in [("median", 0.5), ("p95", 0.95), ("max", 1)]
        },
        "cycle_1s_miss_fraction": float(np.mean(np.array(times) > 1)),
        "cycle_2s_miss_fraction": float(np.mean(np.array(times) > 2)),
        "final_mean_clarity": {
            name: float(q[-1, mask[:, i]].mean()) for i, name in enumerate(["A", "B"])
        },
        "initialization": meta.get("prior_regime", "cold"),
        "limits": "Development seed; empirical dense collision check, no formal safety claim. Oracle is explicitly privileged.",
    }
    path.with_suffix(".audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", type=Path)
    args = p.parse_args()
    for path in args.paths:
        print(json.dumps(audit(path.resolve()), indent=2))


if __name__ == "__main__":
    main()
