"""Known-wind transfer audit, NOT MBD or evidence of an active-learning gain.

Same smooth path/timing family, actuator bounds and wind in both directions.
Only transfer duration changes; no direction-specific route cost is introduced.
"""

import json
import os
from itertools import pairwise, product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import yaml
from scipy.optimize import brentq

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from voi_diffusion_ipp.dynamics import step


def frame(spec):
    a, b = np.array(spec["region_a"]), np.array(spec["region_b"])
    tangent = (b - a) / np.linalg.norm(b - a)
    return a, b, tangent, np.array([-tangent[1], tangent[0]])


def path_kinematics(spec, offset, reverse, points=4001):
    """p, dp/dtau, d²p/dtau² for tau=t/T with rest-to-rest endpoints."""
    a, b, _, normal = frame(spec)
    tau = np.linspace(0, 1, points)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    bump = 16 * s**2 * (1 - s) ** 2
    db = 32 * s - 96 * s**2 + 64 * s**3
    ddb = 32 - 192 * s + 192 * s**2
    position = a + s[:, None] * (b - a) + offset * bump[:, None] * normal
    velocity = ds[:, None] * ((b - a) + offset * db[:, None] * normal)
    acceleration = dds[:, None] * ((b - a) + offset * db[:, None] * normal)
    acceleration += offset * (ds**2 * ddb)[:, None] * normal
    if reverse:
        return position[::-1].copy(), -velocity[::-1].copy(), acceleration[::-1].copy()
    return position, velocity, acceleration


def field(positions, spec, protocol, amplitude, xp=np):
    _, _, tangent, normal = frame(spec)
    tangent, normal = xp.asarray(tangent), xp.asarray(normal)
    sweep = protocol["declared_sweep"]
    p = xp.atleast_2d(positions)
    axial, lateral = p @ tangent, p @ normal
    envelope = xp.exp(
        -0.5
        * ((axial / sweep["longitudinal_width"]) ** 4 + (lateral / sweep["transverse_width"]) ** 2)
    )
    direction = tangent + sweep["cross_acceleration_fraction"] * normal
    return amplitude * envelope[:, None] * direction


def required_controls(kinematics, wind, drag, duration):
    _, velocity, acceleration = kinematics
    return acceleration / duration**2 + drag * velocity / duration - wind


def fastest_duration(kinematics, wind, drag, limit, protocol):
    sweep = protocol["declared_sweep"]
    lo, hi = sweep["duration_range_seconds"]
    durations = np.arange(lo, hi + sweep["duration_scan_step"] / 2, sweep["duration_scan_step"])

    def residual(t):
        return np.max(np.abs(required_controls(kinematics, wind, drag, t))) - limit

    last = residual(lo)
    if last <= 0:
        return float(lo), "lower_search_boundary"
    for a, b in pairwise(durations):
        value = residual(b)
        if value <= 0:
            return float(brentq(residual, a, b, xtol=1e-10)), "interior"
        last = value
    return None, "no_feasible_duration_in_declared_range"


def clearance(positions, spec):
    _, _, _, normal = frame(spec)
    centers = np.array([-normal, normal]) * spec["obstacle_normal_offset"]
    circles = np.linalg.norm(positions[:, None] - centers[None], axis=-1) - spec["obstacle_radius"]
    lower, upper = np.array(spec["bounds"])
    return float(min(circles.min(), (positions - lower).min(), (upper - positions).min()))


def verify(spec, protocol, offset, reverse, amplitude, drag, duration):
    # Held controls at step midpoints; independently integrate actual ODE.
    points = 8001
    k = path_kinematics(spec, offset, reverse, points)
    controls = required_controls(k, field(k[0], spec, protocol, amplitude), drag, duration)
    mids = jnp.asarray((controls[1:] + controls[:-1]) / 2)
    dt = duration / (points - 1)
    wind = lambda p: field(p, spec, protocol, amplitude, jnp)[0]

    def body(x, u):
        xn = step(x, u, wind, dt, drag)
        return xn, xn

    x0 = jnp.asarray(np.r_[k[0][0], [0.0, 0.0]])
    _, states = jax.jit(lambda: jax.lax.scan(body, x0, mids))()
    states = np.concatenate((np.array(x0)[None], np.asarray(states)))
    return {
        "max_position_error": float(np.linalg.norm(states[:, :2] - k[0], axis=1).max()),
        "max_velocity_error": float(np.linalg.norm(states[:, 2:] - k[1] / duration, axis=1).max()),
        "max_axis_control": float(np.abs(mids).max()),
        "minimum_clearance": clearance(states[:, :2], spec),
        "substeps": points - 1,
    }


def main():
    spec = yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text())
    protocol = json.loads((ROOT / "docs/two_region_physics/protocol.json").read_text())
    sweep = protocol["declared_sweep"]
    paths = {
        (h, rev): path_kinematics(spec, h, rev, sweep["verification_points"])
        for h, rev in product(sweep["bypass_offsets"], [False, True])
    }
    results = []
    for drag, amplitude in product(sweep["drag_per_second"], sweep["along_acceleration"]):
        rows = []
        for (h, rev), k in paths.items():
            time, status = fastest_duration(
                k,
                field(k[0], spec, protocol, amplitude),
                drag,
                spec["maximum_control_acceleration"],
                protocol,
            )
            rows.append(
                {
                    "offset": h,
                    "direction": "B_to_A" if rev else "A_to_B",
                    "seconds": time,
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
                    and r["geometric_clearance"] > 0
                ],
                key=lambda r: r["seconds"],
            )
            for direction in ["A_to_B", "B_to_A"]
        }
        middle_return = next(
            r["seconds"] for r in rows if r["direction"] == "B_to_A" and r["offset"] == 0
        )
        gain = 1 - best["B_to_A"]["seconds"] / middle_return if middle_return else None
        passed = best["A_to_B"]["offset"] == 0 and best["B_to_A"]["offset"] != 0 and gain >= 0.10
        results.append(
            {
                "drag": drag,
                "amplitude": amplitude,
                "routes": rows,
                "best": best,
                "return_time_reduction": gain,
                "physics_gate": passed,
            }
        )
    eligible = [r for r in results if r["physics_gate"]]
    chosen = min(eligible, key=lambda r: (r["drag"], r["amplitude"])) if eligible else None
    verification = []
    if chosen:
        for row in chosen["routes"]:
            if row["seconds"] is not None and row["geometric_clearance"] > 0:
                verification.append(
                    row
                    | verify(
                        spec,
                        protocol,
                        row["offset"],
                        row["direction"] == "B_to_A",
                        chosen["amplitude"],
                        chosen["drag"],
                        row["seconds"],
                    )
                )
    out = {
        "label": "Known-wind fixed-family physics audit; not planner/learning evidence",
        "spec": spec,
        "protocol": protocol,
        "results": results,
        "selected": chosen,
        "independent_integration": verification,
    }
    output = ROOT / "docs/two_region_physics"
    (output / "screen.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"selected": chosen, "verification": verification}, indent=2), flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    a, b, _, normal = frame(spec)
    for sign in [-1, 1]:
        axes[0].add_patch(
            Circle(
                sign * normal * spec["obstacle_normal_offset"],
                spec["obstacle_radius"],
                color="#6c777e",
            )
        )
    for h in sweep["bypass_offsets"]:
        p = paths[h, False][0]
        axes[0].plot(*p.T, label=f"offset {h:g} m")
    axes[0].text(*a, "A", fontsize=16)
    axes[0].text(*b, "B", fontsize=16)
    axes[0].set(
        aspect="equal",
        xlabel="x [m]",
        ylabel="y [m]",
        title="Identical route families in both directions",
    )
    axes[0].legend()
    if chosen:
        for i, direction in enumerate(["A_to_B", "B_to_A"]):
            rows = [r for r in chosen["routes"] if r["direction"] == direction]
            axes[1].bar(
                np.arange(3) + i * 0.36, [r["seconds"] for r in rows], width=0.36, label=direction
            )
        axes[1].set(
            xticks=np.arange(3) + 0.18,
            xticklabels=["Outer −", "Middle", "Outer +"],
            ylabel="Shortest feasible duration in tested family [s]",
            title=f"Drag={chosen['drag']:g} s⁻¹ · wind acceleration={chosen['amplitude']:g}",
        )
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No declared case passed; retained for diagnosis", ha="center")
    fig.suptitle("PHYSICS AUDIT ONLY — known wind; no MBD or learning claim")
    fig.savefig(output / "route_times.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
