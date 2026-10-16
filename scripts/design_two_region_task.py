"""Screen monitoring demand BEFORE comparing planners or manufacturing VoI.

Prescribed sensor motions use the actual clarity ODE on the complete grid.
They are not MBD, POAM, wind-aware simulations or evidence of learning benefit.
All motions start at A; B-only policies pay for the initial transfer. Single-ROI
policies include stationary and sweeping sensors. An independent optimistic
bound covers all policies confined to one region after the initial transfer.
Beating that bound establishes a useful revisiting regime, not global optimality.
"""

import json
import os
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import yaml
from scipy.integrate import quad

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from voi_diffusion_ipp.clarity import step


def layout(spec):
    bounds = np.array(spec["bounds"])
    nx, ny = spec["grid_shape"]
    dx, dy = (bounds[1] - bounds[0]) / [nx, ny]
    xx, yy = np.meshgrid(
        np.linspace(bounds[0, 0] + dx / 2, bounds[1, 0] - dx / 2, nx),
        np.linspace(bounds[0, 1] + dy / 2, bounds[1, 1] - dy / 2, ny),
    )
    cells = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    a, b = np.array(spec["region_a"]), np.array(spec["region_b"])
    masks = np.stack(
        [
            np.all(np.abs(cells - center) <= np.array(spec["region_size"]) / 2, axis=1)
            for center in [a, b]
        ]
    )
    assert np.all(masks.sum(1) > 30) and not np.any(masks[0] & masks[1])
    tangent = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-tangent[1], tangent[0]])
    obstacles = np.array(
        [
            np.r_[sign * normal * spec["obstacle_normal_offset"], spec["obstacle_radius"]]
            for sign in [-1, 1]
        ]
    )
    return cells, masks, obstacles


def ease(t):
    t = np.clip(t, 0, 1)
    return t**3 * (10 - 15 * t + 6 * t * t)


def reference_path(spec, times, kind, dwell=4.0):
    a, b = np.array(spec["region_a"]), np.array(spec["region_b"])
    transfer = spec["transfer_seconds"]
    if kind == "shuttle":
        period = 2 * (dwell + transfer)
        phase = times % period
        toward_b = ease((phase - dwell) / transfer)
        toward_a = ease((phase - (2 * dwell + transfer)) / transfer)
        fraction = np.where(phase < dwell + transfer, toward_b, 1 - toward_a)
        return a + (b - a) * fraction[:, None]
    goal = a if kind.startswith("A") else b
    delay = 0 if kind.startswith("A") else transfer
    path = np.broadcast_to(goal, (len(times), 2)).copy()
    if delay:
        path = a + (goal - a) * ease(times / transfer)[:, None]
    if kind.endswith("sweep"):
        tau = np.maximum(times - delay, 0)
        # Smooth startup; periodic scan fits within each large region.
        scale = ease(tau / 8)
        angle = 2 * np.pi * tau / 8
        path += 0.30 * scale[:, None] * np.stack((np.sin(angle), np.sin(2 * angle)), axis=1)
    return path


def evaluator(cells, masks, spec):
    cells = jnp.array(cells)
    weights = jnp.array(masks / masks.sum(1)[:, None])
    union = jnp.any(jnp.array(masks), axis=0)
    targets = jnp.where(union, spec["target"], 0.0)
    q0 = jnp.full(len(cells), spec["initial_clarity"])

    @jax.jit
    def run(path, sigma, Q):
        def body(q, segment):
            p0, p1 = segment
            qn = step(q, p0, p1, cells, sigma, spec["sensor_R"], Q, spec["dt"])
            regional = weights @ jnp.maximum(targets - qn, 0)
            regional_q = weights @ qn
            return qn, (regional, regional_q)

        qf, (deficits, qs) = jax.lax.scan(body, q0, (path[:-1], path[1:]))
        d0 = weights @ jnp.maximum(targets - q0, 0)
        qstart = weights @ q0
        deficits = jnp.concatenate((d0[None], deficits))
        qs = jnp.concatenate((qstart[None], qs))
        cost = jnp.trapezoid(jnp.mean(deficits, axis=1), dx=spec["dt"])
        return cost, deficits, qs, qf

    return run


def confined_region_bound(spec, cells, masks, sigma, Q):
    """Optimistic lower bound for any policy confined to one ROI after 4 s.

    Grant zero deficit during the initial transfer, reset ALL clarity to one,
    and keep the visited ROI perfect forever. For every neglected cell grant
    its maximum possible sensor strength from ANY point of the visited ROI,
    simultaneously. This dominates any realizable confined-ROI sensing policy.
    The independent scalar covariance solution is P=Q*t*tanh(z)/z,
    z=t*sqrt(Q*Cmax^2/R), with P(0)=0 and q=1/(1+P).
    Gaussian sensing tails are retained, not rounded to zero.
    """
    bounds = []
    errors = []
    for i, center in enumerate([spec["region_a"], spec["region_b"]]):
        neglected = cells[masks[1 - i]]
        distance = np.maximum(
            np.abs(neglected - np.array(center)) - np.array(spec["region_size"]) / 2, 0
        )
        a = np.exp(-np.sum(distance**2, axis=1) / sigma**2) / spec["sensor_R"]

        def rate(t, a=a):
            z = t * np.sqrt(Q * a)
            ratio = np.divide(np.tanh(z), z, out=np.ones_like(z), where=z != 0)
            q = 1 / (1 + Q * t * ratio)
            return 0.5 * float(np.maximum(spec["target"] - q, 0).mean())

        value, error = quad(
            rate, 0, spec["mission_seconds"] - spec["transfer_seconds"], epsabs=1e-9, epsrel=1e-9
        )
        bounds.append(value)
        errors.append(error)
    return min(bounds), max(errors)


def main():
    spec = yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text())
    cells, masks, obstacles = layout(spec)
    times = np.arange(round(spec["mission_seconds"] / spec["dt"]) + 1) * spec["dt"]
    run = evaluator(cells, masks, spec)
    paths = {
        kind: reference_path(spec, times, kind)
        for kind in ["A_hover", "A_sweep", "B_hover", "B_sweep"]
    }
    paths |= {
        f"shuttle_{dwell:g}": reference_path(spec, times, "shuttle", dwell)
        for dwell in spec["dwell_seconds"]
    }
    dynamics = {}
    for kind, path in paths.items():
        velocity = np.gradient(path, spec["dt"], axis=0, edge_order=2)
        acceleration = np.gradient(velocity, spec["dt"], axis=0, edge_order=2)
        clearance = (
            np.linalg.norm(path[:, None] - obstacles[None, :, :2], axis=-1) - obstacles[None, :, 2]
        )
        dynamics[kind] = {
            "max_axis_acceleration": float(np.abs(acceleration).max()),
            "minimum_geometric_clearance": float(clearance.min()),
        }
    rows = []
    telemetry = {}
    for sigma, Q in product(spec["sensor_sigmas"], spec["process_noise_values"]):
        costs = {}
        for kind, path in paths.items():
            out = jax.block_until_ready(run(jnp.array(path), sigma, Q))
            costs[kind] = float(out[0])
            telemetry[sigma, Q, kind] = out
        solo = min((k for k in paths if not k.startswith("shuttle")), key=costs.get)
        shuttle = min((k for k in paths if k.startswith("shuttle")), key=costs.get)
        bound, quadrature_error = confined_region_bound(spec, cells, masks, sigma, Q)
        rows.append(
            {
                "sigma": sigma,
                "Q": Q,
                "costs": costs,
                "best_single": solo,
                "best_shuttle": shuttle,
                "shuttle_reduction": 1 - costs[shuttle] / costs[solo],
                "optimistic_confined_region_bound": bound,
                "bound_quadrature_error": quadrature_error,
                "shuttle_reduction_vs_bound": 1 - costs[shuttle] / bound,
                "unsensed_08_to_06_seconds": (1 / 0.6 - 1 / 0.8) / Q,
                "unsensed_08_to_04_seconds": (1 / 0.4 - 1 / 0.8) / Q,
            }
        )
    feasible = all(
        v["max_axis_acceleration"] <= spec["maximum_control_acceleration"]
        and v["minimum_geometric_clearance"] > 0
        for v in dynamics.values()
    )
    eligible = [r for r in rows if r["shuttle_reduction_vs_bound"] >= 0.10 and feasible]
    chosen = max(eligible, key=lambda r: (r["Q"], -r["sigma"])) if eligible else None
    outdir = ROOT / "docs/two_region_design"
    outdir.mkdir(exist_ok=True)
    result = {
        "label": "Task-design diagnostic only: prescribed sensing paths, no planner or wind",
        "grid_cells": len(cells),
        "region_cells": masks.sum(1).tolist(),
        "spec": spec,
        "geometry_and_acceleration": dynamics,
        "rows": rows,
        "selected": chosen,
        "warning": "The analytic lower bound applies to policies confined to one ROI after the initial transfer; it is not the global monitoring optimum. No learning advantage, wind safety or one-second planner is established.",
    }
    (outdir / "screen.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"feasible": feasible, "selected": chosen, "all_cases": rows}, indent=2),
        flush=True,
    )
    if chosen is None:
        return
    sigma, Q = chosen["sigma"], chosen["Q"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    ax = axes[0]
    for center, name, color in [
        (spec["region_a"], "Region A", "#bf625b"),
        (spec["region_b"], "Region B", "#287f83"),
    ]:
        ax.add_patch(
            Rectangle(
                np.array(center) - np.array(spec["region_size"]) / 2,
                *spec["region_size"],
                facecolor=color,
                alpha=0.20,
            )
        )
        ax.text(*center, name, ha="center", va="center", weight="bold")
    for x, y, r in obstacles:
        ax.add_patch(Circle((x, y), r, color="#68747c"))
    a, b = np.array(spec["region_a"]), np.array(spec["region_b"])
    ax.plot([a[0], b[0]], [a[1], b[1]], "--", color="#667580", label="Middle transfer reference")
    # Proposed physical mechanism, not simulated wind in this clarity-only screen.
    ax.annotate(
        "",
        xy=0.35 * b,
        xytext=0.35 * a,
        arrowprops={"arrowstyle": "->", "color": "#4369a6", "lw": 2.5},
    )
    ax.set(
        xlim=[-2.7, 2.7],
        ylim=[-2.1, 2.1],
        aspect="equal",
        xlabel="x [m]",
        ylabel="y [m]",
        title="Large regions; two-obstacle passage",
    )
    ax.legend(fontsize=8, loc="lower right")
    for kind, color in [(chosen["best_shuttle"], "#187e83"), (chosen["best_single"], "#ae7136")]:
        _cost, d, _qs, _qf = telemetry[sigma, Q, kind]
        axes[1].plot(times, np.mean(d, axis=1), color=color, label=kind)
    axes[1].set(
        xlabel="Time [s]",
        ylabel="Mean deficit across A and B",
        title="Actual clarity ODE; prescribed sensor paths",
    )
    axes[1].legend(fontsize=8)
    qhistory = np.asarray(telemetry[sigma, Q, chosen["best_shuttle"]][2])
    axes[2].plot(times, qhistory[:, 0], label="Region A", color="#bf625b")
    axes[2].plot(times, qhistory[:, 1], label="Region B", color="#287f83")
    axes[2].axhline(spec["target"], ls=":", color="#707070", label="Target")
    axes[2].set(
        xlabel="Time [s]",
        ylabel="Mean regional clarity",
        ylim=[0, 1],
        title=f"Shuttle recovery and loss · Q={Q:g}",
    )
    axes[2].legend(fontsize=8)
    fig.suptitle("TASK DESIGN ONLY — no MBD/POAM/wind performance claim", fontsize=12)
    fig.savefig(outdir / "task_design.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
