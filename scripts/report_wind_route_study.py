"""Matched wind-intervention development report; no optimization or truth feedback.

All complete outcomes are retained, including worse task cost and contacts.
The transfer-family calculation is a privileged physics diagnostic, not an
optimal-control solution, MBD comparison, or evidence of anticipated learning.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit_budgeted_mission import audit
from matplotlib.patches import Circle, Rectangle

OUT = ROOT / "docs/wind_route_study"
NAMES = {
    "original": "Original: A → B wind",
    "reversed": "Reversed: B → A wind",
    "crosswind": "Crosswind through the gap",
    "offroute": "Wind away from the route",
    "offroute_surveyed": "Off-route wind: main route surveyed",
}
METHODS = {
    "proposed": "Anticipated learning",
    "current_belief": "Current belief",
    "wind_blind_planner": "Wind-blind plan + safety",
}
COLORS = {"proposed": "#087f83", "current_belief": "#b27738", "wind_blind_planner": "#8171b0"}


def inputs(case):
    if case == "original":
        return {
            m: ROOT / "results/budgeted_missions" / folder / (m + ".npz")
            for m, folder in [
                ("proposed", "persistent_two_region_2s_01"),
                ("current_belief", "persistent_two_region_2s_01"),
                ("wind_blind_planner", "persistent_two_region_safety_only_01"),
            ]
        }
    directory = ROOT / "results/budgeted_missions" / (case + "_wind_01")
    return {p.stem: p for p in sorted(directory.glob("*.npz")) if p.stem in METHODS}


def clarity_check(data, config, dt):
    """Independent NumPy replay on all cells along the recorded executed path."""
    q = np.array(data["q"][0], dtype=float)
    cells = np.array(data["cells"], dtype=float)
    sigma, R, Q = config["sensor_sigma"], config["sensor_R"], config["clarity_decay"]
    errors = []
    for i, (p0, p1) in enumerate(zip(data["x"][:-1, :2], data["x"][1:, :2], strict=True)):
        gains = [
            np.exp(-np.sum((cells - p) ** 2, axis=1) / sigma**2) / R
            for p in [p0, (p0 + p1) / 2, p1]
        ]

        def f(value, gain):
            return gain * (1 - value) ** 2 - Q * value**2

        k1 = f(q, gains[0])
        k2 = f(q + dt * k1 / 2, gains[1])
        k3 = f(q + dt * k2 / 2, gains[1])
        k4 = f(q + dt * k3, gains[2])
        q = np.clip(q + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, 0, 1)
        errors.append(float(np.max(np.abs(q - data["q"][i + 1]))))
    return max(errors)


def matched(paths):
    entries = [
        (p, json.loads(p.with_suffix(".json").read_text()), np.load(p, allow_pickle=False))
        for p in paths.values()
    ]
    _, reference, data = entries[0]
    for _, raw, values in entries[1:]:
        for key in [
            "config",
            "seed",
            "prior_regime",
            "execution_dt",
            "observation_dt",
            "task_sha256",
            "rapid_shared_initializations",
        ]:
            assert reference["metadata"][key] == raw["metadata"][key], key
        for key in [
            "physical_wind",
            "cells",
            "targets",
            "obstacles",
            "bounds",
            "initial_observation_positions",
            "initial_observations",
        ]:
            np.testing.assert_array_equal(data[key], values[key])
        for key in ["x", "q"]:
            assert data[key].shape == values[key].shape
            np.testing.assert_array_equal(data[key][0], values[key][0])
    for _, _, values in entries:
        values.close()
    return True


def report():
    result = {
        "label": "Development seed 8101; no held-out success claim",
        "cases": {},
        "input_sha256": {},
    }
    records = {}
    for case in NAMES:
        paths = {
            m: p
            for m, p in inputs(case).items()
            if json.loads(p.with_suffix(".json").read_text())["complete"]
        }
        if not paths:
            continue
        assert matched(paths)
        records[case] = {}
        rows = {}
        for method, path in paths.items():
            raw = json.loads(path.with_suffix(".json").read_text())
            meta = raw["metadata"]
            with np.load(path, allow_pickle=False) as archive:
                data = {k: archive[k] for k in archive.files}
            a = audit(path)
            error = clarity_check(data, meta["config"], meta["execution_dt"])
            assert error < 2e-5, error
            times = np.arange(len(data["x"])) * meta["execution_dt"]
            deficit = np.maximum(data["targets"] - data["q"], 0).mean(axis=1)
            half = np.searchsorted(times, 60)
            rows[method] = a | {
                "clarity_replay_max_error": error,
                "first_half_cost": float(
                    np.trapezoid(deficit[: half + 1], dx=meta["execution_dt"])
                ),
                "second_half_cost": float(np.trapezoid(deficit[half:], dx=meta["execution_dt"])),
                "early_middle_transfers": sum(
                    t["route"] == "middle" and t["arrival_s"] <= 60 for t in a["transfers"]
                ),
                "late_middle_transfers": sum(
                    t["route"] == "middle" and t["arrival_s"] > 60 for t in a["transfers"]
                ),
            }
            records[case][method] = (data, meta, a)
            for file in [path, path.with_suffix(".json")]:
                result["input_sha256"][str(file.relative_to(ROOT))] = hashlib.sha256(
                    file.read_bytes()
                ).hexdigest()
        result["cases"][case] = {"matched_inputs_checked": True, "methods": rows}
        contacts = [
            r["first_contact_interval_start_s"]
            for r in rows.values()
            if r["first_contact_interval_start_s"] is not None
        ]
        result["cases"][case]["all_missions_without_contact"] = not contacts
        if contacts:
            cut = min(contacts)
            prefix = {}
            for m, (data, meta, _) in records[case].items():
                last = round(cut / meta["execution_dt"])
                rates = np.maximum(data["targets"] - data["q"][: last + 1], 0).mean(axis=1)
                prefix[m] = float(np.trapezoid(rates, dx=meta["execution_dt"]))
            result["cases"][case]["matched_prefix_before_first_contact"] = {
                "through_seconds": cut,
                "costs": prefix,
            }
            result["cases"][case]["cost_warning"] = (
                "Full-duration costs include post-contact integration without impact response; not a comparison of successful completed missions."
            )
        if "proposed" in rows:
            cost = rows["proposed"]["integrated_whole_grid_deficit"]
            result["cases"][case]["proposed_cost_reduction_percent"] = {
                m: 100 * (1 - cost / r["integrated_whole_grid_deficit"])
                for m, r in rows.items()
                if m != "proposed"
            }
    if "reversed" in records:
        np.testing.assert_array_equal(
            records["reversed"]["proposed"][0]["physical_wind"],
            -records["original"]["proposed"][0]["physical_wind"],
        )
        result["exact_sign_reversal_verified"] = True
    (OUT / "results_01.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(
        len(records), 3, figsize=(13, 3.6 * len(records)), squeeze=False, layout="constrained"
    )
    for row, (case, methods) in enumerate(records.items()):
        for col, method in enumerate(["proposed", "current_belief"]):
            ax = axes[row, col]
            if method not in methods:
                ax.set_visible(False)
                continue
            d, meta, a = methods[method]
            spec = meta.get("task_spec") or __import__("yaml").safe_load(
                (ROOT / "configs/two_region_design.yaml").read_text()
            )
            for cx, cy, r in d["obstacles"]:
                ax.add_patch(Circle((cx, cy), r, facecolor="#d5dbe0", edgecolor="#73808a"))
            for name, center in [("A", spec["region_a"]), ("B", spec["region_b"])]:
                size = np.array(spec["region_size"])
                lower = np.array(center) - size / 2
                ax.add_patch(
                    Rectangle(lower, *size, facecolor="#e7f0e8", edgecolor="#4a6956", alpha=0.55)
                )
                ax.text(*center, name, ha="center", va="center", weight="bold", fontsize=12)
            cells = d["cells"]
            wind = d["physical_wind"]
            ix = np.arange(len(cells))
            nx = meta["config"]["grid_nx"]
            keep = (ix % nx % 3 == 1) & (ix // nx % 3 == 1) & (np.linalg.norm(wind, axis=1) > 0.035)
            ax.quiver(
                *cells[keep].T,
                *wind[keep].T,
                angles="xy",
                scale_units="xy",
                scale=3.33,
                color="#88729a",
                alpha=0.7,
                width=0.004,
            )
            ax.plot(*d["x"][:, :2].T, color=COLORS[method], alpha=0.3, lw=0.8)
            for transfer in a["transfers"]:
                start = round(transfer["departure_s"] / meta["execution_dt"])
                end = round(transfer["arrival_s"] / meta["execution_dt"])
                section = d["x"][start : end + 1, :2]
                color = "#176b98" if transfer["direction"] == "A_to_B" else "#ad5170"
                ax.plot(*section.T, color=color, alpha=0.6, lw=1.1)
            ax.set(
                xlim=d["bounds"][:, 0],
                ylim=d["bounds"][:, 1],
                aspect="equal",
                xlabel="x (m)",
                ylabel="y (m)",
            )
            ax.set_title(
                f"{NAMES[case]}\n{METHODS[method]} · cost {a['integrated_whole_grid_deficit']:.3f}",
                fontsize=10,
            )
        ax = axes[row, 2]
        for method, (d, meta, a) in methods.items():
            deficit = np.maximum(d["targets"] - d["q"], 0).mean(axis=1)
            t = np.arange(len(deficit)) * meta["execution_dt"]
            cumulative = np.r_[
                0, np.cumsum((deficit[1:] + deficit[:-1]) * meta["execution_dt"] / 2)
            ]
            contact = a.get("first_contact_interval_start_s")
            if contact is None:
                ax.plot(t, cumulative, label=METHODS[method], color=COLORS[method], lw=1.5)
            else:
                split = np.searchsorted(t, contact)
                ax.plot(
                    t[: split + 1],
                    cumulative[: split + 1],
                    label=METHODS[method] + " · contact",
                    color=COLORS[method],
                    lw=1.5,
                )
                ax.plot(t[split:], cumulative[split:], color=COLORS[method], lw=1.5, linestyle="--")
                ax.axvline(contact, color=COLORS[method], alpha=0.5, linestyle=":")
                ax.text(
                    contact,
                    0.02,
                    f"contact {contact:.1f} s",
                    transform=ax.get_xaxis_transform(),
                    fontsize=7,
                    rotation=90,
                    va="bottom",
                )
        ax.set(
            xlabel="Mission time (s)",
            ylabel="Integrated whole-grid deficit",
            title="Actual executed monitoring cost",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Wind intervention · 120 s missions · development seed 8101\nBlue routes: A → B; rose routes: B → A. Purple arrows: hidden physical wind.",
        fontsize=13,
    )
    fig.savefig(OUT / "overview_01.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            {
                case: {
                    "gain_percent": r.get("proposed_cost_reduction_percent"),
                    "methods": {
                        m: {
                            "cost": v["integrated_whole_grid_deficit"],
                            "transfers": len(v["transfers"]),
                            "contacts": v["contact_intervals"],
                            "routes": v["directional_routes"],
                        }
                        for m, v in r["methods"].items()
                    },
                }
                for case, r in result["cases"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    report()
