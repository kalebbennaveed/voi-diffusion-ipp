"""Archive every completed paired whole-field run, check telemetry, and plot outcomes."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=Path("results/field_poam/recordings"))
parser.add_argument("--output", type=Path, default=Path("docs/field_poam_validation"))
parser.add_argument("--protocol", type=Path)
parser.add_argument("--label-prefix", default="Full POAM · 18 s")
parser.add_argument("--paired-only", action="store_true")
args = parser.parse_args()
root = Path(".")
source = args.source
out = args.output
out.mkdir(parents=True, exist_ok=True)
protocol = json.loads((args.protocol or out / "protocol.json").read_text())
rows = []
catalog = []
manifest = {}
for scenario, seed in protocol["scenarios_and_seeds"]:
    if args.paired_only and not all(
        (source / scenario / f"{scenario}_{method}_{seed}.demo.json").exists()
        for method in protocol["methods"]
    ):
        continue
    pair = []
    for method in protocol["methods"]:
        stem = f"{scenario}_{method}_{seed}"
        file = source / scenario / f"{stem}.demo.json"
        if not file.exists():
            continue
        d = json.loads(file.read_text())
        m = d["metrics"]
        cfg = d["config"]
        assert cfg["whole_field"] and cfg["wind_backend"] == "poam"
        assert len(d["mission"]["q"][0]) == cfg["grid_nx"] * cfg["grid_ny"]
        q = np.asarray(d["mission"]["q"])
        assert np.isfinite(q).all() and np.all((q >= 0) & (q <= 1))
        targets = np.asarray(d["environment"]["targets"])
        deficit = np.maximum(targets[None] - q, 0).mean(1)
        cumulative = np.r_[0, np.cumsum(cfg["dt"] * (deficit[1:] + deficit[:-1]) / 2)]
        assert abs(cumulative[-1] - m["integrated_deficit"]) < 2e-5
        assert len(d["beliefs"]) == cfg["mission_horizons"] + 1
        for belief in d["beliefs"]:
            assert np.asarray(belief["mean"]).shape == (len(targets), 2)
            assert np.isfinite(belief["mean"]).all()
            assert np.isfinite(belief["std"]).all() and np.min(belief["std"]) >= 0
        assert d["source_sha256"] == protocol["source_sha256"], (
            "Numerical provenance changed during experiment"
        )
        pair.append((d, deficit, cumulative))
        (out / f"{stem}.json").write_text(json.dumps(m, indent=2) + "\n")
        for ext in ("json", "npz", "demo.json"):
            p = source / scenario / f"{stem}.{ext}"
            manifest[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        row = {
            k: m[k]
            for k in (
                "scenario",
                "method",
                "seed",
                "integrated_deficit",
                "final_deficit",
                "wind_rmse",
                "wind_95_coverage",
                "minimum_clearance",
                "collision",
                "max_qp_slack",
                "optimizer_failure_count",
                "mean_plan_seconds",
                "compile_seconds",
            )
        }
        row["sampler_covariance_relative_error"] = m["poam"]["sampler_covariance_relative_error"]
        row["cells_at_target_fraction"] = float(np.mean(q[-1] >= targets))
        row["final_deficit_95th_percentile"] = float(
            np.quantile(np.maximum(targets - q[-1], 0), 0.95)
        )
        row["diffusion_changed_warm_start_count"] = sum(
            not np.allclose(p["initial_vector"], p["selected_vector"], atol=1e-6)
            for p in d["planning"]
        )
        changes = [
            p["trace"][0][0] - p["selected_cost"]
            for p in d["planning"]
            if p["trace"][0][0] is not None and p["selected_cost"] is not None
        ]
        row["mean_cost_improvement_after_first_diffusion_level"] = (
            float(np.mean(changes)) if changes else None
        )
        row["mean_matched_continuation_benefit"] = float(
            np.mean([v["net_benefit"] for v in m["matched_continuations"]])
        )
        row["duration_seconds"] = (len(q) - 1) * cfg["dt"]
        if "persistence" in m:
            row["persistence"] = m["persistence"]
        rows.append(row)
        target = root / "demo/data" / f"{stem}.json"
        target.write_text(file.read_text())
        catalog.append(
            {
                "id": stem,
                "label": f"{args.label_prefix} · {scenario.removeprefix('field_').removeprefix('persistent_')} · {method} · seed {seed}",
                "method": method,
                "seed": seed,
                "variant": "multistage"
                if cfg["future_updates"] and cfg["future_steps"] > cfg["stage_steps"]
                else "two_stage",
                "group": f"full-poam-{scenario}",
                "file": str(target),
                "live": False,
            }
        )
    if len(pair) != 2:
        continue
    a, b = [p[0] for p in pair]
    assert a["physical_wind"] == b["physical_wind"]
    assert a["mission"]["q"][0] == b["mission"]["q"][0]
    assert a["mission"]["x"][0] == b["mission"]["x"][0]
    assert a["config"] == b["config"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for (d, deficit, cost), color in zip(pair, ("#087f83", "#b27738")):
        t = np.arange(len(cost)) * d["config"]["dt"]
        label = d["metrics"]["method"]
        axs[0, 0].plot(t, deficit, color=color, label=label)
        axs[0, 1].plot(t, cost, color=color, label=label)
        wh = d["metrics"]["wind_history"]
        axs[1, 0].step(
            [v["time"] for v in wh], [v["rmse"] for v in wh], where="post", color=color, label=label
        )
        axs[1, 1].step(
            [v["time"] for v in wh],
            [v["mean_std"] for v in wh],
            where="post",
            color=color,
            label=label,
        )
    if "persistence" in a["metrics"]:
        initial = np.asarray(a["mission"]["q"][0])
        process = np.asarray(a["environment"]["process_noise"])
        passive = initial / (1 + t[:, None] * process * initial)
        passive_deficit = np.maximum(np.asarray(a["environment"]["targets"]) - passive, 0).mean(1)
        axs[0, 0].plot(t, passive_deficit, ":", color="#68757c", label="No sensing (analytic)")
    for ax, title in zip(
        axs.flat,
        [
            "Mean clarity deficit ↓",
            "Time-integrated clarity deficit ↓",
            "Wind RMSE over full field ↓",
            "Mean POAM posterior standard deviation",
        ],
    ):
        ax.set(title=title, xlabel="Mission time (s)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"{scenario} · paired development seed {seed} · 475 monitoring cells")
    fig.savefig(out / f"{scenario}_comparison.png", dpi=160)
    plt.close(fig)

    # Common scales make spatial coverage and remaining deficit directly comparable.
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    bounds = np.asarray(a["environment"]["bounds"])
    extent = (bounds[0, 0], bounds[1, 0], bounds[0, 1], bounds[1, 1])
    shape = (a["config"]["grid_ny"], a["config"]["grid_nx"])
    for row, d in enumerate((a, b)):
        q = np.asarray(d["mission"]["q"])
        target = np.asarray(d["environment"]["targets"])
        fields = [target, q[-1], np.maximum(target - q[-1], 0)]
        for col, (values, title) in enumerate(
            zip(fields, ["Clarity target", "Final clarity", "Final clarity deficit ↓"])
        ):
            ax = axs[row, col]
            img = ax.imshow(
                values.reshape(shape),
                origin="lower",
                extent=extent,
                vmin=0,
                vmax=1 if col < 2 else 0.8,
                cmap="viridis",
                interpolation="nearest",
            )
            positions = np.asarray(d["mission"]["x"])
            ax.plot(positions[:, 0], positions[:, 1], color="white", lw=1.4)
            ax.plot(*positions[0, :2], "o", ms=4, color="white")
            ax.plot(*positions[-1, :2], "s", ms=4, color="white")
            for ox, oy, radius in d["environment"]["obstacles"]:
                ax.add_patch(Circle((ox, oy), radius, fc="#bec5c9", ec="#263b49", lw=0.6))
            ax.set(title=f"{d['metrics']['method']} · {title}", xlabel="x (m)", ylabel="y (m)")
            fig.colorbar(img, ax=ax, shrink=0.7)
    fig.suptitle(f"{scenario} · actual executed sensing over all 475 cells · seed {seed}")
    fig.savefig(out / f"{scenario}_clarity_fields.png", dpi=160)
    plt.close(fig)
    if "persistence" in a["metrics"]:
        fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        max_passes = max(max(d["metrics"]["persistence"]["sensing_pass_counts"]) for d in (a, b))
        max_recoveries = max(
            max(d["metrics"]["persistence"]["clarity_recovery_counts"]) for d in (a, b)
        )
        for row, d in enumerate((a, b)):
            p = d["metrics"]["persistence"]
            fields = [
                d["environment"]["process_noise"],
                p["sensing_pass_counts"],
                p["clarity_recovery_counts"],
            ]
            for col, (values, title, vmax) in enumerate(
                zip(
                    fields,
                    ["Process noise Qp", "Distinct sensing passes", "Clarity target recoveries"],
                    [0.1, max(max_passes, 1), max(max_recoveries, 1)],
                )
            ):
                ax = axs[row, col]
                img = ax.imshow(
                    np.asarray(values).reshape(shape),
                    origin="lower",
                    extent=extent,
                    vmin=0,
                    vmax=vmax,
                    cmap="viridis",
                    interpolation="nearest",
                )
                ax.set(title=f"{d['metrics']['method']} · {title}", xlabel="x (m)", ylabel="y (m)")
                fig.colorbar(img, ax=ax, shrink=0.7)
        fig.suptitle(f"{scenario} · persistent monitoring diagnostics · seed {seed}")
        fig.savefig(out / f"{scenario}_persistence.png", dpi=160)
        plt.close(fig)

        fig, axs = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
        for ax, d in zip(axs, (a, b)):
            values = np.maximum(
                np.asarray(d["environment"]["targets"]) - np.asarray(d["mission"]["q"]), 0
            )
            duration = (len(values) - 1) * d["config"]["dt"]
            img = ax.imshow(
                values.T,
                origin="lower",
                aspect="auto",
                extent=(0, duration, 0, len(target)),
                vmin=0,
                vmax=0.8,
                cmap="viridis",
                interpolation="nearest",
            )
            ax.set(
                title=f"{d['metrics']['method']} · all cells, fixed row-major grid order",
                xlabel="Mission time (s)",
                ylabel="Monitoring cell index",
            )
            fig.colorbar(img, ax=ax, label="Clarity deficit ↓", shrink=0.8)
        fig.suptitle(f"{scenario} · complete field evolution; no selected cells omitted")
        fig.savefig(out / f"{scenario}_clarity_evolution.png", dpi=160)
        plt.close(fig)
(out / "missions.json").write_text(json.dumps(rows, indent=2) + "\n")
(out / "raw_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
if catalog:
    old = json.loads((root / "demo/data/catalog.json").read_text())
    for r in old:
        if r["id"] not in {v["id"] for v in catalog}:
            if (
                not r.get("kind")
                and not r["id"].startswith(("field_", "persistent_"))
                and not r["label"].startswith("Legacy · ")
            ):
                r["label"] = "Legacy · " + r["label"]
            catalog.append(r)
    temporary = root / "demo/data/catalog.tmp"
    temporary.write_text(json.dumps(catalog, indent=2) + "\n")
    temporary.replace(root / "demo/data/catalog.json")
print(json.dumps(rows, indent=2))
