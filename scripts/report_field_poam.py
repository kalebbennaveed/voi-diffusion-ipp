"""Archive every completed paired whole-field run, check telemetry, and plot outcomes."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(".")
source = root / "results/field_poam/recordings"
out = root / "docs/field_poam_validation"
out.mkdir(parents=True, exist_ok=True)
protocol = json.loads((out / "protocol.json").read_text())
rows = []
catalog = []
manifest = {}
for scenario, seed in protocol["scenarios_and_seeds"]:
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
        targets = np.asarray(d["environment"]["targets"])
        deficit = np.maximum(targets[None] - q, 0).mean(1)
        cumulative = np.r_[0, np.cumsum(cfg["dt"] * (deficit[1:] + deficit[:-1]) / 2)]
        assert abs(cumulative[-1] - m["integrated_deficit"]) < 2e-5
        assert len(d["beliefs"]) == cfg["mission_horizons"] + 1
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
        rows.append(row)
        target = root / "demo/data" / f"{stem}.json"
        target.write_text(file.read_text())
        catalog.append(
            {
                "id": stem,
                "label": f"Full POAM · {scenario.removeprefix('field_')} · {method} · seed {seed}",
                "method": method,
                "seed": seed,
                "variant": "multistage",
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
(out / "missions.json").write_text(json.dumps(rows, indent=2) + "\n")
(out / "raw_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
if catalog:
    old = json.loads((root / "demo/data/catalog.json").read_text())
    for r in old:
        if r["id"] not in {v["id"] for v in catalog} and not r["id"].startswith("field_"):
            if not r["label"].startswith("Legacy · "):
                r["label"] = "Legacy · " + r["label"]
            catalog.append(r)
    (root / "demo/data/catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
print(json.dumps(rows, indent=2))
