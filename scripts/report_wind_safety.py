"""Validate and publish the wind-in-RPCBF study, retaining every completed case."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/wind_safety"
REPORT = ROOT / "docs/wind_safety_validation"
protocol = json.loads((REPORT / "protocol.json").read_text())
rows, records, manifest = [], [], {}
for case in protocol["cases"]:
    scenario, original = case["scenario"], case["policy_seed"]
    for seed in [original, *case["held_out_seeds"]]:
        source = SOURCE / f"{scenario}_{seed}.json"
        if not source.exists():
            continue
        data = json.loads(source.read_text())
        if (
            data["ablation_script_sha256"]
            != protocol["source_sha256"]["scripts/wind_safety_ablation.py"]
        ):
            continue
        for file, sha in data["source_sha256"].items():
            assert protocol["source_sha256"][file] == sha
        assert data["recording_sha256"] == protocol["policy_sha256"][scenario]
        assert data["seed"] == seed and data["policy_seed"] == original
        a, b = data["wind_aware"], data["wind_ignored"]
        np.testing.assert_array_equal(a["mission"]["x"][0], b["mission"]["x"][0])
        np.testing.assert_array_equal(a["mission"]["q"][0], b["mission"]["q"][0])
        for arm in (a, b):
            m, metric = arm["mission"], arm["metrics"]
            q = np.asarray(m["q"])
            assert np.isfinite(q).all() and np.all((q >= 0) & (q <= 1))
            c = np.maximum(np.asarray(data["environment"]["targets"])[None] - q, 0).mean(axis=1)
            np.testing.assert_allclose(
                np.trapezoid(c, dx=data["config"]["dt"]), metric["integrated_deficit"], atol=2e-5
            )
            assert len(arm["beliefs"]) == data["config"]["mission_horizons"] + 1
            assert metric["collision"] == bool(np.min(m["dense_clearance"]) < 0)
            assert metric["integration_endpoint_max_discrepancy"] < 1e-4
        identity = f"safety_{scenario}_{seed}"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest[str(source.relative_to(ROOT))] = digest
        split = "positive-control" if seed == original else "held-out"
        row = {
            "id": identity,
            "scenario": scenario,
            "seed": seed,
            "policy_seed": original,
            "split": split,
            "wind_aware": a["metrics"],
            "wind_ignored": b["metrics"],
        }
        rows.append(row)
        dest = ROOT / f"demo/data/{identity}.json"
        dest.write_bytes(source.read_bytes())
        records.append(
            {
                "id": identity,
                "kind": "safety_ablation",
                "label": f"{scenario.removeprefix('field_')} · seed {seed} · {split}",
                "method": "safety_ablation",
                "seed": seed,
                "group": "wind-safety-ablation",
                "file": str(dest.relative_to(ROOT)),
                "live": False,
            }
        )
heldout = [r for r in rows if r["split"] == "held-out"]
expected = sum(len(c["held_out_seeds"]) for c in protocol["cases"])
helped = sum(not r["wind_aware"]["collision"] and r["wind_ignored"]["collision"] for r in heldout)
harmed = sum(r["wind_aware"]["collision"] and not r["wind_ignored"]["collision"] for r in heldout)
summary = {
    "completed_heldout": len(heldout),
    "expected_heldout": expected,
    "complete": len(heldout) == expected,
    "wind_aware_collisions": sum(r["wind_aware"]["collision"] for r in heldout),
    "wind_ignored_collisions": sum(r["wind_ignored"]["collision"] for r in heldout),
    "discordant_favoring_wind_awareness": helped,
    "discordant_against": harmed,
    "paired_exact_p": float(binomtest(helped, helped + harmed, 0.5).pvalue)
    if helped + harmed and len(heldout) == expected
    else None,
    "scope": protocol["type"],
    "cases": rows,
}
for name, value in [("summary", summary), ("raw_manifest", manifest)]:
    (REPORT / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
index = ROOT / "demo/data/wind_safety_index.json"
index.write_text(json.dumps(summary, separators=(",", ":")))
if heldout:
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    x = np.arange(len(heldout))
    for i, (name, color, label) in enumerate(
        [
            ("wind_aware", "#087f83", "Wind-aware RPCBF"),
            ("wind_ignored", "#c55545", "Wind ignored by RPCBF"),
        ]
    ):
        y = [r[name]["dense_minimum_clearance"] for r in heldout]
        ax.scatter(x + (i - 0.5) * 0.18, y, color=color, label=label, s=25)
    for k, row in enumerate(heldout):
        ax.plot(
            [k - 0.09, k + 0.09],
            [
                row["wind_aware"]["dense_minimum_clearance"],
                row["wind_ignored"]["dense_minimum_clearance"],
            ],
            color="#aab3ba",
            lw=0.8,
            zorder=0,
        )
    ax.axhline(0, color="#c55545", lw=1, ls="--")
    ax.set_xticks(x, [str(r["seed"]) for r in heldout], rotation=45)
    ax.set(
        xlabel="Independent physical wind seed (vortices / shear / mixed)",
        ylabel="Minimum physical clearance (m)",
        title=f"Fixed-policy safety ablation: all {len(heldout)} declared fresh wind seeds",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(REPORT / "clearance_all_seeds.png", dpi=180)
    plt.close(fig)
records.append(
    {
        "id": "wind_safety_index",
        "kind": "safety_index",
        "label": "Wind-in-RPCBF study",
        "method": "safety_ablation",
        "seed": 0,
        "group": "wind-safety-ablation",
        "file": str(index.relative_to(ROOT)),
        "live": False,
    }
)
old = json.loads((ROOT / "demo/data/catalog.json").read_text())
ids = {r["id"] for r in records}
catalog = [r for r in old if r["id"] not in ids] + records
tmp = ROOT / "demo/data/catalog.tmp"
tmp.write_text(json.dumps(catalog, indent=2) + "\n")
tmp.replace(ROOT / "demo/data/catalog.json")
print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))
