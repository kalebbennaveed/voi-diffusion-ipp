"""Archive all refinement attempts and the complete predeclared validation study."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from refine_multistage import core_hashes
from report_recourse import paired

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib.pyplot as plt

    protocol = json.loads((ROOT / "results/multistage_validation/protocol.json").read_text())
    assert protocol["core_sha256"] == core_hashes(), "Frozen numerical source changed"
    folders = {
        "multistage_dev": 6,
        "multistage_corrected": 6,
        "strict_qp_dev": 4,
        "multistage_failure_controls": 1,
        "strict_qp_failure_followup": 2,
        **{
            f"multistage_validation/{arm}": 10
            for arm in ("repeated_updates", "two_stage", "frozen_belief")
        },
    }
    out = ROOT / "docs/multistage_validation"
    out.mkdir(parents=True, exist_ok=True)
    rows, hashes = [], {}
    for suite, count in folders.items():
        found = []
        for file in sorted((ROOT / "results" / suite).glob("*.json")):
            r = json.loads(file.read_text())
            if not isinstance(r, dict) or "integrated_deficit" not in r:
                continue
            with np.load(file.with_suffix(".npz")) as a:
                deficit = np.maximum(a["targets"] - a["q"], 0).mean(1)
                costs = r["config"]["dt"] * (deficit[:-1] + deficit[1:]) / 2
                r["period_costs"] = costs.reshape(-1, r["config"]["stage_steps"]).sum(1).tolist()
                np.testing.assert_allclose(
                    sum(r["period_costs"]), r["integrated_deficit"], atol=2e-5
                )
                r["infeasible_qp_steps"] = int((~a["feasible"]).sum())
            r["suite"] = suite
            r["status"] = (
                "superseded comparator warm-start bug" if suite == "multistage_dev" else "complete"
            )
            found.append(r)
            for p in (file, file.with_suffix(".npz")):
                hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        if len(found) != count:
            raise ValueError(f"Incomplete {suite}: {len(found)} / {count}")
        rows.extend(found)
    comparisons = []
    for baseline in ("frozen_belief", "two_stage"):
        comparison_rows = []
        for r in rows:
            if r["suite"] in (
                "multistage_validation/repeated_updates",
                f"multistage_validation/{baseline}",
            ):
                comparison_rows.append(
                    {
                        **r,
                        "suite": baseline,
                        "method": "proposed"
                        if r["suite"].endswith("/repeated_updates")
                        else "current_belief",
                    }
                )
        comparisons.append(paired(comparison_rows, baseline))
    aggregates = []
    groups = sorted({(r["suite"], r["method"]) for r in rows})
    for suite, method in groups:
        rs = [r for r in rows if r["suite"] == suite and r["method"] == method]
        aggregates.append(
            {
                "suite": suite,
                "method": method,
                "n": len(rs),
                "mean_cost": float(np.mean([r["integrated_deficit"] for r in rs])),
                "cost_sd": float(np.std([r["integrated_deficit"] for r in rs], ddof=1))
                if len(rs) > 1
                else None,
                "access": sum(r["roi_a_access"] for r in rs),
                "collisions": sum(r["collision"] for r in rs),
                "infeasible_steps": sum(r["infeasible_qp_steps"] for r in rs),
                "fallbacks": sum(r["optimizer_failure_count"] for r in rs),
                "mean_plan_seconds": float(np.mean([r["mean_plan_seconds"] for r in rs])),
            }
        )
    for name, data in (
        ("missions", rows),
        ("aggregate", aggregates),
        ("paired", comparisons),
        ("protocol", protocol),
    ):
        (out / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    columns = [
        "suite",
        "status",
        "method",
        "seed",
        "integrated_deficit",
        "roi_a_access",
        "roi_a_time",
        "collision",
        "infeasible_qp_steps",
        "minimum_clearance",
        "max_qp_slack",
        "optimizer_failure_count",
        "mean_plan_seconds",
    ]
    with (out / "missions.csv").open("w") as file:
        w = csv.DictWriter(file, columns, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for source, destination in (
        ("results/tests_demo_final.log", "tests.txt"),
        ("results/demo_js_tests.log", "frontend_tests.txt"),
        ("results/multistage_failures.json", "failure_diagnostics.json"),
        ("results/multistage_sampling_diagnosis.json", "sampling_diagnostics.json"),
    ):
        shutil.copy2(ROOT / source, out / destination)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "mission_count": len(rows),
                "raw_sha256": hashes,
                "frozen_core_verified": True,
                "note": "All 49 refinement attempts retained, including six explicitly superseded warm-start runs. The 30 predeclared validation missions alone form the fresh-seed comparisons. Runtime includes host contention.",
            },
            indent=2,
        )
        + "\n"
    )
    table = [
        "# Complete repeated-update study",
        "",
        "49 missions retained. Six initial development runs are superseded because their comparator warm start anticipated future updates. They are excluded from all scientific comparisons below.",
        "",
        "| Suite | Method | n | Mean actual cost | ROI access | Collisions | Infeasible QP steps | Optimizer fallbacks | Plan s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in aggregates:
        table.append(
            f"| {a['suite']} | {a['method']} | {a['n']} | {a['mean_cost']:.4f} | {a['access']}/{a['n']} | {a['collisions']} | {a['infeasible_steps']} | {a['fallbacks']} | {a['mean_plan_seconds']:.2f} |"
        )
    for c in comparisons:
        table += [
            "",
            f"## Repeated updates versus {c['suite']}",
            "",
            f"Relative mean reduction: {c['relative_mean_reduction']:.2%}. Mean paired benefit: {c['mean_benefit']:.4f}. Paired bootstrap 95% interval: {c['paired_percentile_bootstrap_95']}. Wins / ties / losses: {c['wins']} / {c['ties']} / {c['losses']}.",
            "",
            "| Seed | Repeated updates | Comparator | Benefit | Initial extra cost | Later benefit |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in c["pairs"]:
            table.append(
                f"| {r['seed']} | {r['proposed_cost']:.4f} | {r['comparator_cost']:.4f} | {r['benefit']:.4f} | {r['initial_extra_cost']:.4f} | {r['later_benefit']:.4f} |"
            )
    table += [
        "",
        "Positive benefit means lower actual monitoring cost with repeated updates. Intervals use 10,000 paired percentile-bootstrap resamples, RNG seed 2026. They describe a small sample in a selected scenario family, not broad generalization or a safety guarantee.",
    ]
    (out / "summary.md").write_text("\n".join(table) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, c in zip(axes[:2], comparisons):
        benefit = [p["benefit"] for p in c["pairs"]]
        ax.bar(range(10), benefit, color=["#087f83" if b >= 0 else "#bb514c" for b in benefit])
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(range(10), [p["seed"] for p in c["pairs"]], rotation=45)
        ax.set(
            title=f"Versus {c['suite'].replace('_', ' ')}",
            ylabel="Comparator cost − repeated-update cost",
            xlabel="Fresh seed",
        )
    vals = [
        next(a for a in aggregates if a["suite"] == f"multistage_validation/{arm}")
        for arm in ("frozen_belief", "two_stage", "repeated_updates")
    ]
    axes[2].bar(
        range(3), [v["infeasible_steps"] for v in vals], color=["#b27738", "#8171b0", "#087f83"]
    )
    axes[2].set_xticks(
        range(3), ["Frozen belief", "Two-stage", "Repeated updates"], rotation=30, ha="right"
    )
    axes[2].set(title="Safety-filter relaxation", ylabel="Infeasible QP steps / 800 executed steps")
    fig.savefig(out / "paired_results.png", dpi=160)
    plt.close(fig)
    print(json.dumps(comparisons, indent=2))
    print(f"Archived {len(rows)} missions")


if __name__ == "__main__":
    main()
