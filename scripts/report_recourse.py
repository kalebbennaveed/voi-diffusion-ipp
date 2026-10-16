"""Archive every run in the frozen recourse study; refuse incomplete evidence."""

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(rows, suite):
    group = [r for r in rows if r["suite"] == suite]
    seeds = sorted({r["seed"] for r in group})
    records = {(r["method"], r["seed"]): r for r in group}
    pairs = []
    for seed in seeds:
        p, b = (records[(method, seed)] for method in ("proposed", "current_belief"))
        pairs.append(
            {
                "seed": seed,
                "proposed_cost": p["integrated_deficit"],
                "comparator_cost": b["integrated_deficit"],
                "benefit": b["integrated_deficit"] - p["integrated_deficit"],
                "initial_extra_cost": p["period_costs"][0] - b["period_costs"][0],
                "later_benefit": sum(b["period_costs"][1:]) - sum(p["period_costs"][1:]),
            }
        )
    benefit = np.array([p["benefit"] for p in pairs])
    rng = np.random.default_rng(2026)
    bootstrap = rng.choice(benefit, size=(10000, len(benefit)), replace=True).mean(axis=1)
    baseline = np.mean([p["comparator_cost"] for p in pairs])
    return {
        "suite": suite,
        "pairs": pairs,
        "mean_benefit": float(benefit.mean()),
        "relative_mean_reduction": float(benefit.mean() / baseline),
        "paired_percentile_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "wins": int((benefit > 1e-5).sum()),
        "losses": int((benefit < -1e-5).sum()),
        "ties": int((abs(benefit) <= 1e-5).sum()),
        "mean_initial_extra_cost": float(np.mean([p["initial_extra_cost"] for p in pairs])),
        "mean_later_benefit": float(np.mean([p["later_benefit"] for p in pairs])),
    }


def main():
    out = ROOT / "docs/recourse_validation"
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / "results/recourse_validation/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    for name, expected in protocol["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Numerical source changed after freeze: {name}")
    folders = {
        "recourse_pilot": 6,
        "recourse_validation": 20,
        "recourse_no_measurements": 2,
        **{
            f"recourse_ablations/{variant}": 3
            for variant in ("threshold_feedback", "task_openloop", "threshold_openloop")
        },
    }
    rows, hashes = [], {}
    for suite, expected in folders.items():
        found = []
        for path in sorted((ROOT / "results" / suite).glob("*.json")):
            r = json.loads(path.read_text())
            if not isinstance(r, dict) or "integrated_deficit" not in r:
                continue
            with np.load(path.with_suffix(".npz")) as a:
                deficit = np.maximum(a["targets"] - a["q"], 0).mean(axis=1)
                costs = r["config"]["dt"] * (deficit[:-1] + deficit[1:]) / 2
                r["period_costs"] = costs.reshape(-1, r["config"]["stage_steps"]).sum(1).tolist()
                np.testing.assert_allclose(
                    sum(r["period_costs"]), r["integrated_deficit"], atol=2e-5
                )
                r["executed_branches"] = a["branches"].tolist()
            r["suite"] = suite
            found.append(r)
            for file in (path, path.with_suffix(".npz")):
                hashes[str(file.relative_to(ROOT))] = digest(file)
        if len(found) != expected:
            raise ValueError(f"Incomplete {suite}: {len(found)} / {expected}")
        rows.extend(found)
    noobs = ROOT / "results/recourse_no_measurements"
    with (
        np.load(noobs / "localized_corridor_proposed_20.npz") as p,
        np.load(noobs / "localized_corridor_current_belief_20.npz") as b,
    ):
        for key in ("x", "q", "branches", "nominal", "control", "correction", "feasible"):
            np.testing.assert_array_equal(p[key], b[key], err_msg=f"No-observation control: {key}")
    groups = {}
    for r in rows:
        groups.setdefault((r["suite"], r["method"]), []).append(r)
    aggregates = []
    for (suite, method), group in sorted(groups.items()):
        costs = [r["integrated_deficit"] for r in group]
        aggregates.append(
            {
                "suite": suite,
                "method": method,
                "n": len(group),
                "cost_mean": float(np.mean(costs)),
                "cost_sd": float(np.std(costs, ddof=1)) if len(group) > 1 else None,
                "access": sum(r["roi_a_access"] for r in group),
                "collisions": sum(r["collision"] for r in group),
                "qp_infeasible": float(np.mean([r["qp_infeasible_fraction"] for r in group])),
                "fallbacks": sum(r["optimizer_failure_count"] for r in group),
                "plan_seconds": float(np.mean([r["mean_plan_seconds"] for r in group])),
            }
        )
    comparisons = [paired(rows, name) for name in ("recourse_pilot", "recourse_validation")]
    for name, data in (("missions", rows), ("aggregate", aggregates), ("paired", comparisons)):
        (out / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    columns = [
        "suite",
        "method",
        "seed",
        "integrated_deficit",
        "roi_a_access",
        "roi_a_time",
        "collision",
        "minimum_clearance",
        "qp_infeasible_fraction",
        "optimizer_failure_count",
        "mean_plan_seconds",
        "probe_seconds",
    ]
    with (out / "missions.csv").open("w") as file:
        writer = csv.DictWriter(file, columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for source, name in (
        (protocol_path, "protocol.json"),
        (ROOT / "results/tests_research_v2.log", "tests.txt"),
        (ROOT / "results/recourse_screen/screen.json", "development_screen.json"),
    ):
        shutil.copy2(source, out / name)
        hashes[str(source.relative_to(ROOT))] = digest(source)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "mission_count": len(rows),
                "raw_sha256": hashes,
                "frozen_source_verified": True,
                "no_measurements_exact_equality": True,
                "note": "All development and validation missions retained; timing includes concurrent host load.",
            },
            indent=2,
        )
        + "\n"
    )
    table = [
        "# Recourse study: complete evidence",
        "",
        "All 37 missions are included. Cost is actual integrated clarity deficit (lower is better).",
        "Development seeds: 20–22; frozen validation seeds: 100–109. These are now used validation data.",
        "Timing includes concurrent host load. No-measurement trajectories and controls match exactly.",
        "",
        "| Suite | Method | n | Mean cost | ROI access | Collisions | QP infeasible | Fallbacks | Plan s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in aggregates:
        table.append(
            f"| {r['suite']} | {r['method']} | {r['n']} | {r['cost_mean']:.4f} | "
            f"{r['access']}/{r['n']} | {r['collisions']} | {r['qp_infeasible']:.2%} | "
            f"{r['fallbacks']} | {r['plan_seconds']:.2f} |"
        )
    for comparison in comparisons:
        lo, hi = comparison["paired_percentile_bootstrap_95"]
        table += [
            "",
            f"## {comparison['suite']}",
            "",
            (
                f"Mean paired benefit {comparison['mean_benefit']:.4f} "
                f"({comparison['relative_mean_reduction']:.2%} relative mean cost reduction). "
                f"Paired percentile-bootstrap 95% interval: [{lo:.4f}, {hi:.4f}]. "
                "10,000 resamples, RNG seed 2026; small-sample descriptive uncertainty, not a guarantee."
            ),
            f"Wins / ties / losses: {comparison['wins']} / {comparison['ties']} / {comparison['losses']}.",
            "",
            "| Seed | Proposed | Frozen-belief comparator | Benefit | Initial extra cost | Later benefit |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in comparison["pairs"]:
            table.append(
                f"| {r['seed']} | {r['proposed_cost']:.4f} | {r['comparator_cost']:.4f} | "
                f"{r['benefit']:.4f} | {r['initial_extra_cost']:.4f} | {r['later_benefit']:.4f} |"
            )
    (out / "summary.md").write_text("\n".join(table) + "\n")
    validation = comparisons[1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    pairs = validation["pairs"]
    seeds = [p["seed"] for p in pairs]
    benefit = [p["benefit"] for p in pairs]
    axes[0].bar(
        range(len(pairs)), benefit, color=["#168575" if b > 0 else "#c95d4b" for b in benefit]
    )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(range(len(pairs)), seeds)
    axes[0].set(
        xlabel="Fresh validation seed",
        ylabel="Comparator cost − proposed cost",
        title="All ten paired outcomes",
    )
    for method, label, color in (
        ("proposed", "Anticipated learning", "#168575"),
        ("current_belief", "Frozen future belief", "#6a65a8"),
    ):
        costs = np.array(
            [
                r["period_costs"]
                for r in rows
                if r["suite"] == "recourse_validation" and r["method"] == method
            ]
        )
        axes[1].plot(np.arange(1, 6) * 2.4, costs.mean(0), "o-", color=color, label=label)
    axes[1].set(
        xlabel="Period end [s]",
        ylabel="Mean actual period cost",
        title="Initial cost and subsequent recovery",
    )
    axes[1].legend()
    fig.savefig(out / "paired_validation.png", dpi=170)
    plt.close(fig)
    print(json.dumps(comparisons, indent=2))
    print(f"Archived {len(rows)} missions in {out}")


if __name__ == "__main__":
    main()
