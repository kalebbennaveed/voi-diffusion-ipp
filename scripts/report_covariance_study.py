"""Summarize all declared covariance diagnostics without choosing a winning method."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/covariance_study"


def main():
    protocol = json.loads((OUT / "budgets_01/protocol.json").read_text())
    if protocol["status"] != "complete":
        raise RuntimeError("Wait for the frozen budget study to finish")
    rows = json.loads((OUT / "budgets_01/results.json").read_text())["rows"]
    expected = np.prod([len(protocol[k]) for k in ("periods", "modes", "variants", "seeds")])
    assert len(rows) == expected
    summary = []
    for r in rows:
        cost = np.array(r["evaluation_bank"]["costs"])
        gap = np.array(r["evaluation_bank"]["clearance"])
        s = r["selection_index"]
        summary.append(
            {
                **{
                    k: r[k]
                    for k in [
                        "period",
                        "mode",
                        "variant",
                        "optimizer_seed",
                        "train_cost",
                        "optimizer_s",
                        "compile_s",
                        "objective_evaluations",
                    ]
                },
                "test_before": float(cost[0].mean()),
                "test_after": float(cost[s].mean()),
                "test_contacts_before": int((gap[0] < 0).sum()),
                "test_contacts_after": int((gap[s] < 0).sum()),
                "selection_changed": s != 0,
                "selection_feasible": r["selection_has_feasible_candidate"],
            }
        )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    variants = protocol["variants"]
    modes = protocol["modes"]
    for i, mode in enumerate(modes):
        for j, period in enumerate(protocol["periods"]):
            ax = axes[i, j]
            for seed, offset in zip(protocol["seeds"], [-0.08, 0.08], strict=True):
                rs = [
                    next(
                        r
                        for r in summary
                        if r["mode"] == mode
                        and r["period"] == period
                        and r["variant"] == v
                        and r["optimizer_seed"] == seed
                    )
                    for v in variants
                ]
                x = np.arange(len(rs)) + offset
                ax.plot(
                    x,
                    [r["train_cost"] for r in rs],
                    "o--",
                    color="#297d8c",
                    alpha=0.8,
                    label="Training objective" if offset < 0 else None,
                )
                ax.plot(
                    x,
                    [r["test_before"] for r in rs],
                    "s-",
                    color="#bf713a",
                    alpha=0.8,
                    label="Independent worlds" if offset < 0 else None,
                )
                ax.plot(
                    x,
                    [r["test_after"] for r in rs],
                    "x",
                    color="#9e4673",
                    label="After independent selection" if offset < 0 else None,
                )
            ax.set(title=f"{mode} · mission t={period * 2}s", ylabel="10s integrated deficit")
            ax.set_xticks(
                np.arange(len(variants)),
                ["4×8", "8×16", "16×32", "1×(4×7)", "3×(4×2)"],
                rotation=25,
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fixed-state optimization vs fresh-world performance · development diagnostic")
    fig.savefig(OUT / "budget_comparison.png", dpi=160)
    plt.close(fig)
    equal_budget = []
    expanded = []
    for mode in modes:
        for period in protocol["periods"]:
            for seed in protocol["seeds"]:
                lookup = {
                    r["variant"]: r
                    for r in summary
                    if r["mode"] == mode and r["period"] == period and r["optimizer_seed"] == seed
                }
                a, b = lookup["one_chain_4x7"], lookup["three_chains_4x2"]
                assert a["objective_evaluations"] == b["objective_evaluations"] == 37
                equal_budget.append(
                    {
                        "mode": mode,
                        "period": period,
                        "seed": seed,
                        "train_improvement": a["train_cost"] - b["train_cost"],
                        "test_improvement": a["test_before"] - b["test_before"],
                    }
                )
                a, b = lookup["4x8"], lookup["16x32"]
                expanded.append(
                    {
                        "mode": mode,
                        "period": period,
                        "seed": seed,
                        "train_improvement": a["train_cost"] - b["train_cost"],
                        "test_improvement": a["test_before"] - b["test_before"],
                    }
                )
    result = {
        "scope": "Three saved states, two optimizer seeds. Development diagnostics, not independent mission trials or calibration evidence. Independent banks still use the model posterior; same inner draws.",
        "rows": summary,
        "equal_budget_chain_pairs": equal_budget,
        "expanded_budget_pairs": expanded,
        "selection_changed": sum(r["selection_changed"] for r in summary),
        "selection_helped_test": sum(r["test_after"] < r["test_before"] - 1e-7 for r in summary),
        "selection_hurt_test": sum(r["test_after"] > r["test_before"] + 1e-7 for r in summary),
        "test_contacts_before": sum(r["test_contacts_before"] for r in summary),
        "test_contacts_after": sum(r["test_contacts_after"] for r in summary),
    }
    (OUT / "budget_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    mech = []
    for name in ["mechanism_02", "mechanism_03"]:
        mech += json.loads((OUT / name / "results.json").read_text())["rows"]
    assert len(mech) == 24
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, period in zip(axes, protocol["periods"], strict=True):
        rs = [r for r in mech if r["period"] == period]
        pred = np.array([r["frozen"][0] - r["covariance"][0] for r in rs])
        stress = np.array(
            [
                np.array(r["fresh_field_stress"])[:, 0, 0]
                - np.array(r["fresh_field_stress"])[:, 1, 0]
                for r in rs
            ]
        )
        ax.bar(np.arange(8) - 0.15, pred, width=0.3, label="Mean-wind forecast", color="#297d8c")
        ax.bar(
            np.arange(8) + 0.15,
            stress.mean(axis=1),
            width=0.3,
            label="8 fresh physical fields",
            color="#bf713a",
        )
        ax.axhline(0, color="black", lw=0.7)
        ax.set(
            title=f"Mission t={period * 2}s",
            xlabel="Candidate (7 = archived winner)",
            ylabel="Frozen cost − covariance cost",
        )
        ax.set_xticks(range(8))
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Covariance mechanism: improvement and regression both retained")
    fig.savefig(OUT / "mechanism_comparison.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["rows", "equal_budget_chain_pairs", "expanded_budget_pairs"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
