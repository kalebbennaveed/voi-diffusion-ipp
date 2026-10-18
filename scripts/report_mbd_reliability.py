"""Summarize the declared saved-state study without selecting a winning method."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(protocol_path):
    protocol = json.loads(protocol_path.read_text())
    folder = Path(protocol["output"])
    variants = [v[0] for v in protocol["variants"]]
    records, groups, comparisons, evidence = [], [], [], {}
    for job in protocol["jobs"]:
        path = folder / (job["name"] + ".json")
        data = json.loads(path.read_text())
        assert data["complete"], path
        assert len(data["optimization"]) == len(variants) * 2 * len(protocol["optimizer_seeds"])
        assert sorted(e["bank"] for e in data["evaluation"]) == list(range(protocol["evaluation_banks"]))
        evidence[str(path)] = digest(path)
        all_costs, all_gaps = {}, {}
        for label in ("fixed_inner", "fresh_inner"):
            costs, gaps = [], []
            for evaluation in sorted(data["evaluation"], key=lambda e: e["bank"]):
                array_path = folder / f'{job["name"]}_test_{evaluation["bank"]}_{label}.npz'
                with np.load(array_path) as arrays:
                    cost, gap = arrays["cost"], arrays["clearance"]
                    np.testing.assert_allclose(cost, evaluation["metrics"][label]["cost"], atol=0, rtol=0)
                    np.testing.assert_allclose(gap, evaluation["metrics"][label]["clearance"], atol=0, rtol=0)
                    assert cost.shape == gap.shape == (len(data["optimization"]), 4)
                    assert np.isfinite(cost).all() and np.isfinite(gap).all()
                    costs.append(cost)
                    gaps.append(gap)
                evidence[str(array_path)] = digest(array_path)
            all_costs[label] = np.concatenate(costs, axis=1)
            all_gaps[label] = np.concatenate(gaps, axis=1)
        for i, row in enumerate(data["optimization"]):
            record = {k: v for k, v in row.items() if k not in ("vector", "seed_costs")}
            record["job"] = job["name"]
            for label in all_costs:
                record[label] = {
                    "raw_mean_cost": float(all_costs[label][i].mean()),
                    "contact_count": int((all_gaps[label][i] < 0).sum()),
                    "worlds": int(all_gaps[label].shape[1]),
                    "minimum_clearance": float(all_gaps[label][i].min()),
                }
            records.append(record)
        for variant in variants:
            for anticipate in (True, False):
                rows = [r for r in records if r["job"] == job["name"] and r["variant"] == variant and r["anticipate"] == anticipate]
                group = {"job": job["name"], "variant": variant, "anticipate": anticipate,
                         "train_cost": float(np.mean([r["train_cost"] for r in rows])),
                         "optimizer_s": float(np.mean([r["optimizer_s"] for r in rows])),
                         "objective_evaluations": rows[0]["objective_evaluations"],
                         "outer_rollout_evaluations": rows[0]["outer_rollout_evaluations"]}
                for label in all_costs:
                    group[label] = {"raw_mean_cost": float(np.mean([r[label]["raw_mean_cost"] for r in rows])),
                                    "contact_exposures": sum(r[label]["contact_count"] for r in rows),
                                    "policy_world_exposures": sum(r[label]["worlds"] for r in rows)}
                groups.append(group)
            paired = []
            for seed in protocol["optimizer_seeds"]:
                pair = {r["anticipate"]: r for r in records if r["job"] == job["name"] and r["variant"] == variant and r["seed"] == seed}
                paired.append({"seed": seed, **{
                    label: {"raw_gain_percent": 100 * (pair[False][label]["raw_mean_cost"] - pair[True][label]["raw_mean_cost"]) / pair[False][label]["raw_mean_cost"],
                            "anticipated_contacts": pair[True][label]["contact_count"],
                            "frozen_contacts": pair[False][label]["contact_count"]}
                    for label in all_costs}})
            comparisons.append({"job": job["name"], "variant": variant, "paired_seeds": paired})
    summary = {"complete": True, "protocol": str(protocol_path), "protocol_sha256": digest(protocol_path),
               "evidence_sha256": evidence, "records": records, "groups": groups, "anticipation_comparisons": comparisons,
               "interpretation": [
                   "Three saved development states and two optimizer seeds; descriptive results, not mission-level confirmation.",
                   "16 independent posterior physical worlds per policy and inner setting; repeated policies and shared worlds are paired, not independent replicates.",
                   "Raw cost includes post-contact motion and must not be interpreted as successful monitoring when contact occurs.",
                   "All selected policies receive causal hypothetical updates at evaluation; frozen denotes only the planning objective.",
                   "Test banks never rerank policies. More training worlds changes the sampled contact-admissibility set as well as the sample average.",
                   "Optimizer-only synchronized CPU timings exclude cache construction, real POAM, verification and execution; cold compilation is recorded separately.",
               ]}
    return summary, variants


def plot(summary, variants, destination):
    jobs = list(dict.fromkeys(g["job"] for g in summary["groups"]))
    fig, axes = plt.subplots(3, len(jobs), figsize=(15, 10), constrained_layout=True)
    x = np.arange(len(variants))
    labels = [v.replace("_w", "\nworlds=").replace("x", "×") for v in variants]
    labels[-2:] = ["1 chain\n4×7", "3 chains\n4×2"]
    for col, job in enumerate(jobs):
        for anticipate, color, name in ((True, "#126f87", "Anticipated"), (False, "#b55b2b", "Frozen planning")):
            groups = [next(g for g in summary["groups"] if g["job"] == job and g["variant"] == v and g["anticipate"] == anticipate) for v in variants]
            axes[0, col].plot(x, [g["train_cost"] for g in groups], "--", color=color, alpha=.5)
            axes[0, col].plot(x, [g["fresh_inner"]["raw_mean_cost"] for g in groups], "o-", color=color, label=name)
            axes[1, col].plot(x, [g["optimizer_s"] for g in groups], "o-", color=color)
            axes[2, col].bar(x + (-.18 if anticipate else .18), [g["fresh_inner"]["contact_exposures"] for g in groups], width=.35, color=color)
        axes[0, col].set_title(job.replace("_", " ").title())
        axes[0, col].set_ylabel("Raw monitoring cost")
        axes[1, col].set_ylabel("Optimizer only (seconds, log scale)")
        axes[1, col].set_yscale("log")
        axes[1, col].axhline(2, color="0.5", ls=":", lw=1)
        axes[2, col].set_ylabel("Contact exposures (out of 32)")
        axes[2, col].set_ylim(0, 3)
        axes[2, col].set_yticks([0, 1, 2, 3])
        for row in range(3):
            axes[row, col].set_xticks(x, labels, fontsize=8)
            axes[row, col].grid(axis="y", alpha=.2)
            axes[row, col].spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("MBD–RPCBF saved-state diagnosis • two optimizer seeds • 16 fresh wind worlds per policy\nSolid: independent evaluation. Dashed: training objective. Raw costs include contacts; no test-bank selection.", fontsize=12)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    summary, variants = summarize(args.protocol)
    folder = args.protocol.parent
    (folder / "search_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(summary, variants, folder / "search_summary.png")
    for row in summary["groups"]:
        test = row["fresh_inner"]
        print(row["job"], row["variant"], row["anticipate"], f'train={row["train_cost"]:.6f}',
              f'test={test["raw_mean_cost"]:.6f}', f'contacts={test["contact_exposures"]}/32',
              f's={row["optimizer_s"]:.3f}')


if __name__ == "__main__":
    main()
