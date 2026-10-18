"""Report every declared persistent covariance comparison, including contacts."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "proposed": "Realization anticipation",
    "current_belief": "Wind-aware frozen",
    "covariance_anticipated": "Covariance anticipation",
    "mean_frozen": "Mean-dynamics frozen",
    "wind_blind_matched": "Wind-blind matched",
}


def verify_clarity_numpy(raw, meta):
    """Independent float64 whole-grid integration along archived executed motion."""
    cfg = meta["config"]
    assert cfg["scenario"] == "two_region_shuttle"
    recorded_q = np.array(raw["q"], dtype=float)
    q = recorded_q[0].copy()
    cells = np.array(raw["cells"], dtype=float)
    positions = np.array(raw["x"][:, :2], dtype=float)
    dt, decay = meta["execution_dt"], cfg["clarity_decay"]
    errors = []
    for i in range(len(positions) - 1):
        p0, p1 = positions[i : i + 2]
        rates = [
            np.exp(-np.sum((cells - p) ** 2, axis=1) / cfg["sensor_sigma"] ** 2) / cfg["sensor_R"]
            for p in [p0, (p0 + p1) / 2, p1]
        ]
        f = lambda value, a: a * (1 - value) ** 2 - decay * value**2
        k1 = f(q, rates[0])
        k2 = f(q + dt * k1 / 2, rates[1])
        k3 = f(q + dt * k2 / 2, rates[1])
        k4 = f(q + dt * k3, rates[2])
        q = np.clip(q + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, 0, 1)
        errors.append(float(np.max(np.abs(q - recorded_q[i + 1]))))
    error = max(errors)
    assert error < 2e-5, error
    no_sensing = recorded_q[0] / (1 + decay * recorded_q[0] * (len(positions) - 1) * dt)
    assert np.all(q >= no_sensing - 2e-5)
    return {
        "max_global_clarity_reconstruction_error": error,
        "uniform_Q": decay,
        "no_sensing_final_clarity_range": [float(no_sensing.min()), float(no_sensing.max())],
        "interpretation": "Same archived 0.05s segment quadrature, independently implemented in NumPy float64; no JAX clarity import. No-sensing value is the analytic Q-only counterfactual, not a new mission.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("protocol", type=Path)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    status = json.loads((ROOT / protocol["status"]).read_text())
    if status["state"] != "complete":
        raise RuntimeError("Do not silently omit incomplete declared missions")
    out = args.protocol.parent
    rows = []
    paths = {}
    for job in protocol["jobs"]:
        for method in job["methods"]:
            path = ROOT / job["output"] / f"{method}.npz"
            doc = json.loads(path.with_suffix(".json").read_text())
            audit = json.loads(path.with_suffix(".audit.json").read_text())
            assert audit["complete"] and abs(audit["duration_s"] - 120) < 1e-8
            assert audit["recorded_cost_disagreement"] < 2e-5
            assert audit["max_local_state_error"] < 1e-4
            seed = doc["metadata"]["seed"]
            paths[seed, method] = path
            raw = np.load(path, allow_pickle=False)
            candidates = np.array([r["candidates"] for r in doc["periods"]])
            weights = np.array([r["weights"] for r in doc["periods"]])
            outside = (
                np.any(np.abs(candidates) > meta_bound, axis=-1)
                if (meta_bound := doc["metadata"]["config"]["mbd_latent_bound"])
                else np.zeros(weights.shape, dtype=bool)
            )
            assert np.all(weights[outside] == 0)
            assert np.max(np.abs([r["vector"] for r in doc["periods"]])) <= meta_bound
            gaps = [
                abs(r["cached_objective"] - r["full_objective"])
                for r in doc["periods"]
                if r["cached_objective"] is not None and r["full_objective"] is not None
            ]
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "cost": audit["integrated_whole_grid_deficit"],
                    "contact_intervals": audit["contact_intervals"],
                    "clearance": audit["minimum_dense_clearance"],
                    "transfers": len(audit["transfers"]),
                    "directional_routes": audit["directional_routes"],
                    "backup_periods": audit["backup_periods"],
                    "cycle_median_s": audit["steady_cycle_s"]["median"],
                    "cycle_p95_s": audit["steady_cycle_s"]["p95"],
                    "cycle_2s_miss_fraction": audit["cycle_2s_miss_fraction"],
                    "startup_cycles_s": audit["startup_cycle_s"],
                    "initial_poam_preparation_s": doc["metadata"]["initial_poam_update_s"]
                    + doc["metadata"]["initial_snapshot_s"],
                    "simulated_execution_compute_median_s": float(
                        np.median([r["execution_s"] for r in doc["periods"][2:]])
                    ),
                    "mean_control_correction": float(raw["correction"].mean()),
                    "max_qp_slack": float(raw["slack"].max()),
                    "passage_rmse_initial_final": audit["passage_rmse_initial_final"],
                    "passage_std_initial_final": audit["passage_std_initial_final"],
                    "clarity_audit": verify_clarity_numpy(raw, doc["metadata"]),
                    "proposals_outside_target": int(outside.sum()),
                    "total_proposals": int(outside.size),
                    "selected_cached_full_cost_gap_median_max": [
                        float(np.median(gaps)),
                        float(np.max(gaps)),
                    ],
                    "initial_candidate_selected_periods": sum(
                        r["vector"] == r["initial_vector"] for r in doc["periods"]
                    ),
                }
            )
    comparisons = []
    for seed in sorted({r["seed"] for r in rows}):
        by = {r["method"]: r for r in rows if r["seed"] == seed}
        for a, b in [
            ("covariance_anticipated", "mean_frozen"),
            ("proposed", "current_belief"),
            ("covariance_anticipated", "proposed"),
            ("proposed", "wind_blind_matched"),
            ("current_belief", "wind_blind_matched"),
            ("covariance_anticipated", "wind_blind_matched"),
        ]:
            if a not in by or b not in by:
                continue
            comparisons.append(
                {
                    "seed": seed,
                    "method": a,
                    "comparator": b,
                    "cost_reduction_percent": 100 * (by[b]["cost"] - by[a]["cost"]) / by[b]["cost"],
                    "both_contact_free": by[a]["contact_intervals"]
                    == by[b]["contact_intervals"]
                    == 0,
                }
            )
    summary = {
        "scope": "Three development seeds with fixed original physical wind. No held-out method-selection claim, no confidence interval or universal safety assertion. Post-contact cost is raw cost, not successful monitoring.",
        "rows": rows,
        "comparisons": comparisons,
    }
    summary["aggregate"] = {
        method: {
            "mean_cost": float(np.mean([r["cost"] for r in rows if r["method"] == method])),
            "mean_of_cycle_medians_s": float(
                np.mean([r["cycle_median_s"] for r in rows if r["method"] == method])
            ),
            "mean_2s_miss_fraction": float(
                np.mean([r["cycle_2s_miss_fraction"] for r in rows if r["method"] == method])
            ),
            "directional_routes": {
                direction: {
                    "transfers": sum(
                        r["directional_routes"][direction]["count"]
                        for r in rows
                        if r["method"] == method
                    ),
                    "middle": sum(
                        r["directional_routes"][direction]["route_counts"]["middle"]
                        for r in rows
                        if r["method"] == method
                    ),
                }
                for direction in ["A_to_B", "B_to_A"]
            },
        }
        for method in LABELS
    }
    (out / "mission_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Covariance-anticipated MBD + RPCBF: development results",
        "",
        "The separate covariance planner is implemented and tested. These are development comparisons, not a general learning-benefit or safety validation. Original scenarios, methods, recordings and 3D demo are preserved. No information reward was introduced.",
        "",
        "## What the completed study establishes",
        "",
        "The covariance variant does **not** improve monitoring in this study: it costs 0.94–4.44% more than its matched mean-dynamics frozen control, in all three seeds (mean paired regret 2.90%). It makes 10–12 A/B transfers versus 13–17 for that control. Although covariance reduction changes RPCBF, that mechanism alone is insufficient to improve this compact policy's persistent mission.",
        "",
        "The retained realization-anticipated method has 3.81–12.93% lower cost than wind-aware frozen (mean paired reduction 7.59%), and 9.76–14.30% lower cost than wind-blind matched (mean paired reduction 12.69%). Wind-aware frozen itself beats wind-blind matched by 1.57–8.50%. These are all three declared development seeds with **one shared physical wind field**, not independent environment trials or held-out statistical confirmation.",
        "",
        "Covariance anticipation averages 1.84s across the three per-mission steady-cycle medians, versus 2.05s for realization anticipation. Their two-second miss fractions across equally sized missions are 19.0% and 75.9%; covariance p95 is 2.16–2.35s, realization p95 2.33–2.51s. This is a modest latency reduction, not the hoped-for large speedup or a hard real-time guarantee. Mean-dynamics frozen is 0.58s and wind-aware frozen 1.08s by the same median summary. Startup and real POAM preparation are separate below/in the numeric report.",
        "",
        "All fifteen missions are contact-free under the independent dense physical checks. This does not negate the contacts in the separate independent-posterior-field mechanism stress test or establish general safety. Independent NumPy float64 whole-grid clarity reconstruction differs by at most 5.37e-7; unobserved clarity decays under Q=0.16 and regional recovery follows revisits.",
        "",
        "Decision: retain the new mode as an experimental approximation, keep the original realization method as the reference, and do not run an expensive confirmation sweep or promote extra search based on these development results. The remaining research questions are predictive-GP versus future-POAM consistency, the mean-wind dynamics approximation, and continuation-policy expressivity. Improve those in separately declared tests before new persistent missions; do not tune against held-out wind or add an uncertainty reward.",
        "",
        "## Persistent missions",
        "",
        "Fifteen 120-second missions, three sensor/optimizer seeds, the same original physical wind. Every method uses the proper finite-domain target, 10-second forecast, two-second execution, 4×8 MBD, full real POAM and the same wind-aware execution controller/RPCBF. The primary isolated covariance comparison is against `mean_frozen`. Comparing to the original realization method also changes physical forecast assumptions.",
        "",
        "All controls were independently reintegrated with 32 substeps per execution interval. Physical contact is separate from QP slack. A raw cost after contact cannot be interpreted as successful monitoring. First two planning cycles are excluded from steady timing and saved separately in mission_summary.json. Timing was measured during a continuous serial development study on the M3; brief browser QA overlapped the first seed, and no other heavy JAX worker ran.",
        "",
        "| Seed | Method | Deficit cost ↓ | Contacts (intervals) | Transfers | Median / p95 cycle (s) | >2s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {LABELS[r['method']]} | {r['cost']:.4f} | {r['contact_intervals']} | {r['transfers']} | {r['cycle_median_s']:.3f} / {r['cycle_p95_s']:.3f} | {100 * r['cycle_2s_miss_fraction']:.1f}% |"
        )
    lines += [
        "",
        "![All mission costs and timing](mission_comparison.png)",
        "",
        "## Paired task comparisons",
        "",
        "Positive percentages mean lower raw monitoring cost. Every pairing is listed; no best seed is selected. Three seeds do not establish the requested held-out confidence targets.",
        "",
        "| Seed | Method | Comparator | Cost reduction | Both contact-free |",
        "|---|---|---|---:|---|",
    ]
    for r in comparisons:
        lines.append(
            f"| {r['seed']} | {r['method']} | {r['comparator']} | {r['cost_reduction_percent']:+.2f}% | {r['both_contact_free']} |"
        )
    lines += [
        "",
        "![First declared seed: execution and regional clarity](persistent_execution.png)",
        "",
        "## What the robot actually does",
        "",
        "The physical jet points A→B. Across the three missions, the original realization planner uses the middle passage for 7/23 A→B transfers but only 2/22 return transfers. Wind-blind planning uses it for 12/19 outbound and 11/18 return transfers. This supports the intended direction-dependent route behavior, without claiming the selected route is globally optimal. Route labels use the recorded crossing nearest the separating midplane; the exact thresholds and all individual transfers are in the [dense audit archives](mission_audits/) and summary JSON.",
        "",
        "The covariance variant uses the middle for 9/17 A→B transfers and 0/17 returns, yet its overall monitoring cost is worse than the matched control. Favoring the wind-assisted passage more often is not by itself evidence of improved persistent monitoring. Wind-blind execution also acquires accurate passage estimates while repeatedly trying the middle; its final passage component-RMSE is 0.114–0.143 versus 0.123–0.159 for realization anticipation, despite higher task cost. Learning wind accurately and using that knowledge effectively for the task are distinct.",
        "",
        "## Mechanism and search diagnostics",
        "",
        "Twenty-four fixed policies at three saved states were evaluated with and without covariance conditioning. First-stage motion was identical; later RPCBF interventions and monitoring costs changed in both directions. The best candidate in the common bank did not change at any of these three states. Central-probe covariance changed little even when local filter interventions changed, exposing a limitation of the compact continuation policy. Full predictive spatial residuals were retained. Conditioning the current POAM predictive GP is not identical to future variational refitting/relocation or kernel training.",
        "",
        "![All mechanism outcomes](mechanism_comparison.png)",
        "",
        "Sixty optimizer calls compare 4×8, 8×16, 16×32 and equal-37-evaluation single/multiple chains. Training worlds are fixed; shortlist selection uses four fresh worlds and final reporting eight additional worlds. The independent evaluator includes causal random posterior-mean updates. These are posterior-model checks, not independent evidence of physical calibration.",
        "",
        "The 16×32 setting improved fresh-world mean cost in 8/12 paired comparisons (four initial-state ties), although training cost slightly regressed in two. Median optimizer-only time was 19.59s for realization and 18.46s for covariance, versus 1.48s and 1.35s at 4×8. Multiple chains helped at one state and hurt at another. Independent reranking changed 17 selections: the separate test bank improved in seven and regressed in ten; contact incidences across these repeated checks changed from four to two. These are correlated candidate checks, not independent safety trials. Neither larger search nor reranking was promoted to the online planner.",
        "",
        "![All search comparisons](budget_comparison.png)",
        "",
        "## Forecast versus real-estimator consistency",
        "",
        "A post-hoc check at declared periods 0, 5 and 18 conditions every forecast operator on identical actual next-period positions, giving it perfectly known locations for this diagnostic only. With zero-innovation measurements and frozen kernel, full predictive-GP conditioning reduces mean grid variance by 0.00290 / 0.00296 / 0.00229, while a relocated POAM variational refit reduces it by 0.00174 / 0.00131 / 0.00019. The refit can also move the mean even for those pseudo-values, because sparse variational projection is not exact Gaussian conditioning of the current predictive process.",
        "",
        "The actual next online POAM update, including real values and kernel training, has lower spatially averaged variance than the GP forecast in these three snapshots, yet 19.6–37.8% of cells remain more uncertain than the forecast. The mismatch is spatially mixed, not uniform optimism. This identifies a modeling limitation; it does not prove the cause of the mission regression, certify calibration, or justify discarding residual covariance. See [declared operator comparison](estimator_consistency_01/protocol.json) and [all results](estimator_consistency_01/results.json).",
        "",
        "## Reproducibility and limits",
        "",
        "36 component/regression tests pass, including full Gaussian conditioning, residual/cross-covariance preservation, conditional sample moments/derivatives, causal rollout boundaries, bounded target weights against quadrature, and the unchanged original schedule. Original unbounded first-12-second states, controls, clarity and POAM maps reproduce exactly. Initial failed diagnostics and the repaired test failure are retained in the worklog and results logs.",
        "",
        "The covariance approximation freezes expected mean, kernel and inducing representation in planning. It cannot anticipate discoveries of strongly favorable/adverse wind or all observation-dependent route changes. Full real online POAM still learns those quantities. Finite inner RPCBF samples do not certify safety; covariance prediction does not establish posterior calibration. Finite warm-started MBD is an optimizer approximation, not an exact trajectory-distribution sampler.",
        "",
        "Replay verification checks all 900 selected policy vectors, weights, original uncached final curves and all real execution/POAM snapshots against their numeric archives. Cached candidate curves can differ from that final check: the worst selected-policy path discrepancy is 0.959m at the ten-second forecast endpoint (realization, seed 985201, period 6), growing from 0.00358m within the first two seconds. Its cached/full costs are 0.25343/0.25211. For the new covariance variant the largest discrepancy is 0.0918m. These forecast sensitivities are retained, not silently replaced by smoothed animation, and motivate an interpolation/sensitivity check before stronger optimizer-quality claims. They are not physical execution errors. See [all export checks](export_verification.json) and [every mission's worst cache discrepancy](cache_path_diagnostic.json).",
        "",
        "Exact protocols: [persistent missions](persistent_protocol_01.json), [optimizer study](budgets_01/protocol.json). All numeric summaries: [missions](mission_summary.json), [budgets](budget_summary.json), [mechanism](mechanism_summary.json). Implementation and equations: [method note](../covariance_anticipated_method.md). Raw mission checkpoints, complete denoising vectors and observations remain under results/budgeted_missions/covariance_mission_01_* inside this repository.",
        "",
        "Interactive side-by-side execution, whole-grid clarity, regional decay/recovery, POAM wind/error and genuine MBD denoising: [new 2D covariance study](http://127.0.0.1:8766/covariance/). Start the local server with `JAX_PLATFORMS=cpu uv run python -m voi_diffusion_ipp.viz.demo_server --port 8766`. Run selectors pair identical configurations and seeds; all five methods and all three seeds are retained. Original 2D/3D demos remain available at their original paths.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    methods = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for i, m in enumerate(methods):
        selected = [r for r in rows if r["method"] == m]
        for j, row in enumerate(selected):
            x = i + (j - 1) * 0.13
            marker = "x" if row["contact_intervals"] else "o"
            axes[0].scatter(
                x,
                row["cost"],
                marker=marker,
                color=f"C{j}",
                label=f"Seed {row['seed']}" if i == 0 else None,
            )
            axes[1].scatter(x, row["cycle_median_s"], marker="o", color=f"C{j}")
            axes[1].plot([x, x], [row["cycle_median_s"], row["cycle_p95_s"]], color=f"C{j}")
    for ax in axes:
        ax.set_xticks(range(5), [LABELS[m].replace(" ", "\n", 1) for m in methods], fontsize=9)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set(
        title="120s monitoring cost · all declared seeds",
        ylabel="Integrated whole-grid clarity deficit",
    )
    axes[1].set(
        title="Full cycle latency · median to p95", ylabel="Seconds (startup reported separately)"
    )
    axes[1].axhline(2, ls="--", color="black", lw=1, label="2s soft budget")
    axes[0].legend(fontsize=8)
    axes[1].legend()
    fig.savefig(out / "mission_comparison.png", dpi=160)
    plt.close(fig)
    seed = protocol["seeds"][0]
    fig, axes = plt.subplots(5, 2, figsize=(11, 14), layout="constrained")
    for i, m in enumerate(methods):
        path = paths[seed, m]
        d = np.load(path, allow_pickle=False)
        meta = json.loads(path.with_suffix(".json").read_text())["metadata"]
        ax, aq = axes[i]
        ax.plot(d["x"][:, 0], d["x"][:, 1], lw=0.8, color=f"C{i}")
        for ob in d["obstacles"]:
            ax.add_patch(plt.Circle(ob[:2], ob[2], color="#677980"))
        spec = meta["task_spec"]
        t = np.arange(len(d["q"])) * meta["execution_dt"]
        for region, color in [("a", "#287f8d"), ("b", "#a25b90")]:
            center = np.array(spec["region_" + region])
            size = np.array(spec["region_size"])
            mask = np.all(np.abs(d["cells"] - center) <= size / 2, axis=1)
            aq.plot(t, d["q"][:, mask].mean(axis=1), color=color, lw=0.9, label=region.upper())
            ax.add_patch(plt.Rectangle(center - size / 2, *size, fill=False, edgecolor=color))
        aq.set(ylim=(0, 1), xlim=(0, 120), ylabel="Regional mean clarity", xlabel="Mission seconds")
        aq.legend(fontsize=8)
        aq.grid(alpha=0.2)
        ax.set(title=LABELS[m], xlim=d["bounds"][:, 0], ylim=d["bounds"][:, 1], aspect="equal")
    fig.suptitle(f"Original physical wind · development seed {seed} · no forced alternation")
    fig.savefig(out / "persistent_execution.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
