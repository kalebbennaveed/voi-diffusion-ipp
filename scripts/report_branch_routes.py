"""Audit the optional route-policy pair and retain its compact-policy comparison."""

import json

import report_wind_route_study as study


def main():
    study.OUT = study.ROOT / "docs/wind_route_study/branch_policy_results_01"
    study.OUT.mkdir(exist_ok=True)
    study.NAMES = {"branch_reversed": "Reversed wind · two-route policy"}
    study.inputs = lambda _: {
        method: study.ROOT / "results/budgeted_missions/reversed_branches_01" / (method + ".npz")
        for method in ["proposed", "current_belief"]
    }
    study.report()
    new = json.loads((study.OUT / "results_01.json").read_text())
    old = json.loads((study.ROOT / "docs/wind_route_study/results_01.json").read_text())
    new_rows = new["cases"]["branch_reversed"]["methods"]
    old_rows = old["cases"]["reversed"]["methods"]
    result = {
        "label": "Development policy-class comparison; not a held-out validation",
        "caveat": "24 versus 18 parameters changes MBD proposal draws; within each pair only anticipation differs.",
        "cost_reduction_vs_compact_percent": {
            m: 100
            * (
                1
                - r["integrated_whole_grid_deficit"] / old_rows[m]["integrated_whole_grid_deficit"]
            )
            for m, r in new_rows.items()
        },
    }
    (study.OUT / "policy_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
