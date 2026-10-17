"""Declared paired regret analysis, with every seed and independent audits retained."""

import hashlib
import json
from pathlib import Path

import numpy as np
from report_wind_route_study import clarity_check, matched
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]


def main():
    protocol = json.loads(
        (ROOT / "docs/wind_route_study/low_value_confirmation_protocol_01.json").read_text()
    )
    rows = []
    manifest = {}
    for job in protocol["jobs"]:
        paths = {m: ROOT / job["output"] / (m + ".npz") for m in ["proposed", "current_belief"]}
        assert matched(paths)
        methods = {}
        for method, path in paths.items():
            raw = json.loads(path.with_suffix(".json").read_text())
            assert raw["complete"] and raw["metadata"]["seed"] == job["seed"]
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            a = json.loads(path.with_suffix(".audit.json").read_text())
            error = clarity_check(data, raw["metadata"]["config"], raw["metadata"]["execution_dt"])
            assert error < 2e-5
            methods[method] = a | {"clarity_replay_max_error": error}
            for file in [path, path.with_suffix(".json"), path.with_suffix(".audit.json")]:
                manifest[str(file.relative_to(ROOT))] = hashlib.sha256(
                    file.read_bytes()
                ).hexdigest()
        cp, cb = [
            methods[m]["integrated_whole_grid_deficit"] for m in ["proposed", "current_belief"]
        ]
        rows.append(
            {"seed": job["seed"], "paired_regret_percent": 100 * (cp / cb - 1), "methods": methods}
        )
    values = np.array([r["paired_regret_percent"] for r in rows])
    half = t.ppf(0.95, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    interval = [float(values.mean() - half), float(values.mean() + half)]
    no_contact = all(m["contact_intervals"] == 0 for r in rows for m in r["methods"].values())
    result = {
        "label": "Prospectively declared fresh-seed comparison in one fixed low-value environment",
        "n": len(values),
        "mean_paired_regret_percent": float(values.mean()),
        "paired_t_90_percent_interval": interval,
        "one_sided_95_percent_upper_regret": interval[1],
        "noninferiority_margin_percent": protocol["margin_percent"],
        "noninferiority_passed": interval[1] < protocol["margin_percent"] and no_contact,
        "equivalence_within_plus_minus_5_percent": interval[0] > -5
        and interval[1] < 5
        and no_contact,
        "all_missions_no_dense_contact": no_contact,
        "rows": rows,
        "input_sha256": manifest,
        "scope": "Seed variation covers observation and optimizer randomness for this fixed physical field/task/prior, not a population of environments. Student-t interval assumes independent paired seeds and approximately normal mean regret. Development seed 8101 is excluded.",
    }
    (ROOT / "docs/wind_route_study/low_value_confirmation_results_01.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in result.items() if k not in ["rows", "input_sha256"]}, indent=2)
    )


if __name__ == "__main__":
    main()
