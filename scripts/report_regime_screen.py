"""Validate and publish completed pairs from the frozen high-Q development screen.

No numerical models are imported: the active worktree's source remains frozen.
This translates its declared directory layout for the shared archival reporter.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def prepare(worktree, output):
    worktree, output = worktree.resolve(), output.resolve()
    if not worktree.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError("Study and outputs must remain inside voi-diffusion-ipp")
    original = worktree / "docs/regime_screen/protocol.json"
    protocol = json.loads(original.read_text())
    source = worktree / "results/regime_screen"
    expected = {k: v for k, v in protocol["source_sha256"].items() if k.startswith("src/")}
    cases, directories, states = [], {}, []
    for case, seed in protocol["cases"]:
        overrides = yaml.safe_load(protocol["configs"][case])
        scenario = overrides["scenario"]
        if scenario in directories and directories[scenario] != case:
            raise ValueError("Multiple configurations share a scenario identity")
        directories[scenario] = case
        cases.append([scenario, seed])
        records = {}
        for method in protocol["methods"]:
            path = source / case / f"{scenario}_{method}_{seed}.demo.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            if record["source_sha256"] != expected:
                raise ValueError(f"Source mismatch in {path}")
            cfg = record["config"]
            for key, value in overrides.items():
                if key == "profile":
                    # load_config expands this selector; it is not a Config field.
                    continue
                if cfg[key] != value:
                    raise ValueError(f"Declared configuration mismatch: {path}: {key}")
            metrics = record["metrics"]
            if (metrics["scenario"], metrics["method"], metrics["seed"]) != (
                scenario,
                method,
                seed,
            ):
                raise ValueError(f"Recording identity mismatch: {path}")
            steps = cfg["mission_horizons"] * cfg["stage_steps"]
            if len(record["mission"]["q"]) != steps + 1:
                raise ValueError(f"Incomplete mission in {path}")
            if abs(steps * cfg["dt"] - protocol["duration_seconds"]) > 1e-8:
                raise ValueError(f"Duration mismatch in {path}")
            records[method] = metrics
        state = {"scenario": scenario, "seed": seed, "completed_methods": list(records)}
        if set(records) == set(protocol["methods"]):
            proposed, current = (records[k] for k in ("proposed", "current_belief"))
            base = current["integrated_deficit"]
            state.update(
                proposed_cost=proposed["integrated_deficit"],
                current_belief_cost=base,
                reduction_percent=100 * (base - proposed["integrated_deficit"]) / base
                if base
                else None,
                proposed_collision=proposed["collision"],
                current_belief_collision=current["collision"],
                proposed_fallbacks=proposed["optimizer_failure_count"],
                current_belief_fallbacks=current["optimizer_failure_count"],
            )
        states.append(state)
    output.mkdir(parents=True, exist_ok=True)
    translated = {
        "type": "Archival adapter for an unchanged development-screen protocol",
        "original_protocol": str(original.relative_to(ROOT)),
        "original_protocol_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "scenarios_and_seeds": cases,
        "recording_directory_by_scenario": directories,
        "methods": protocol["methods"],
        "source_sha256": expected,
        "interpretation": "Development only; these seeds cannot establish significance or equivalence.",
    }
    path = output / "archival_protocol.json"
    path.write_text(json.dumps(translated, indent=2) + "\n")
    return source, path, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, default=ROOT / "tmp/learning-regimes")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/regime_screen_validation")
    args = parser.parse_args()
    source, translated, states = prepare(args.worktree, args.output)
    log = ROOT / "results/regime_screen_report.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("w") as stream:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/report_field_poam.py"),
                "--source",
                str(source),
                "--output",
                str(args.output.resolve()),
                "--protocol",
                str(translated),
                "--label-prefix",
                "High-Q POAM · 90 s",
                "--paired-only",
            ],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    summary = {
        "status": "Development screen: no superiority or equivalence claim",
        "complete_pairs": sum("proposed_cost" in row for row in states),
        "expected_pairs": len(states),
        "cases": states,
    }
    (args.output / "paired_outcomes.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
