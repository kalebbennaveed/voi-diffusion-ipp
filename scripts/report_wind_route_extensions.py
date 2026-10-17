"""Retain all development cases and the explicitly privileged information reference."""

import json

import numpy as np
import report_wind_route_study as study


def main():
    previous_inputs = study.inputs
    extensions = {
        "reversed_branches": "Reversed wind · two-route policy",
        "open_passage": "Open passage · off-route wind",
    }
    study.NAMES.update(extensions)
    study.OUT = study.ROOT / "docs/wind_route_study/extended_results_01"
    study.OUT.mkdir(exist_ok=True)

    def inputs(case):
        if case not in extensions:
            return previous_inputs(case)
        folder = "reversed_branches_01" if case == "reversed_branches" else "open_passage_wind_01"
        return {
            m: study.ROOT / "results/budgeted_missions" / folder / (m + ".npz")
            for m in ["proposed", "current_belief"]
        }

    study.inputs = inputs
    study.report()
    reference = study.ROOT / "results/budgeted_missions/reversed_informed_01/oracle.npz"
    original = study.ROOT / "results/budgeted_missions/reversed_wind_01/current_belief.npz"
    rm = json.loads(reference.with_suffix(".json").read_text())["metadata"]
    om = json.loads(original.with_suffix(".json").read_text())["metadata"]
    assert rm["config"] == (om["config"] | {"budgeted_policy": "compact"})
    for key in [
        "seed",
        "task_sha256",
        "physical_wind_spec",
        "prior_regime",
        "execution_dt",
        "observation_dt",
    ]:
        assert rm[key] == om[key], key
    changed = [f for f, h in rm["source_sha256"].items() if om["source_sha256"].get(f) != h]
    assert set(changed) <= {
        "src/voi_diffusion_ipp/config.py",
        "src/voi_diffusion_ipp/planning/budgeted.py",
    }
    with np.load(reference, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    with np.load(original, allow_pickle=False) as archive:
        for key in [
            "initial_observation_positions",
            "initial_observations",
            "physical_wind",
            "targets",
            "obstacles",
            "cells",
        ]:
            np.testing.assert_array_equal(data[key], archive[key])
        for key in ["x", "q"]:
            np.testing.assert_array_equal(data[key][0], archive[key][0])
    error = study.clarity_check(data, rm["config"], rm["execution_dt"])
    assert error < 2e-5
    result = {
        "label": "Privileged fully informed compact planner; finite MBD budget, not a global optimal bound",
        "audit": json.loads(reference.with_suffix(".audit.json").read_text()),
        "clarity_replay_max_error": error,
        "matching": "Same normalized default configuration, task, initial states and archived survey. Optional branch representation was added to source but is disabled in this reference.",
        "changed_source_files": changed,
        "limitation": "This measures a reference with perfect physical wind knowledge for planning, compensation and safety; it does not isolate anticipated learning or demonstrate that POAM can earn this information.",
    }
    (study.OUT / "fully_informed_reference.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
