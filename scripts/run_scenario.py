"""Run one paper scenario with its saved CFD field and recorded initial state."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Scenario numbers follow the project page. Each duration is its comparison window.
SCENARIOS = {
    1: ((1.45, 0.875, 0, 0), 532),
    2: ((1.45, 0.875, 0, 0), 560),
    3: ((1.4, 4.15, 0, 0), 514),
    4: ((1.95, 1, 0, 0), 544),
}
# The first value selects the acquisition policy and the second the planner/filter.
METHODS = {
    "proposed": ("pathmask", "learning"),
    "global": ("global", "learning"),
    "aware": ("aware", "aware"),
    "filtered": ("filtered", "mbd-filtered"),
    "unfiltered": ("unfiltered", "mbd-unfiltered"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=int, choices=SCENARIOS, default=1)
    parser.add_argument("--method", choices=METHODS, default="proposed")
    parser.add_argument("--seconds", type=int, help="Simulation duration in seconds (default: scenario comparison window)")
    parser.add_argument("--output", type=Path, help="A new directory for this run")
    args = parser.parse_args()
    initial, window = SCENARIOS[args.scenario]
    seconds = window if args.seconds is None else args.seconds
    if seconds <= 0:
        parser.error("--seconds must be positive")
    mode, method = METHODS[args.method]
    output = args.output or ROOT / "runs" / f"scenario-{args.scenario}-{args.method}"
    command = [
        sys.executable, "-u", str(ROOT / "scripts/airflow/run_full_learning.py"),
        "--output", str(output.resolve()), "--seconds", str(seconds),
        "--environment-spec", str(ROOT / "data" / f"scenario{args.scenario}" / "geometry.json"),
        "--wind-site", str(ROOT), "--scenarios", str(args.scenario),
        "--cases", "center", "--initial-state", *map(str, initial),
        "--modes", mode, "--comparison-method", method,
        "--uncertainty-only", "--global-std-threshold", "1.1", "--task-std-threshold", "1.1",
        "--efficient-planning", "--allow-validation-slack", "--observation-first-inducing",
        "--exact-safety-variance", "--gp-lengths", ".1,.2,.5,1", "--train-kernel",
        "--factorized-support", "--plan-limit", "2", "--wall-limit", "4800",
        "--learning-controller", "reachable-waypoints", "--global-information-goals",
        "--prefix-screen", "braking-feasible", "--diversify-stalled-routes",
    ]
    if args.scenario in (3, 4):
        command.extend(["--initial-region-clarity", ".35"])
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
