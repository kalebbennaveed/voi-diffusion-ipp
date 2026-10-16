"""Run and recover the declared 122.4-second full-POAM monitoring comparisons.

Three independent processes fit the native 16 GB development machine. Completed
pairs are published to the demo; incomplete recordings never masquerade as results.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
OUTPUT = ROOT / "results/persistent_poam/recordings"
REPORT = ROOT / "docs/persistent_poam_validation"
CASES = [("persistent_archipelago", 400), ("persistent_channels", 401), ("persistent_mosaic", 402)]
METHODS = ["proposed", "current_belief"]


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def hashes():
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src").rglob("*.py"))
    }


def protocol():
    REPORT.mkdir(parents=True, exist_ok=True)
    configuration = {case: (ROOT / "configs" / f"{case}.yaml").read_text() for case, _ in CASES}
    path = REPORT / "protocol.json"
    if path.exists():
        declared = json.loads(path.read_text())
        if declared["source_sha256"] != hashes() or declared["config_by_scenario"] != configuration:
            raise ValueError(
                "Frozen numerical source/config changed; preserve this study and use a new protocol"
            )
        return declared
    declared = {
        "purpose": "Persistent whole-field monitoring with process-noise decay and repeated sensing, paired against no anticipated learning.",
        "selection": "All three scenarios, both methods and seeds declared before persistent mission outcomes; retain failures and regressions.",
        "scenarios_and_seeds": CASES,
        "methods": METHODS,
        "config_by_scenario": configuration,
        "source_sha256": hashes(),
        "declared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": 122.4,
        "primary_metric": "Time integral of mean positive clarity deficit across all 475 equal-area cells.",
        "initial_condition": "QoI clarity starts 0.04 above each target (an explicit prior assumption); wind begins with the same untrained POAM prior for both methods.",
        "process_noise": "Diagonal QoI process matrix: uniform archipelago, spatially striped channels, spatially patchy mosaic. No time reset or target-arrival rewards.",
        "comparison": "Identical physical wind draw, initial state, initial clarity, observation-noise stream and compute budget. Both methods assimilate real wind measurements; only the proposed objective anticipates future learning.",
        "planner": "Full MBD and spline RPCBF with soft QP slack; 24 executed plus 24 future steps, dt=0.15, 34 replans, four worlds/four inner samples/two recourse samples/four diffusion levels/twelve proposals.",
        "wind": "Full POAM attentive kernel, inducing relocation, transported statistics and variational EM; no finite-anchor inference substitution. Frozen hyperparameters within planning, online adaptation between plans.",
        "sampler": "Full-kernel support correction on 29x23 points, retained between-support spectral residual and sparse-posterior Matheron conditioning. Numerical covariance approximation error is reported.",
        "revisit_definition": "Sensing strength C^2>=0.25 for at least two samples; absence under three seconds merges passes.",
        "recovery_definition": "A previously satisfied cell drops below target for at least one second and subsequently reaches target again.",
        "interpretation": "Three paired development seeds, not a statistical superiority study. Planning is offline on CPU; playback speed is independent of computation.",
    }
    atomic_json(path, declared)
    return declared


def completed(scenario, method, seed, declared):
    path = OUTPUT / scenario / f"{scenario}_{method}_{seed}.demo.json"
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    if data["source_sha256"] != declared["source_sha256"]:
        raise ValueError(f"Source mismatch in completed recording {path}")
    return True


def publish():
    command = [
        sys.executable,
        str(ROOT / "scripts/report_field_poam.py"),
        "--source",
        str(OUTPUT),
        "--output",
        str(REPORT),
        "--label-prefix",
        "Persistent POAM · 122 s",
        "--paired-only",
    ]
    with (OUTPUT / "report.log").open("a") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def worker(scenario, method, seed):
    # Import JAX only in workers, keeping the coordinator lightweight.
    from voi_diffusion_ipp.config import load_config
    from voi_diffusion_ipp.viz.demo_data import export_demo

    protocol()  # Reject mixing changed implementations into a partially finished study.
    cfg = load_config(ROOT / "configs" / f"{scenario}.yaml")
    print(time.strftime("%H:%M:%S"), "START", scenario, method, seed, flush=True)
    export_demo(
        cfg,
        method,
        seed,
        OUTPUT / scenario,
        lambda message: print(time.strftime("%H:%M:%S"), message, flush=True),
    )
    print(time.strftime("%H:%M:%S"), "COMPLETE", scenario, method, seed, flush=True)


def coordinate(workers, initialize_only):
    declared = protocol()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if initialize_only:
        print(REPORT / "protocol.json")
        return
    jobs = [
        {
            "scenario": scenario,
            "method": method,
            "seed": seed,
            "status": "complete" if completed(scenario, method, seed, declared) else "queued",
        }
        for scenario, seed in CASES
        for method in METHODS
    ]
    active = {}
    partial_times = {}
    started = time.time()

    def save_status():
        for job in jobs:
            partial = (
                OUTPUT
                / job["scenario"]
                / f"{job['scenario']}_{job['method']}_{job['seed']}.partial.json"
            )
            if partial.exists() and partial.stat().st_mtime_ns != partial_times.get(str(partial)):
                data = json.loads(partial.read_text())
                job["completed_periods"] = len(data.get("execution", []))
                partial_times[str(partial)] = partial.stat().st_mtime_ns
            job["total_periods"] = 34
        atomic_json(
            OUTPUT / "status.json",
            {
                "pid": os.getpid(),
                "updated_at": time.time(),
                "started_at": started,
                "duration_seconds": 122.4,
                "jobs": jobs,
            },
        )

    try:
        publish()
        while any(job["status"] in ("queued", "running") for job in jobs):
            for i, job in enumerate(jobs):
                if len(active) >= workers:
                    break
                if job["status"] != "queued":
                    continue
                log = OUTPUT / f"{job['scenario']}_{job['method']}.log"
                stream = log.open("a", buffering=1)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    job["scenario"],
                    job["method"],
                    str(job["seed"]),
                ]
                process = subprocess.Popen(
                    command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
                )
                active[i] = process, stream
                job.update(status="running", pid=process.pid, log=str(log.relative_to(ROOT)))
                print("START", job["scenario"], job["method"], "pid", process.pid, flush=True)
            for i, (process, stream) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                job = jobs[i]
                valid = code == 0 and completed(
                    job["scenario"], job["method"], job["seed"], declared
                )
                job.update(status="complete" if valid else "failed", exit_code=code)
                del active[i]
                print(job["status"].upper(), job["scenario"], job["method"], flush=True)
                publish()
            save_status()
            if active:
                time.sleep(5)
    except BaseException:
        for process, stream in active.values():
            process.terminate()
            stream.close()
        raise
    if any(job["status"] == "failed" for job in jobs):
        raise SystemExit("At least one mission failed; logs and checkpoints retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--worker", nargs=3, metavar=("SCENARIO", "METHOD", "SEED"))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker[0], args.worker[1], int(args.worker[2]))
    else:
        coordinate(args.workers, args.initialize_only)
