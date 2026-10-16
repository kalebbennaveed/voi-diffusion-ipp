"""Expose genuine completed execution from unfinished missions without running JAX.

Both available arms are cut at the same completed period. Missing future data and
unavailable nominal forecasts are omitted, never synthesized. Final reports stay
separate from these explicitly incomplete previews.
"""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/demo/previews"
SOURCES = [
    ROOT / "results/persistent_poam/recordings",
    ROOT / "tmp/learning-regimes/results/regime_screen",
]


def convert(data, periods, source):
    cfg = data["metadata"]["config"]
    execution = data["execution"][:periods]
    arrays = {}
    for key in ("x", "q"):
        arrays[key] = np.concatenate([execution[0][key][:1]] + [r[key][1:] for r in execution])
    for key in (
        "nominal",
        "controls",
        "observations",
        "correction",
        "feasible",
        "clearance",
        "admissible",
        "slack",
    ):
        arrays["control" if key == "controls" else key] = np.concatenate(
            [r[key] for r in execution]
        )
    q = arrays["q"]
    deficit = np.maximum(np.asarray(data["environment"]["targets"]) - q, 0).mean(1)
    cost = float(np.trapezoid(deficit, dx=cfg["dt"]))
    np.testing.assert_allclose(cost, sum(r["cost"] for r in execution), atol=2e-5)
    assert len(q) == periods * cfg["stage_steps"] + 1
    assert np.isfinite(q).all() and np.all((q >= 0) & (q <= 1))
    times, counts = [], []
    checkpoint = source.with_name(source.name.replace(".partial.json", ".checkpoints"))
    for h in range(periods):
        with np.load(checkpoint / f"plan_{h:04d}.npz") as plan:
            np.testing.assert_array_equal(plan["vector"], data["planning"][h]["selected_vector"])
            times.append(float(plan["seconds"]))
            counts.append(int(plan["evaluations"]))
    metrics = {
        "scenario": cfg["scenario"],
        "method": data["metadata"]["method"],
        "seed": data["metadata"]["seed"],
        "integrated_deficit": cost,
        "final_deficit": float(deficit[-1]),
        "collision": bool(np.min(arrays["clearance"]) < 0),
        "minimum_clearance": float(np.min(arrays["clearance"])),
        "max_qp_slack": float(np.max(arrays["slack"])),
        "optimizer_failure_count": sum(
            p["selected_cost"] is None for p in data["planning"][:periods]
        ),
        "mean_plan_seconds": float(np.mean(times)),
        "objective_evaluations_per_plan": float(np.mean(counts)),
        "jax_version": importlib.metadata.version("jax"),
    }
    return {
        "schema": 2,
        "title": "Unfinished mission · executed preview",
        "metrics": metrics,
        "config": cfg,
        "mission": {k: v.tolist() for k, v in arrays.items()},
        **{k: data[k] for k in ("environment", "wind_grid", "physical_wind")},
        "beliefs": data["beliefs"][: periods + 1],
        "planning": data["planning"][:periods],
        "source_sha256": data["metadata"]["source_sha256"],
        "wind_model": "Full online POAM",
        "forecast_note": "Actual recorded MBD forecasts from planning world 1. Execution is recorded separately. This mission is unfinished.",
        "preview": {
            "complete": False,
            "completed_periods": periods,
            "planned_periods": cfg["mission_horizons"],
            "available_seconds": periods * cfg["stage_steps"] * cfg["dt"],
            "planned_seconds": cfg["mission_horizons"] * cfg["stage_steps"] * cfg["dt"],
            "source_file": str(source.relative_to(ROOT)),
            "source_sha256": data["snapshot_sha256"],
            "note": "Actual completed execution only; no final performance conclusion. Nominal open-loop paths are unavailable until finalization.",
        },
    }


def publish():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for directory in SOURCES:
        status_file = directory / "status.json"
        stopped = bool(
            status_file.exists() and json.loads(status_file.read_text()).get("stopped_reason")
        )
        groups = {}
        for source in directory.glob("*/*.partial.json"):
            raw = source.read_bytes()
            data = json.loads(raw)
            data["snapshot_sha256"] = hashlib.sha256(raw).hexdigest()
            meta = data["metadata"]
            if not data["execution"]:
                continue
            groups.setdefault((meta["config"]["scenario"], meta["seed"]), []).append((source, data))
        for (scenario, seed), group in groups.items():
            reference = group[0][1]
            for _, other in group[1:]:
                assert reference["metadata"]["config"] == other["metadata"]["config"]
                assert reference["metadata"]["source_sha256"] == other["metadata"]["source_sha256"]
                assert reference["physical_wind"] == other["physical_wind"]
                assert reference["execution"][0]["x"][0] == other["execution"][0]["x"][0]
                assert reference["execution"][0]["q"][0] == other["execution"][0]["q"][0]
            if (
                all(
                    p.with_name(p.name.replace(".partial.json", ".demo.json")).exists()
                    for p, _ in group
                )
                and len(group) == 2
            ):
                continue
            periods = min(len(d["execution"]) for _, d in group)
            for source, data in group:
                method = data["metadata"]["method"]
                identity = f"preview_{scenario}_{method}_{seed}"
                output = OUT / f"{identity}_{periods}{'_stopped' if stopped else ''}.json"
                cfg = data["metadata"]["config"]
                duration = periods * cfg["stage_steps"] * cfg["dt"]
                if not output.exists():
                    converted = convert(data, periods, source)
                    converted["preview"]["stopped"] = stopped
                    temporary = output.with_suffix(".tmp")
                    temporary.write_text(
                        json.dumps(converted, separators=(",", ":"), allow_nan=False)
                    )
                    temporary.replace(output)
                records.append(
                    {
                        "id": identity,
                        "kind": "mission_preview",
                        "scenario": scenario,
                        "method": method,
                        "seed": seed,
                        "group": f"preview-{scenario}-{seed}",
                        "label": f"{'STOPPED FOR REDESIGN' if stopped else 'IN PROGRESS'} · {duration:.1f} s recorded · {scenario.removeprefix('persistent_')} · {method}",
                        "stopped": stopped,
                        "available_seconds": duration,
                        "periods": periods,
                        "file": str(output.relative_to(ROOT)),
                        "live": False,
                    }
                )
    with (ROOT / "results/demo_catalog.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        catalog = ROOT / "demo/data/catalog.json"
        old = json.loads(catalog.read_text())
        prior_files = {r["file"] for r in old if r.get("kind") == "mission_preview"}
        # Local unfinished data belongs to the server's existing runtime catalog,
        # not the committed catalog (which must remain usable in fresh clones).
        runtime = ROOT / "results/demo/live"
        existing = list(runtime.glob("preview_*/record.json"))
        prior_files |= {json.loads(p.read_text())["file"] for p in existing}
        for record in records:
            target = runtime / record["id"] / "record.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(target)
        identities = {r["id"] for r in records}
        for path in existing:
            if path.parent.name not in identities:
                path.unlink()
        new = [r for r in old if r.get("kind") != "mission_preview"]
        if new != old:
            temporary = catalog.with_suffix(".tmp")
            temporary.write_text(json.dumps(new, indent=2) + "\n")
            temporary.replace(catalog)
        keep = prior_files | {r["file"] for r in records}
        for path in OUT.glob("preview_*.json"):
            if str(path.relative_to(ROOT)) not in keep:
                path.unlink()
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "publisher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            rows = publish()
            print([(r["id"], r["available_seconds"]) for r in rows], flush=True)
            if not args.watch:
                break
            active = False
            for directory in SOURCES:
                status = directory / "status.json"
                if status.exists():
                    active |= any(
                        j["status"] in ("running", "queued")
                        for j in json.loads(status.read_text())["jobs"]
                    )
            if not active:
                break
            time.sleep(20)
