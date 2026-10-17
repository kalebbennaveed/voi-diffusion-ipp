"""Check the unintended chunk-size change in the expanded-budget diagnostic.

Same actual state, clarity, initializations, MBD key and 256 proposals; only JAX
candidate batching differs. This does not assert complete closed-loop equality.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.planning.budgeted import warm_starts
from voi_diffusion_ipp.planning.wind_blind import make_nominal_planner

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = json.loads(args.path.with_suffix(".json").read_text())
    meta = raw["metadata"]
    data = np.load(args.path, allow_pickle=False)
    cfg = Config(**meta["config"])
    env, _, _, _, _ = make_two_region(
        yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text()), cfg
    )
    indices = np.flatnonzero(np.asarray(env.targets) > 0)
    planners = [make_nominal_planner(replace(cfg, candidate_chunk=i), indices) for i in [1, 2]]
    stride = round(cfg.stage_steps * cfg.dt / meta["execution_dt"])
    results, arrays = [], {}
    for i in [0, 15, 30]:
        x, q = jnp.asarray(data["x"][i * stride]), jnp.asarray(data["q"][i * stride])
        initials = warm_starts(x, env, cfg, meta["rapid_shared_initializations"])
        key = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(meta["seed"]), 100 + i), 4)[2]
        a, b = [jax.block_until_ready(plan(x, q, env, initials, key)) for plan in planners]
        c1, c2 = np.asarray(a.history.costs), np.asarray(b.history.costs)
        mask = np.isfinite(c1) & np.isfinite(c2)
        row = {
            "period": i,
            "selected_costs": [float(a.cost), float(b.cost)],
            "selected_vector_max_abs_difference": float(np.max(np.abs(a.vector - b.vector))),
            "all_candidate_vectors_max_abs_difference": float(
                np.max(np.abs(a.history.candidates - b.history.candidates))
            ),
            "finite_candidate_cost_max_abs_difference": float(np.max(np.abs(c1[mask] - c2[mask])))
            if mask.any()
            else None,
            "finite_classification_disagreements": int(np.sum(np.isfinite(c1) != np.isfinite(c2))),
        }
        results.append(row)
        for label, plan in [("chunk1", a), ("chunk2", b)]:
            arrays[f"period{i}_{label}_vector"] = np.asarray(plan.vector)
            arrays[f"period{i}_{label}_candidates"] = np.asarray(plan.history.candidates)
            arrays[f"period{i}_{label}_costs"] = np.asarray(plan.history.costs)
        print(row, flush=True)
    args.output.write_text(
        json.dumps(
            {
                "input": str(args.path),
                "periods": results,
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "limits": "Three selected states only; does not erase the recorded protocol deviation or prove closed-loop equivalence.",
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()
