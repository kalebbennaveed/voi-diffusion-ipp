"""Replay every infeasible QP in the recourse study and verify with independent LP."""

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linprog

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.scenarios import make_scenario
from voi_diffusion_ipp.safety.rpcbf import constraints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directories",
        nargs="+",
        default=["recourse_pilot", "recourse_validation", "recourse_ablations"],
    )
    parser.add_argument("--output", default="docs/recourse_validation/failure_diagnostics.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    records = []
    for folder in args.directories:
        for path in sorted((root / "results" / folder).rglob("*.json")):
            r = json.loads(path.read_text())
            if not isinstance(r, dict) or not r.get("qp_infeasible_fraction", 0):
                continue
            cfg = Config(
                **{
                    **r["config"],
                    "seeds": tuple(r["config"]["seeds"]),
                    "methods": tuple(r["config"]["methods"]),
                }
            )
            env, prior, _, _ = make_scenario(cfg)
            with np.load(path.with_suffix(".npz")) as a:
                for index in np.flatnonzero(~a["feasible"]):
                    h = int(index) // cfg.stage_steps
                    belief = prior
                    for period in range(h):
                        start, end = period * cfg.stage_steps, (period + 1) * cfg.stage_steps
                        belief = belief.condition(
                            jnp.asarray(a["x"][start + 1 : end + 1, :2]),
                            jnp.asarray(a["observations"][start:end]),
                        )
                    if "belief_coefficients" in a:
                        belief = prior._replace(
                            mean=jnp.asarray(a["belief_coefficients"][h]),
                            covariance=jnp.asarray(a["belief_covariances"][h]),
                        )
                    key = jax.random.fold_in(jax.random.PRNGKey(r["seed"]), 300000 + h)
                    eps = jax.random.normal(key, (cfg.inner_samples, *belief.mean.shape))
                    inner = belief.inner_fields(eps)
                    evaluate = jax.jit(
                        partial(constraints, belief=belief, inner=inner, env=env, cfg=cfg)
                    )
                    A, b, B, _ = evaluate(jnp.asarray(a["x"][index]))
                    A, b, B = map(np.asarray, (A, b, B))
                    lp = linprog(
                        [0, 0, 1],
                        A_ub=np.column_stack((A, -np.ones(len(b)))),
                        b_ub=b,
                        bounds=[(-cfg.umax, cfg.umax)] * 2 + [(0, None)],
                        method="highs",
                    )
                    if not lp.success:
                        raise RuntimeError(lp.message)
                    # project() accepts every row, including actuator bounds,
                    # within 2e-5. Compare its slack against the same tolerances,
                    # while also retaining the strictly bounded independent LP.
                    tolerance = 2e-5
                    tolerant_lp = linprog(
                        [0, 0, 1],
                        A_ub=np.column_stack((A, -np.ones(len(b)))),
                        b_ub=b + tolerance,
                        bounds=[(-cfg.umax - tolerance, cfg.umax + tolerance)] * 2 + [(0, None)],
                        method="highs",
                    )
                    if not tolerant_lp.success:
                        raise RuntimeError(tolerant_lp.message)
                    if lp.fun <= 0:
                        raise AssertionError("Independent replay did not confirm infeasibility")
                    records.append(
                        {
                            "run": str(path.relative_to(root)),
                            "step": int(index),
                            "step_end_seconds": float((index + 1) * cfg.dt),
                            "saved_slack": float(a["slack"][index]),
                            "independent_minimum_common_slack": float(lp.fun),
                            "independent_slack_with_projection_tolerance": float(tolerant_lp.fun),
                            "saved_vs_tolerant_lp_error": float(
                                a["slack"][index] - tolerant_lp.fun
                            ),
                            "clearance": float(a["clearance"][index]),
                            "control": a["control"][index].tolist(),
                            "A": A.tolist(),
                            "b": b.tolist(),
                            "barrier_values": B.tolist(),
                            "interpretation": "Independently reconstructed bounded halfspaces require positive slack. Projection tolerances and the saved-versus-recomputed numerical discrepancy are reported explicitly; exact equality of separately compiled float32 spline gradients is not claimed. This was a recorded relaxed-QP fallback, not an unreported collision or optimizer failure.",
                        }
                    )
    out = (root / args.output).resolve()
    out.relative_to(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
