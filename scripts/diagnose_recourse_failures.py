"""Replay every infeasible QP in the recourse study and verify with independent LP."""

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
    root = Path(__file__).resolve().parents[1]
    records = []
    for folder in ("recourse_pilot", "recourse_validation", "recourse_ablations"):
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
                    np.testing.assert_allclose(lp.fun, a["slack"][index], atol=5e-5)
                    records.append(
                        {
                            "run": str(path.relative_to(root)),
                            "step": int(index),
                            "step_end_seconds": float((index + 1) * cfg.dt),
                            "saved_slack": float(a["slack"][index]),
                            "independent_minimum_common_slack": float(lp.fun),
                            "clearance": float(a["clearance"][index]),
                            "control": a["control"][index].tolist(),
                            "A": A.tolist(),
                            "b": b.tolist(),
                            "barrier_values": B.tolist(),
                            "interpretation": "Bounded robust halfspaces are infeasible; independent LP confirms positive minimum slack. This was a recorded relaxed-QP fallback, not an unreported collision or optimizer failure.",
                        }
                    )
    out = root / "docs/recourse_validation/failure_diagnostics.json"
    out.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
