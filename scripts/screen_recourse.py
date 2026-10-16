"""Development-only first-action landscape, without selecting favorable true worlds."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import config_dict, load_config
from voi_diffusion_ipp.experiments.scenarios import make_scenario
from voi_diffusion_ipp.planning.mpc import initial_policy
from voi_diffusion_ipp.planning.two_stage import evaluate_one, make_buffers


def main():
    cfg = load_config("configs/recourse_dev.yaml")
    env, belief, x, q = make_scenario(cfg)
    vectors, goals = [], []
    for gx in (-1.45, -1.05, -0.7, -0.35):
        for gy in (-0.6, 0.0, 0.45):
            goal = jnp.array([gx, gy])
            vectors.append(initial_policy(x, q, belief, env, cfg, goal))
            goals.append([gx, gy])
    buf = make_buffers(belief, jax.random.PRNGKey(7020), cfg)
    rows = []
    for mode in ("proposed", "current_belief"):

        def evaluate(vector, mode=mode):
            def one(args):
                coefficients, noise = args
                result = evaluate_one(
                    vector,
                    x,
                    q,
                    belief,
                    coefficients,
                    noise,
                    buf.inner_prior_eps,
                    buf.inner_future_eps,
                    env,
                    cfg,
                    mode,
                )
                return jnp.array(
                    [
                        result.cost,
                        result.first.cost,
                        result.future.cost,
                        result.branch,
                        jnp.min(result.first.clearance),
                        jnp.min(result.future.clearance),
                    ]
                )

            return jax.lax.map(one, (buf.world_coefficients, buf.observation_noise))

        fn = jax.jit(evaluate)
        for goal, vector in zip(goals, vectors):
            a = np.asarray(fn(vector))
            record = {
                "mode": mode,
                "goal": goal,
                "objective": float(a[:, 0].mean()),
                "first_cost": float(a[:, 1].mean()),
                "future_cost": float(a[:, 2].mean()),
                "branch_0_fraction": float((a[:, 3] == 0).mean()),
                "collision_scenarios": int((a[:, 4:].min(axis=1) < 0).sum()),
            }
            rows.append(record)
            print(record, flush=True)
    out = Path("results/recourse_screen")
    out.mkdir(parents=True, exist_ok=True)
    (out / "screen.json").write_text(
        json.dumps({"config": config_dict(cfg), "rows": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
