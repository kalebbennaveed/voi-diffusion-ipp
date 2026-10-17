"""Reconstruct the retained long-mission POAM failure without rerunning MBD."""

import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.wind.poam import POAM, chol_solve, select_inducing

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "results/budgeted_missions/persistent_two_region_01/current_belief"
d = np.load(path.with_suffix(".npz"), allow_pickle=False)
meta = json.loads(path.with_suffix(".json").read_text())["metadata"]
cfg = Config(**meta["config"])
env, model, _, _, _ = make_two_region(
    yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text()), cfg
)
model = POAM.from_data(
    d["initial_observation_positions"],
    d["initial_observations"],
    jax.random.PRNGKey(600000),
    len(model.z),
    noise_std=cfg.observation_std,
    prior_std=cfg.wind_std,
    spectral_count=cfg.poam_spectral_nodes,
    prior_support=model.prior_support,
)
root = jax.random.PRNGKey(meta["seed"])
rows = []
for i in range(len(d["observations"]) // cfg.stage_steps):
    sl = slice(i * cfg.stage_steps, (i + 1) * cfg.stage_steps)
    x, y = jnp.asarray(d["observation_positions"][sl]), jnp.asarray(d["observations"][sl])
    if i >= 70:
        z = select_inducing(
            model.parameters, jnp.concatenate((model.z, x)), model.lengths, len(model.z)
        )
        P = chol_solve(model.old_K, model.kernel(model.z, z))
        K = model.kernel(z, z)
        Kuf = model.kernel(z, x)
        B = P.T @ model.B @ P + Kuf @ Kuf.T
        a = P.T @ model.a + Kuf @ y
        A = K + B / cfg.observation_std**2
        S = K @ chol_solve(A, K)
        S = (S + S.T) / 2
        row = {
            "period": i,
            "min_K_eigenvalue": float(np.linalg.eigvalsh(np.array(K, dtype=float)).min()),
            "min_B_eigenvalue": float(
                np.linalg.eigvalsh(np.array((B + B.T) / 2, dtype=float)).min()
            ),
            "min_A_eigenvalue": float(
                np.linalg.eigvalsh(np.array((A + A.T) / 2, dtype=float)).min()
            ),
            "min_S_eigenvalue": float(np.linalg.eigvalsh(np.array(S, dtype=float)).min()) if np.isfinite(np.array(S)).all() else None,
            "S_asymmetry": float(jnp.abs(K @ chol_solve(A, K) - (K @ chol_solve(A, K)).T).max()),
        }
        rows.append(row)
        print(row, flush=True)
        np.savez_compressed(
            ROOT / f"docs/budgeted_planner/poam_failure_period_{i}.npz",
            K=np.array(K),
            B=np.array(B),
            a=np.array(a),
            S=np.array(S),
            noise_std=cfg.observation_std,
        )
    losses = model.update(
        x, y, jax.random.fold_in(root, 500000 + i), train_steps=cfg.poam_train_steps
    )
    finite = all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(model.parameters))
    if i >= 70:
        print("losses", losses, "finite", finite, flush=True)
    if not finite:
        break
(ROOT / "docs/budgeted_planner/poam_long_failure.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
