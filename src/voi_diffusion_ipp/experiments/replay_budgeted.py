"""Read-only numeric replay of earned POAM snapshots, with provenance checks."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from ..config import Config
from ..wind.poam import POAM
from .two_region import make_two_region

ROOT = Path(__file__).resolve().parents[3]


def snapshots(path, periods):
    """Permit new planner code, but require unchanged real inference sources.

    Every requested predictive mean/std is also checked against the recording.
    No hidden physical field is returned to a planner.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    raw = json.loads(path.with_suffix(".json").read_text())
    meta = raw["metadata"]
    for name in ("poam.py", "poam_belief.py"):
        relative = "src/voi_diffusion_ipp/wind/" + name
        if (
            hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            != meta["source_sha256"][relative]
        ):
            raise ValueError("Real POAM source changed: " + name)
    cfg = Config(**meta["config"])
    spec_path = ROOT / meta.get("task_file", "configs/two_region_design.yaml")
    if hashlib.sha256(spec_path.read_bytes()).hexdigest() != meta["task_sha256"]:
        raise ValueError("Archived task differs")
    env, template, _, _, _ = make_two_region(yaml.safe_load(spec_path.read_text()), cfg)
    if meta["prior_regime"] not in ("outer-survey", "routes-survey"):
        raise ValueError("This replay supports recorded data-initialized survey missions")
    model = POAM.from_data(
        data["initial_observation_positions"],
        data["initial_observations"],
        jax.random.PRNGKey(600000),
        len(template.z),
        noise_std=cfg.observation_std,
        prior_std=cfg.wind_std,
        spectral_count=cfg.poam_spectral_nodes,
        prior_support=template.prior_support,
    )
    snapshot = jax.jit(lambda b: b.predict(env.cells))
    root = jax.random.PRNGKey(meta["seed"])
    stride = round(cfg.stage_steps * cfg.dt / meta["execution_dt"])
    for i in range(max(periods) + 1):
        if i in periods:
            belief = model.planning_belief()
            mean, cov = snapshot(belief)
            np.testing.assert_allclose(mean, data["mean"][i], atol=3e-5, rtol=3e-5)
            np.testing.assert_allclose(jnp.sqrt(cov[:, 0, 0]), data["std"][i], atol=3e-5, rtol=3e-5)
            yield (
                i,
                cfg,
                env,
                belief,
                jnp.asarray(data["x"][i * stride]),
                jnp.asarray(data["q"][i * stride]),
                raw["periods"][i],
                meta,
            )
        if i < max(periods):
            sl = slice(i * cfg.stage_steps, (i + 1) * cfg.stage_steps)
            model.update(
                data["observation_positions"][sl],
                data["observations"][sl],
                jax.random.fold_in(root, 500000 + i),
                train_steps=cfg.poam_train_steps,
            )
