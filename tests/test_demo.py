import json
from dataclasses import replace

import jax
import numpy as np
import pytest

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.scenarios import make_scenario
from voi_diffusion_ipp.planning.mpc import initial_policy, next_initial
from voi_diffusion_ipp.planning.two_stage import make_buffers, objective
from voi_diffusion_ipp.viz.demo_data import serializable
from voi_diffusion_ipp.viz.demo_server import validate_request


def test_demo_rejects_unbounded_or_ambiguous_jobs():
    for data in (
        {"method": "shell"},
        {"proposals": 100000},
        {"seed": -1},
        {"seed": 1.5},
        {"wind_std": float("nan")},
        {"measurements": "false"},
        {"output": "/tmp/example"},
    ):
        with pytest.raises(ValueError):
            validate_request(data)
    cfg, method, seed = validate_request(
        {"method": "cem", "seed": 37, "variant": "multistage", "measurements": False}
    )
    assert method == "cem" and seed == 37 and cfg.future_updates and not cfg.measurements


def test_strict_demo_json_encodes_rejected_candidates_as_null():
    value = serializable({"costs": np.array([1.0, np.inf, np.nan]), "safe": np.bool_(True)})
    assert json.loads(json.dumps(value, allow_nan=False)) == {
        "costs": [1.0, None, None],
        "safe": True,
    }


def test_frozen_comparator_does_not_anticipate_future_measurements_in_warm_start():
    cfg = replace(
        Config(),
        scenario="localized_corridor",
        stage_steps=3,
        future_steps=6,
        control_representation="waypoint",
        branch_rule="task",
        backup_steps=3,
        inner_samples=1,
        recourse_samples=1,
        worlds=1,
    )
    updated = replace(cfg, future_updates=True)
    env, b, x, q = make_scenario(cfg)
    vector = initial_policy(x, q, b, env, cfg)
    before = jax.jit(lambda: next_initial(vector, x, q, b, env, cfg, False))()
    after = jax.jit(lambda: next_initial(vector, x, q, b, env, updated, False))()
    np.testing.assert_array_equal(before, after)
    buf = make_buffers(b, jax.random.PRNGKey(22), cfg)
    a = jax.jit(lambda: objective(vector, x, q, b, buf, env, cfg, "current_belief"))()
    z = jax.jit(lambda: objective(vector, x, q, b, buf, env, updated, "current_belief"))()
    np.testing.assert_array_equal(a, z)
