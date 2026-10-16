"""Isolation checks for the explicit wind-in-RPCBF counterfactual study."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.scenarios import make_scenario, make_truth
from voi_diffusion_ipp.planning.rollout import rollout, waypoint_controller
from voi_diffusion_ipp.wind import InnerFields
from voi_diffusion_ipp.wind.physical import PhysicalWind

spec = importlib.util.spec_from_file_location(
    "wind_safety_ablation", Path(__file__).resolve().parents[1] / "scripts/wind_safety_ablation.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_zero_wind_filter_ablation_preserves_real_dynamics_and_sensing():
    cfg = replace(
        Config(),
        whole_field=True,
        wind_backend="poam",
        scenario="field_mixed",
        grid_nx=9,
        grid_ny=7,
        poam_inducing_x=3,
        poam_inducing_y=3,
        poam_spectral_nodes=4,
        stage_steps=3,
        backup_steps=3,
        inner_samples=1,
        control_representation="waypoint",
    )
    env, belief, x, q = make_scenario(cfg)
    world = make_truth(cfg, belief, 302)
    inner = InnerFields(belief.basis, jnp.zeros((1, *belief.sample_shape)))
    controls = jnp.tile(jnp.array([0.6, 0.4]), (cfg.stage_steps, 1))
    noise = jnp.zeros_like(controls)
    expected = jax.jit(lambda: rollout(x, q, controls, belief, inner, world, noise, env, cfg))()
    ablated = jax.jit(
        lambda: study.replay_rollout(x, q, controls, belief, inner, world, noise, env, cfg, False)
    )()
    # When the real filter is supplied zero mean and zero samples, the ablation
    # must be exactly the same controller, including physical wind and sensing.
    for a, b in zip(expected, ablated):
        np.testing.assert_allclose(a, b, atol=2e-6)
    np.testing.assert_allclose(ablated.observations, world.at(ablated.x[1:, :2]), atol=1e-6)
    assert np.linalg.norm(ablated.observations) > 0.01
    assert np.max(np.abs(ablated.q[-1] - q)) > 0.01
    learned = belief.condition(jnp.tile(x[:2], (3, 1)), jnp.ones((3, 2)) * 0.3)
    result = jax.jit(
        lambda: study.replay_rollout(x, q, controls, learned, inner, world, noise, env, cfg, False)
    )()
    np.testing.assert_allclose(
        result.nominal[0], waypoint_controller(x, controls[0], learned, cfg), atol=1e-6
    )
    assert np.linalg.norm(result.nominal[0] - ablated.nominal[0]) > 0.01


def test_dense_collision_check_detects_between_endpoint_crossing():
    cfg = replace(Config(), scenario="localized_corridor")
    env, _, _, _ = make_scenario(cfg)
    env = env._replace(obstacles=jnp.array([[0.0, 0.0, 0.3]]))
    zero = PhysicalWind(jnp.zeros((1, 2)), jnp.ones(1), jnp.zeros(1), jnp.zeros(2), jnp.array(0.0))
    states = jnp.array([[-1.0, 0.0, 2.0, 0.0], [1.0, 0.0, 2.0, 0.0]])
    clearances, error = study.dense_clearance(states, jnp.zeros((1, 2)), zero, env, 1.0)
    assert min(float(env.clearance(x)) for x in states) > 0
    np.testing.assert_allclose(clearances, [-0.3], atol=1e-6)
    assert float(error) < 1e-6
