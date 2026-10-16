from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from voi_diffusion_ipp.clarity import deficit, step
from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.experiments.scenarios import make_scenario, make_truth
from voi_diffusion_ipp.wind.poam_belief import POAMBelief


def test_full_field_grid_and_clarity_cost_include_every_cell():
    cfg = load_config("configs/field_poam.yaml")
    env, belief, x, q = make_scenario(cfg)
    assert isinstance(belief, POAMBelief)
    assert len(q) == cfg.grid_nx * cfg.grid_ny == 475
    assert len(np.unique(np.asarray(env.cells)[:, 0])) == cfg.grid_nx
    assert len(np.unique(np.asarray(env.cells)[:, 1])) == cfg.grid_ny
    np.testing.assert_allclose(np.asarray(env.cells).mean(0), 0, atol=1e-6)
    # A cell away from every legacy ROI changes the cost by exactly its area fraction.
    changed = q.at[-1].add(-0.05)
    np.testing.assert_allclose(
        deficit(changed, env.targets) - deficit(q, env.targets), 0.05 / len(q), atol=1e-7
    )
    out = step(
        q, x[:2], x[:2], env.cells, cfg.sensor_sigma, cfg.sensor_R, cfg.clarity_decay, cfg.dt
    )
    nearest = jnp.argmin(jnp.linalg.norm(env.cells - x[:2], axis=1))
    farthest = jnp.argmax(jnp.linalg.norm(env.cells - x[:2], axis=1))
    assert out[nearest] > q[nearest]
    assert out[farthest] < q[farthest]
    assert jnp.all((out >= 0) & (out <= 1))
    with pytest.raises(ValueError, match="full POAM"):
        replace(cfg, wind_backend="feature")


def test_held_out_physics_does_not_depend_on_estimator_or_grid():
    cfg = load_config("configs/field_poam.yaml")
    _, belief, _, _ = make_scenario(cfg)
    cfg2 = replace(cfg, poam_inducing_x=3, grid_nx=11, poam_spectral_nodes=16)
    _, other, _, _ = make_scenario(cfg2)
    x = jnp.array([[0.2, -0.7], [-1.0, 1.0], [1.3, 0.5]])
    a, b = make_truth(cfg, belief, 300), make_truth(cfg2, other, 300)
    np.testing.assert_array_equal(a.at(x), b.at(x))
    assert not np.allclose(a.at(x), make_truth(cfg, belief, 301).at(x))


def test_clarity_rk4_matches_refined_whole_field_integration():
    cfg = load_config("configs/field_poam.yaml")
    env, _, x, q = make_scenario(cfg)
    p1 = x[:2] + jnp.array([0.10, -0.06])
    coarse = step(
        q, x[:2], p1, env.cells, cfg.sensor_sigma, cfg.sensor_R, cfg.clarity_decay, cfg.dt
    )
    fine = q
    for i in range(20):
        a = x[:2] + (p1 - x[:2]) * i / 20
        b = x[:2] + (p1 - x[:2]) * (i + 1) / 20
        fine = step(
            fine, a, b, env.cells, cfg.sensor_sigma, cfg.sensor_R, cfg.clarity_decay, cfg.dt / 20
        )
    np.testing.assert_allclose(coarse, fine, atol=5e-4)
