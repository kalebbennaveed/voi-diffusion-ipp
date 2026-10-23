"""Finite sensing geometry and the monitoring dynamics use the same footprint."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.quarter.environment import Settings
from voi_diffusion_ipp.quarter.sensing import rate, step


def scene(cells, rectangles=()):
    return SimpleNamespace(
        cells=jnp.array(cells, dtype=float),
        rectangles=jnp.array(rectangles, dtype=float).reshape(-1, 5),
        process_noise=jnp.full(len(cells), .05),
    )


def test_narrower_camera_excludes_cells_previously_inside_the_footprint():
    env = scene([[1.4, 0.], [1.6, 0.], [3.3, 0.]])
    q = jnp.full(3, .4)
    position = jnp.zeros(2)
    narrow = np.asarray(rate(q, position, env, Settings(sensor_half_angle_degrees=25.), 3.2))
    wide = np.asarray(rate(q, position, env, Settings(sensor_half_angle_degrees=45.), 3.2))
    decay = -.05 * .4**2

    # At 3.2 m altitude the two radii are 1.492 m and 3.2 m.
    assert narrow[0] > 0 and wide[0] > 0
    np.testing.assert_allclose(narrow[1:], decay, atol=1e-8)
    assert wide[1] > 0
    np.testing.assert_allclose(wide[2], decay, atol=1e-8)


def test_footprint_scales_with_altitude_and_disappears_below_ground():
    env = scene([[.8, 0.], [1.2, 0.]])
    cfg = Settings(sensor_half_angle_degrees=25.)
    q = jnp.zeros(2)
    low = np.asarray(rate(q, jnp.zeros(2), env, cfg, 2.))
    high = np.asarray(rate(q, jnp.zeros(2), env, cfg, 3.2))
    assert low[0] > 0 and low[1] == 0
    assert np.all(high > 0)
    np.testing.assert_array_equal(rate(q, jnp.zeros(2), env, cfg, -1.), 0.)


def test_building_blocks_sensing_even_inside_the_camera_cone():
    env = scene([[1., 0.], [0., 1.]], [[.4, -.2, .6, .2, 4.]])
    q = jnp.full(2, .4)
    derivative = np.asarray(rate(q, jnp.zeros(2), env, Settings(sensor_half_angle_degrees=25.), 3.2))
    np.testing.assert_allclose(derivative[0], -.05 * .4**2, atol=1e-8)
    assert derivative[1] > 0


def test_static_sensing_integration_converges_and_unseen_cells_decay():
    env = scene([[1., 0.], [2., 0.]])
    cfg = Settings(sensor_half_angle_degrees=25.)
    position = jnp.zeros(2)
    initial = jnp.full(2, .4)
    coarse = step(initial, position, position, env, cfg, .1, 3.2)
    fine = initial
    for _ in range(10):
        fine = step(fine, position, position, env, cfg, .01, 3.2)

    np.testing.assert_allclose(coarse, fine, atol=2e-6, rtol=0.)
    assert coarse[0] > initial[0] and coarse[1] < initial[1]
    # Outside the cone, dq/dt = -Q q^2 has an independent closed-form solution.
    np.testing.assert_allclose(coarse[1], .4 / (1 + .05 * .4 * .1), atol=1e-7)
