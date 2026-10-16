import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.clarity import deficit, step
from voi_diffusion_ipp.dynamics import step as dynamics_step


def test_constant_wind_integrator():
    x = dynamics_step(jnp.zeros(4), jnp.array([1.0, 0]), lambda p: jnp.array([0.0, 2.0]), 0.2)
    np.testing.assert_allclose(x, [0.02, 0.04, 0.2, 0.4], atol=1e-7)


def test_clarity_analytic_decay_and_observation():
    q = jnp.array([0.6])
    cells = jnp.array([[0.0, 0.0]])
    for _ in range(100):
        q = step(
            q, jnp.array([100.0, 100.0]), jnp.array([100.0, 100.0]), cells, 0.4, 0.2, 0.1, 0.01
        )
    np.testing.assert_allclose(q, [0.6 / (1 + 0.06)], atol=2e-6)
    q = jnp.array([0.2])
    for _ in range(100):
        q = step(q, jnp.zeros(2), jnp.zeros(2), cells, 0.4, 1.0, 0.0, 0.01)
    np.testing.assert_allclose(q, [1 - 0.8 / 1.8], atol=2e-6)
    assert float(deficit(jnp.array([0.8, 0.1]), jnp.array([0.5, 0.5]))) == np.float32(0.2)
