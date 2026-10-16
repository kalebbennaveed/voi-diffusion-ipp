import jax.numpy as jnp
import numpy as np
from voi_diffusion_ipp.clarity import step, deficit
from voi_diffusion_ipp.dynamics import step as dynamics_step


def test_constant_wind_integrator():
    x = dynamics_step(jnp.zeros(4), jnp.array([1.,0]), lambda p: jnp.array([0.,2.]), .2)
    np.testing.assert_allclose(x, [.02,.04,.2,.4], atol=1e-7)


def test_clarity_analytic_decay_and_observation():
    q = jnp.array([.6]); cells = jnp.array([[0.,0.]])
    for _ in range(100):
        q = step(q, jnp.array([100.,100.]), jnp.array([100.,100.]), cells, .4, .2, .1, .01)
    np.testing.assert_allclose(q, [.6/(1+.06)], atol=2e-6)
    q = jnp.array([.2])
    for _ in range(100):
        q = step(q, jnp.zeros(2), jnp.zeros(2), cells, .4, 1., 0., .01)
    np.testing.assert_allclose(q, [1-.8/1.8], atol=2e-6)
    assert float(deficit(jnp.array([.8,.1]), jnp.array([.5,.5]))) == np.float32(.2)
