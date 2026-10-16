from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.environment import Environment
from voi_diffusion_ipp.planning.budgeted import evaluate, future_reference, reference_matrices
from voi_diffusion_ipp.wind.poam import POAM


def fixture():
    cfg = replace(
        Config(),
        linear_drag=1.0,
        stage_steps=2,
        future_steps=15,
        backup_steps=3,
        worlds=1,
        inner_samples=1,
        dt=0.2,
        reject_collisions=False,
    )
    belief = POAM(
        jnp.array([[-1.0, -0.5], [-1.0, 0.5], [1.0, -0.5], [1.0, 0.5]]),
        jax.random.PRNGKey(801),
        spectral_count=4,
    ).planning_belief()
    env = Environment(
        jnp.array([[-0.5, 0.0], [0.5, 0.0]]),
        jnp.full(2, 0.7),
        jnp.array([[0.0, 2.0, 0.2]]),
        jnp.array([[-4.0, -4.0], [4.0, 4.0]]),
        jnp.array([-0.5, 0.0]),
        jnp.array([0.5, 0.0]),
        jnp.zeros(2),
        jnp.full(2, 0.16),
    )
    return cfg, belief, env


def test_clamped_reference_uses_measured_initial_state_and_posterior_only():
    cfg, b, env = fixture()
    x = jnp.array([-0.5, 0.1, 0.2, -0.1])
    vector = jnp.zeros(18).at[12:].set(jnp.array([0.2, 0.1, 0.3, -0.1, 0.4, 0.2]))
    matrix = reference_matrices(9.0, 45)
    before = future_reference(vector, x, b, env, matrix, cfg)
    after = b.condition(jnp.array([[0.0, 0.0]]), jnp.array([[0.8, -0.5]]))
    changed = future_reference(vector, x, after, env, matrix, cfg)
    np.testing.assert_allclose(before[0, 0], x[:2], atol=1e-7)
    np.testing.assert_allclose(before[0, 1], x[2:], atol=1e-7)
    np.testing.assert_allclose(changed[0, :2], before[0, :2], atol=1e-7)
    assert float(jnp.linalg.norm(changed[1:, 0] - before[1:, 0])) > 0.01


def test_causal_update_and_future_noise_cannot_change_earlier_execution():
    cfg, b, env = fixture()
    eps = jax.random.normal(jax.random.PRNGKey(802), (2, *b.sample_shape))
    vector = jnp.zeros(18).at[12:].set(0.1)
    vector = vector.at[4:12].set(jnp.tile(jnp.array([jnp.arctanh(-0.5 / 3.88), 0.0]), 4))
    x = jnp.array([-0.5, 0.0, 0.0, 0.0])
    q = jnp.array([0.7, 0.7])
    noise = jnp.zeros((1, 17, 2))
    fn = jax.jit(lambda n: evaluate(vector, x, q, b, None, n, env, cfg, True, True, eps))
    actual = fn(noise)
    changed = fn(noise.at[:, 2:].set(10.0))
    # Future observations are recorded but never available within that frozen stage.
    np.testing.assert_array_equal(actual[1][0].controls, changed[1][0].controls)
    np.testing.assert_array_equal(actual[1][1].controls, changed[1][1].controls)
    passive = jax.jit(lambda: evaluate(vector, x, q, b, None, noise, env, cfg, False, True, eps))()
    np.testing.assert_array_equal(actual[1][0].x, passive[1][0].x)
    assert not np.allclose(actual[1][1].controls, passive[1][1].controls)
    assert np.isfinite(actual[0])
