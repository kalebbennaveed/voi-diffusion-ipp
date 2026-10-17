import jax
import jax.numpy as jnp
import numpy as np
import pytest

from voi_diffusion_ipp.planning.sampling import couple_outer_samples


def test_iid_mode_and_safety_draws_are_unchanged():
    eps = jax.random.normal(jax.random.PRNGKey(181), (7, 9, 2))
    noise = jax.random.normal(jax.random.PRNGKey(182), (4, 50, 2))
    e, n = couple_outer_samples(eps, noise, 4)
    np.testing.assert_array_equal(e, eps)
    np.testing.assert_array_equal(n, noise)
    e, n = jax.jit(lambda e, n: couple_outer_samples(e, n, 4, 'antithetic'))(eps, noise)
    np.testing.assert_array_equal(e[4:], eps[4:])
    np.testing.assert_array_equal(e[:4:2], eps[:4:2])
    np.testing.assert_array_equal(e[1:4:2], -e[:4:2])
    np.testing.assert_array_equal(n[1::2], -n[::2])


def test_outer_pair_preserves_gaussian_quadratic_expectation():
    # A known nonlinear integrand checks that coupling retains the integral;
    # mean cancellation alone would not detect collapsing the variance.
    e = jax.random.normal(jax.random.PRNGKey(183), (8192, 3, 1, 2))
    n = jax.random.normal(jax.random.PRNGKey(184), (8192, 2, 1, 2))
    paired, ns = jax.vmap(lambda e, n: couple_outer_samples(e, n, 2, 'antithetic'))(e, n)
    value = (paired[:, :2] + 0.4 * ns) ** 2
    np.testing.assert_allclose(value.mean(), 1.16, atol=.03)
    np.testing.assert_allclose(paired[:, :2].mean((0, 2, 3)), 0, atol=.03)
    np.testing.assert_allclose(paired[:, :2].var((0, 2, 3)), 1, atol=.04)
    np.testing.assert_array_equal(paired[:, 2], e[:, 2])


def test_invalid_modes_and_shapes_fail_explicitly():
    e, n = jnp.zeros((5, 3, 2)), jnp.zeros((2, 4, 2))
    with pytest.raises(ValueError):
        couple_outer_samples(e, n, 2, 'unknown')
    with pytest.raises(ValueError):
        couple_outer_samples(e, n, 1, 'antithetic')
    with pytest.raises(ValueError):
        couple_outer_samples(e, n, 4, 'antithetic')
