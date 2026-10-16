"""Numerical checks for the explicitly experimental field cache."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from voi_diffusion_ipp.wind.poam import POAM

spec = importlib.util.spec_from_file_location(
    "runtime_profile", Path(__file__).resolve().parents[1] / "scripts/profile_runtime_redesign.py"
)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_cubic_cache_affine_values_gradients_and_domain():
    xx, yy = jnp.meshgrid(jnp.arange(9.0), jnp.arange(7.0))
    field = jnp.stack((2 * xx + 3 * yy, -xx + yy), axis=-1)
    basis = runtime.TableBasis(jnp.zeros(2), jnp.ones(2))
    for x in [jnp.array([3.0, 2.3]), jnp.array([3.15, 2.3])]:
        np.testing.assert_allclose(
            basis.evaluate(x, field)[0], [2 * x[0] + 3 * x[1], -x[0] + x[1]], atol=2e-6
        )
        grad = jax.jacfwd(lambda p: basis.evaluate(p, field)[0])(x)
        np.testing.assert_allclose(grad, [[2, 3], [-1, 1]], atol=3e-6)
    assert np.isnan(basis.evaluate(jnp.array([0.0, 2.0]), field)).all()
    assert np.isnan(basis.evaluate(jnp.array([9.0, 2.0]), field)).all()


@pytest.mark.parametrize("cache_kernel", [False, True])
def test_cached_prior_reused_across_actual_hypothetical_conditioning(cache_kernel):
    centers = jnp.array([[-1.0, -0.5], [-1.0, 0.5], [1.0, -0.5], [1.0, 0.5]])
    b = POAM(centers, jax.random.PRNGKey(40), spectral_count=4).planning_belief()
    eps = jax.random.normal(jax.random.PRNGKey(41), (2, *b.sample_shape))
    cache = jax.jit(runtime.prior_cache_builder(b, eps, 65, cache_kernel))(b)
    observed = jnp.array([[-0.8, 0.0], [0.0, 0.2]])
    post = b.condition(observed, jnp.array([[0.4, -0.2], [0.3, -0.1]]))
    # At interpolation nodes the complete field must equal the original sampler,
    # before AND after a posterior update, using the same unchanged prior cache.
    points = cache.basis.table.lower + cache.basis.table.spacing * jnp.array(
        [[29.0, 25.0], [35.0, 30.0]]
    )
    for belief in [b, post]:
        cached = cache.condition(belief)
        np.testing.assert_allclose(
            cached.at(points), belief.inner_fields(eps).at(points), rtol=3e-5, atol=3e-5
        )
    assert (
        float(jnp.linalg.norm(cache.condition(post).at(points) - cache.condition(b).at(points)))
        > 0.01
    )
