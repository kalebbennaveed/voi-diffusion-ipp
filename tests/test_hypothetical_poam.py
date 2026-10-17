import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.wind.cache import build_prior_cache
from voi_diffusion_ipp.wind.poam import POAM


def test_hypothetical_relocation_matches_real_poam_statistic_transport():
    centers = jnp.array([[-1.0, -0.5], [-1.0, 0.5], [1.0, -0.5], [1.0, 0.5]])
    model = POAM(centers, jax.random.PRNGKey(9401), spectral_count=4)
    # Accumulated old information must be transported, not discarded.
    model.update(
        jnp.array([[-0.8, 0.3], [-0.5, -0.4]]),
        jnp.array([[0.2, -0.1], [0.3, 0.4]]),
        jax.random.PRNGKey(9402),
        train_steps=2,
    )
    before = model.planning_belief()
    x = jnp.array([[1.8, 0.8], [2.1, 0.1], [2.0, -1.4]])
    y = jnp.array([[1.0, -0.5], [0.8, -0.3], [0.6, -0.2]])
    hypothetical = jax.jit(lambda b: b.condition(x, y, relocate=True))(before)
    model.update(x, y, jax.random.PRNGKey(9403), train_steps=0, relocate=True)
    actual = model.planning_belief()
    np.testing.assert_allclose(hypothetical.basis.centers, actual.basis.centers, atol=1e-6)
    assert not np.array_equal(before.basis.centers, actual.basis.centers)
    for name in ["mean", "covariance", "a", "B", "old_K", "inverse"]:
        np.testing.assert_allclose(
            getattr(hypothetical, name), getattr(actual, name), atol=2e-5, rtol=2e-5
        )
    query = jnp.concatenate((x, centers))
    for h, a in zip(hypothetical.joint(query), actual.joint(query), strict=True):
        np.testing.assert_allclose(h, a, atol=2e-5, rtol=2e-5)
    eps = jax.random.normal(jax.random.PRNGKey(9404), (2, *before.sample_shape))
    cache = jax.jit(lambda b, e: build_prior_cache(b, e, 129, True))(before, eps)
    shifted = cache.relocate(hypothetical)
    reused = cache.relocate(hypothetical, x)
    np.testing.assert_allclose(
        reused.basis.kernel_grid, shifted.basis.kernel_grid, atol=2e-7, rtol=2e-6
    )
    mean, fields = shifted.tabulate(hypothetical)
    np.testing.assert_allclose(mean.mean_at(query), hypothetical.mean_at(query), atol=0.002)
    np.testing.assert_allclose(
        fields.at(query), hypothetical.inner_fields(eps).at(query), atol=0.008
    )


def test_poam_data_initialization_uses_only_available_locations():
    x = jnp.array([[-1.0, 0.0], [-0.4, 0.5], [0.0, 0.7], [0.5, 0.3], [1.0, 0.0]])
    y = jnp.stack((x[:, 0] ** 2, x[:, 1]), axis=-1)
    model = POAM.from_data(x, y, jax.random.PRNGKey(9405), 4, train_steps=0, spectral_count=4)
    assert np.all(np.any(np.all(np.array(model.z)[:, None] == np.array(x)[None], axis=-1), axis=1))
    np.testing.assert_array_equal(model.data_x, x)
    assert np.all(np.isfinite(model.predict(x)[0]))
