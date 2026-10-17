from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from test_budgeted import fixture

from voi_diffusion_ipp.planning import covariance
from voi_diffusion_ipp.wind.cache import build_prior_cache
from voi_diffusion_ipp.wind.covariance_forecast import CovarianceForecast, posterior_cross
from voi_diffusion_ipp.wind.poam import POAM


def test_full_predictive_conditioning_matches_dense_gaussian_and_sequential_updates():
    _, b, _ = fixture()
    z = jnp.array([[0.1, 0.3], [0.6, -0.3], [0.1, 0.3]])
    query = jnp.array([[-0.1, 0.4], [0.5, 0.0], [2.0, 1.1]])
    original = np.asarray(b.joint(jnp.concatenate((query, z)))[1], dtype=np.float64)
    n = len(query)
    expected = original[:n, :n] - original[:n, n:] @ np.linalg.solve(
        original[n:, n:] + (float(b.noise_std) ** 2 + 1e-7) * np.eye(len(z)), original[n:, :n]
    )
    initial = CovarianceForecast.empty(b, 5)
    np.testing.assert_allclose(initial.joint(query)[1], original[:n, :n], atol=2e-6)
    together = jax.jit(lambda f: f.condition_locations(z))(initial)
    sequential = initial
    for p in z:
        sequential = sequential.condition_locations(p[None])
    for updated in (together, sequential):
        mean, cov = updated.joint(query)
        np.testing.assert_array_equal(mean, b.mean_at(query))
        np.testing.assert_allclose(cov, expected, atol=3e-6, rtol=3e-5)
        np.testing.assert_allclose(updated.predict(query)[1][:, 0, 0], np.diag(expected), atol=3e-6)
        assert np.linalg.eigvalsh(np.asarray(cov)).min() > -2e-6
        assert np.linalg.eigvalsh(original[:n, :n] - cov).min() > -2e-6
    # An off-inducing-point observation removes predictive *residual* uncertainty.
    k = b.basis.kernel_centers(z)
    residual = b.kernel(z, z) - k @ b.inverse @ k.T
    assert float(jnp.diag(residual).max()) > 0.05
    assert float(together.predict(z)[1][0, 0, 0]) < 0.01


def test_forecast_matches_variational_poam_for_inducing_observations_only():
    _, b, _ = fixture()
    z = b.basis.centers
    query = jnp.array([[0.2, 0.4], [1.1, -0.3]])
    forecast = CovarianceForecast.empty(b, len(z)).condition_locations(z)
    refit = b.condition(z, b.mean_at(z), relocate=False)
    # Residual vanishes at inducing inputs, so both Gaussian updates agree there.
    np.testing.assert_allclose(forecast.joint(query)[1], refit.joint(query)[1], atol=3e-5)
    # Away from inducing inputs variational refitting and full GP conditioning
    # are different operations. Do not hide that by removing the residual.
    remote = jnp.array([[0.2, 0.1]])
    full = CovarianceForecast.empty(b, 1).condition_locations(remote)
    sparse = b.condition(remote, b.mean_at(remote), relocate=False)
    assert float(sparse.predict(remote)[1][0, 0, 0] - full.predict(remote)[1][0, 0, 0]) > 0.05


def test_conditional_samples_remain_spatially_coherent_and_match_covariance():
    z = jnp.array([[-1.0, -0.5], [-1.0, 0.5], [1.0, -0.5], [1.0, 0.5]])
    query = jnp.array([[0.1, 0.1], [0.6, -0.3], [-0.8, 0.4]])
    model = POAM(
        z, jax.random.PRNGKey(9701), spectral_count=32, prior_support=jnp.concatenate((z, query))
    )
    b = model.planning_belief()
    forecast = CovarianceForecast.empty(b, 2).condition_locations(query[:1])
    k1, k2 = jax.random.split(jax.random.PRNGKey(9702))
    eps = jax.random.normal(k1, (5000, *b.sample_shape))
    noise = jax.random.normal(k2, (5000, 2, 2))
    fields = jax.jit(lambda e, n: forecast.inner_fields(b.inner_fields(e), n))(eps, noise)
    values = np.asarray(jax.jit(lambda f: f.at(query))(fields))
    target = np.asarray(forecast.joint(query)[1])
    np.testing.assert_allclose(np.cov(values[:, :, 0].T), target, atol=0.02, rtol=0.09)
    np.testing.assert_allclose(values.mean(0), b.mean_at(query), atol=0.025)
    np.testing.assert_allclose(fields.at(query[:1]), values[:, :1], atol=2e-6)
    gradient = jax.jacfwd(lambda x: fields.basis.evaluate(x, fields.coefficients[0])[0])(query[0])
    assert np.isfinite(gradient).all()


def test_cached_conditional_samples_and_gradients_match_direct_fields():
    _, b, _ = fixture()
    forecast = CovarianceForecast.empty(b, 4).condition_locations(
        jnp.array([[0.1, 0.2], [0.5, -0.3]])
    )
    eps = jax.random.normal(jax.random.PRNGKey(9703), (2, *b.sample_shape))
    noise = jax.random.normal(jax.random.PRNGKey(9704), (2, 4, 2))
    cache = jax.jit(lambda b, e: build_prior_cache(b, e, 129, True))(b, eps)
    mean, inner = cache.tabulate(b)
    _, cached = jax.jit(lambda f: f.tabulate(mean, inner, noise))(forecast)
    exact = forecast.inner_fields(b.inner_fields(eps), noise)
    query = jnp.array([[-0.2, 0.2], [0.3, -0.1], [1.1, 0.8]])
    np.testing.assert_allclose(cached.at(query), exact.at(query), atol=0.006)

    def grad(f):
        return jax.vmap(jax.jacfwd(lambda x: f.basis.evaluate(x, f.coefficients[0])[0]))(query)

    np.testing.assert_allclose(grad(cached), grad(exact), atol=0.03)


def test_covariance_rollout_causality_and_frozen_mean():
    cfg, b, env = fixture()
    cfg = replace(cfg, budgeted_update_steps=5)
    capacity = cfg.stage_steps + cfg.future_steps - 5
    eps = jax.random.normal(jax.random.PRNGKey(9705), (cfg.inner_samples, *b.sample_shape))
    noise = jax.random.normal(jax.random.PRNGKey(9706), (cfg.inner_samples, capacity, 2))
    x, q, v = jnp.array([-0.5, 0.0, 0.0, 0.0]), jnp.array([0.7, 0.7]), jnp.zeros(18)

    def run(anticipate, n):
        return covariance.evaluate(v, x, q, b, None, eps, n, env, cfg, anticipate, True)

    active = jax.jit(lambda n: run(True, n))(noise)
    frozen = jax.jit(lambda n: run(False, n))(noise)
    altered = jax.jit(lambda n: run(True, n))(noise.at[:, cfg.stage_steps :].set(20))
    for other in (frozen, altered):
        np.testing.assert_array_equal(active[1][0].x, other[1][0].x)
        np.testing.assert_array_equal(active[1][0].q, other[1][0].q)
    np.testing.assert_array_equal(active[1][1].controls[:, :5], altered[1][1].controls[:, :5])
    np.testing.assert_array_equal(active[1][1].x[:, 0], active[1][0].x[:, -1])
    np.testing.assert_array_equal(active[1][1].q[:, 0], active[1][0].q[:, -1])
    assert np.isfinite(active[0])
    np.testing.assert_allclose(
        posterior_cross(b, env.cells, env.cells), b.joint(env.cells)[1], atol=1e-6
    )
