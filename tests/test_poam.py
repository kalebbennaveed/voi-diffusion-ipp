import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.wind.poam import POAM


def test_poam_kernel_psd_batch_statistics_and_projection():
    z = jnp.array([[-1.0, 0.0], [-0.3, 0.2], [0.4, -0.2], [1.0, 0.0]])
    x = z
    y = jnp.stack((jnp.sin(x[:, 0]), jnp.cos(x[:, 0])), axis=1)
    p = POAM(z, jax.random.PRNGKey(1))
    K = np.asarray(p.kernel(z, z))
    assert np.linalg.eigvalsh(K).min() > 0
    np.testing.assert_allclose(np.diag(K), 0.55**2, atol=1e-6)
    custom = POAM(z, jax.random.PRNGKey(1), prior_std=0.2)
    # Prediction solves add 1e-6 Cholesky jitter; allow that perturbation in float32.
    np.testing.assert_allclose(custom.predict(z)[1][:, 0, 0], 0.2**2, atol=2e-6)
    p.update(x[:2], y[:2], jax.random.PRNGKey(2), train_steps=0, relocate=False)
    p.update(x[2:], y[2:], jax.random.PRNGKey(3), train_steps=0, relocate=False)
    batch = POAM(z, jax.random.PRNGKey(1))
    batch.update(x, y, jax.random.PRNGKey(4), train_steps=0, relocate=False)
    np.testing.assert_allclose(p.mean, batch.mean, atol=2e-4)
    np.testing.assert_allclose(p.covariance, batch.covariance, atol=2e-4)
    # At inducing observations, sparse prediction agrees with exact GP.
    expected = K @ np.linalg.solve(K + 0.01 * np.eye(4), y)
    np.testing.assert_allclose(p.joint(z)[0], expected, atol=2e-4)
    losses = p.update(
        jnp.array([[0.8, 0.1]]), jnp.array([[0.7, 0.6]]), jax.random.PRNGKey(5), train_steps=3
    )
    assert np.isfinite(losses).all()
    mu, cov = p.joint(z)
    assert np.linalg.eigvalsh(cov).min() > -1e-6
    snap = p.planning_belief()
    np.testing.assert_allclose(snap.joint(z)[0], mu, atol=2e-4)
    np.testing.assert_allclose(snap.joint(z)[1], cov, atol=2e-4)


def test_full_prediction_retains_residual_and_matches_independent_algebra():
    z = jnp.array([[-1.0, -0.5], [1.0, 0.5]])
    p = POAM(z, jax.random.PRNGKey(8), spectral_count=128)
    x = jnp.array([[-0.4, 0.1], [0.5, -0.2], [2.0, -1.0]])
    y = jnp.array([[0.3, -0.2], [0.6, 0.1], [-0.4, 0.2]])
    p.update(x, y, jax.random.PRNGKey(9), train_steps=0, relocate=False)
    b = p.planning_belief()
    Kzz = np.asarray(p.kernel(z, z)) + np.eye(len(z)) * 1e-6
    A = np.asarray(p.kernel(x, z)) @ np.linalg.inv(Kzz)
    expected = (
        np.asarray(p.kernel(x, x))
        - A @ np.asarray(p.kernel(z, x))
        + A @ np.asarray(p.covariance) @ A.T
    )
    np.testing.assert_allclose(b.joint(x)[1], expected, atol=2e-6)
    residual = np.asarray(p.kernel(x, x)) - A @ np.asarray(p.kernel(z, x))
    assert np.diag(residual).min() > 0.05  # must not silently discard it
    np.testing.assert_allclose(b.predict(x)[1][:, 0, 0], np.diag(expected), atol=2e-6)
    _, sampled_cov = b.sample_moments(x)
    np.testing.assert_allclose(sampled_cov, expected, atol=0.012)


def test_hypothetical_poam_update_matches_real_frozen_kernel_update():
    z = jnp.array([[-1.0, 0.0], [0.0, 0.5], [1.0, 0.0]])
    p = POAM(z, jax.random.PRNGKey(17))
    p.update(z[:1], jnp.array([[0.4, -0.3]]), jax.random.PRNGKey(18), train_steps=2)
    b = p.planning_belief()
    x, y = jnp.array([[0.2, -0.1], [-0.7, 0.4]]), jnp.array([[0.1, 0.3], [0.4, -0.1]])
    hypothetical = jax.jit(lambda b: b.condition(x, y))(b)
    p.update(x, y, jax.random.PRNGKey(19), train_steps=0, relocate=False)
    np.testing.assert_allclose(hypothetical.mean, p.mean, atol=2e-5)
    np.testing.assert_allclose(hypothetical.covariance, p.covariance, atol=2e-5)
    np.testing.assert_allclose(hypothetical.joint(x)[1], p.joint(x)[1], atol=2e-5)


def test_decoupled_samples_coherent_differentiable_and_calibrated():
    from voi_diffusion_ipp.wind import WorldField

    z = jnp.array([[-1.0, 0.0], [0.0, 0.5], [1.0, 0.0]])
    p = POAM(z, jax.random.PRNGKey(21), spectral_count=64)
    p.update(
        z, jnp.array([[0.3, -0.2], [0.2, 0.4], [-0.4, 0.1]]), jax.random.PRNGKey(22), train_steps=0
    )
    b = p.planning_belief()
    x = jnp.array([[-0.5, -0.4], [0.3, 0.2], [1.6, -0.6]])
    coeff = b.sample_field(jax.random.PRNGKey(23), 4000)
    values = jax.vmap(lambda c: WorldField(b.basis, c).at(x))(coeff)
    mu, cov = b.joint(x)
    np.testing.assert_allclose(values.mean(0), mu, atol=0.025)
    np.testing.assert_allclose(np.cov(np.asarray(values[:, :, 0]).T), cov, atol=0.025)
    field = WorldField(b.basis, coeff[0])
    np.testing.assert_array_equal(field.at(x), field.at(x))
    grad = jax.jacfwd(lambda xx: field.at(xx)[0])(x[0])
    delta = 1e-3
    finite = np.stack(
        [
            (
                np.asarray(field.at(x[0] + jnp.eye(2)[d] * delta))[0]
                - np.asarray(field.at(x[0] - jnp.eye(2)[d] * delta))[0]
            )
            / (2 * delta)
            for d in range(2)
        ],
        axis=1,
    )
    np.testing.assert_allclose(grad, finite, atol=2e-3)


def test_contracted_field_evaluation_preserves_values_and_gradients():
    z = jnp.array([[-1.0, 0.0], [0.0, 0.5], [1.0, 0.0]])
    b = POAM(z, jax.random.PRNGKey(31), spectral_count=128).planning_belief()
    coeff = b.sample_field(jax.random.PRNGKey(32), 1)[0]
    x = jnp.array([[-0.3, 0.2], [1.4, -0.5]])
    expanded = lambda x: b.basis.features(x) @ coeff
    contracted = lambda x: b.basis.evaluate(x, coeff)
    np.testing.assert_allclose(expanded(x), contracted(x), atol=2e-6)
    np.testing.assert_allclose(jax.jacfwd(expanded)(x[0]), jax.jacfwd(contracted)(x[0]), atol=3e-6)


def test_support_corrected_sampler_matches_full_covariance_off_grid_after_learning():
    from voi_diffusion_ipp.config import load_config
    from voi_diffusion_ipp.experiments.scenarios import make_scenario, new_poam

    cfg = load_config("configs/field_poam.yaml")
    env, b, _, _ = make_scenario(cfg)
    p = new_poam(cfg, b.basis.centers)
    obs = jax.random.uniform(
        jax.random.PRNGKey(44), (48, 2), minval=env.bounds[0], maxval=env.bounds[1]
    )
    y = jnp.stack((jnp.sin(obs[:, 0]), 0.4 * jnp.cos(2 * obs[:, 1])), axis=1)
    p.update(obs, y, jax.random.PRNGKey(45), train_steps=5)
    b = p.planning_belief()
    query = jax.random.uniform(
        jax.random.PRNGKey(46), (90, 2), minval=env.bounds[0], maxval=env.bounds[1]
    )
    full = np.asarray(b.joint(query)[1])
    sampled = np.asarray(b.sample_moments(query)[1])
    assert np.linalg.norm(sampled - full) / np.linalg.norm(full) < 0.005
    assert np.max(np.abs(np.diag(sampled - full))) < 0.001
    # Residual spectral features survive conditioning on dense support.
    phi = b.basis.prior_features(query)
    assert np.linalg.norm(np.asarray(phi[:, :512])) > 0


def test_full_poam_rpcbf_gradient_matches_finite_differences():
    from voi_diffusion_ipp.config import load_config
    from voi_diffusion_ipp.experiments.scenarios import make_scenario
    from voi_diffusion_ipp.safety.rpcbf import constraints, values

    cfg = load_config("configs/field_poam.yaml")
    env, b, _, _ = make_scenario(cfg)
    eps = jax.random.normal(jax.random.PRNGKey(71), (cfg.inner_samples, *b.sample_shape))
    inner = b.inner_fields(eps)
    x = jnp.array([-1.35, 0.17, 0.14, -0.09])
    f = jax.jit(lambda x: values(x, b, inner, env, cfg))
    _, _, value, grad = jax.jit(lambda x: constraints(x, b, inner, env, cfg))(x)
    delta = 0.002
    fd = np.stack(
        [
            (np.asarray(f(x + jnp.eye(4)[i] * delta)) - np.asarray(f(x - jnp.eye(4)[i] * delta)))
            / (2 * delta)
            for i in range(4)
        ],
        axis=1,
    )
    np.testing.assert_allclose(grad, fd, rtol=0.025, atol=0.015)
    assert np.isfinite(value).all()
