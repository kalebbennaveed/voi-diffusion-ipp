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
    snap = p.planning_belief(z)
    np.testing.assert_allclose(snap.joint(z)[0], mu, atol=2e-4)
    np.testing.assert_allclose(snap.joint(z)[1], cov, atol=2e-4)
