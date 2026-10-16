import jax
import jax.numpy as jnp
import numpy as np
from voi_diffusion_ipp.wind import prior, WorldField


def test_condition_matches_batch_and_uses_values():
    b = prior([[0, 0], [1, 0], [2, 0]])
    p = jnp.array([[0.3, 0.1], [0.8, 0.0]])
    y = jnp.array([[0.8, -0.3], [0.6, 0.1]])
    H = np.asarray(b.basis.features(p)); S = np.asarray(b.covariance)
    K = S @ H.T @ np.linalg.inv(H @ S @ H.T + .01*np.eye(2))
    post = jax.jit(lambda b: b.condition(p, y))(b)
    np.testing.assert_allclose(post.mean, K @ y, atol=2e-6)
    np.testing.assert_allclose(post.covariance, S-K @ H @ S, atol=2e-6)
    assert np.linalg.eigvalsh(post.covariance).min() > 0
    assert not np.allclose(post.mean, b.condition(p, -y).mean)


def test_coherent_field_and_cross_covariance():
    b = prior([[0, 0], [1, 0], [2, 0]])
    p = jnp.array([[0.2, 0], [0.25, 0]])
    coeff = b.sample_field(jax.random.PRNGKey(8), 8000)
    values = jax.vmap(lambda c: WorldField(b.basis, c).at(p))(coeff)
    np.testing.assert_allclose(np.cov(np.asarray(values[:,:,0]).T), b.joint(p)[1], atol=.015)
    f = WorldField(b.basis, coeff[0])
    np.testing.assert_array_equal(f.at(p), f.at(p))
    assert b.condition(p[:1], jnp.zeros((1,2))).joint(p)[1][1,1] < b.joint(p)[1][1,1]
