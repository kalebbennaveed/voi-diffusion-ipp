from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.planning.mbd import optimize, schedule, score, target_objective


def test_bounded_target_score_matches_independent_quadrature():
    bound, ab, lam, y = 0.7, 0.65, 0.2, 0.8
    cfg = replace(Config(), mbd_latent_bound=bound)
    objective = target_objective(lambda v: (v[0] - 0.2) ** 2, cfg)
    key = jax.random.PRNGKey(978001)
    samples = y / np.sqrt(ab) + np.sqrt((1 - ab) / ab) * jax.random.normal(key, (180000, 1))
    costs = jax.vmap(objective)(samples)
    estimated, w = score(jnp.array([y]), samples, costs, ab, lam)
    density = lambda x: np.exp(-0.5 * (y - np.sqrt(ab) * x) ** 2 / (1 - ab) - (x - 0.2) ** 2 / lam)
    norm = quad(density, -bound, bound)[0]
    moment = quad(lambda x: x * density(x), -bound, bound)[0] / norm
    expected = (-y + np.sqrt(ab) * moment) / (1 - ab)
    se = np.sqrt(
        np.sum((np.asarray(w) * (np.asarray(samples[:, 0]) - float(w @ samples[:, 0]))) ** 2)
    )
    assert abs(float(estimated[0]) - expected) < 5 * se * np.sqrt(ab) / (1 - ab)
    assert np.all(np.asarray(w)[np.abs(np.asarray(samples[:, 0])) > bound] == 0)
    # The raw proposal distribution is unchanged; invalid samples are not clipped.
    assert float(jnp.max(jnp.abs(samples))) > bound * 2


def test_bound_is_enforced_for_initial_candidates_and_final_selection():
    cfg = replace(Config(), mbd_latent_bound=0.6, diffusion_steps=4, proposals=20)
    result = jax.jit(lambda k: optimize(lambda y: (y[0] - 2) ** 2, jnp.zeros(1), k, cfg))(
        jax.random.PRNGKey(978002)
    )
    assert np.all(np.abs(result.vector) <= 0.6)
    invalid = np.any(np.abs(np.asarray(result.history.candidates)) > 0.6, axis=-1)
    assert invalid.any()
    assert np.all(np.asarray(result.history.weights)[invalid] == 0)
    assert np.isfinite(result.cost)


def test_budget_comparison_preserves_noise_endpoints_and_legacy_schedule():
    old_a, old_ab = schedule(4, 0.005, 0.18)
    a, ab = schedule(4, 0.005, 0.18, 4)
    np.testing.assert_array_equal(a, old_a)
    np.testing.assert_array_equal(ab, old_ab)
    for levels in (8, 16):
        a, ab = schedule(levels, 0.005, 0.18, 4)
        np.testing.assert_allclose(ab[jnp.array([0, -1])], old_ab[jnp.array([0, -1])], atol=1e-7)
        np.testing.assert_allclose(np.cumprod(a), ab, atol=2e-7)
        assert np.all(np.diff(ab) < 0)


def test_all_invalid_level_keeps_iterate_and_reports_no_effective_samples():
    cfg = replace(Config(), diffusion_steps=4, proposals=8)
    result = jax.jit(lambda k: optimize(lambda y: jnp.inf, jnp.ones(2), k, cfg))(
        jax.random.PRNGKey(990001)
    )
    np.testing.assert_array_equal(result.history.noisy_after, result.history.noisy_before)
    np.testing.assert_array_equal(result.history.weights, 0)
    np.testing.assert_array_equal(result.trace[:, 2], 0)
    np.testing.assert_array_equal(result.vector, jnp.ones(2))
    assert np.isinf(result.cost)  # A finite vector is not a successful optimization.
