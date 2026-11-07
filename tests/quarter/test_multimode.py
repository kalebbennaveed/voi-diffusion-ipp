"""Mode-local MBD equations, rollout budgets and numerical recovery."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from voi_diffusion_ipp.quarter.diffusion import cosine_schedule
from voi_diffusion_ipp.quarter.mbd_core import score_and_update
from voi_diffusion_ipp.quarter.multimode import (
    mode_score_and_update, search, unique_rows_mask,
)


def settings(**changes):
    values = dict(iterations=6, proposals=24, latent_bound=12., cosine_end=.35,
                  batch_size=8, archive_size=4, temperature=.7)
    values.update(changes)
    return SimpleNamespace(**values)


def test_one_mode_score_is_identical_to_ordinary_mbd():
    rng = np.random.default_rng(17)
    y = jnp.asarray(rng.normal(size=12))
    samples = jnp.asarray(rng.normal(size=(19, 12)))
    logw = jnp.asarray(-rng.uniform(0, 5, 19)).at[2].set(-jnp.inf)
    ordinary = score_and_update(y, samples, logw, .84, .39)
    modes = mode_score_and_update(y[None], samples, logw, jnp.zeros(19, int), .84, .39)
    for expected, actual in zip(ordinary, (modes[0][0], modes[1][0], modes[2], modes[3][0])):
        np.testing.assert_array_equal(actual, expected)


def test_incompatible_modes_are_never_averaged_together():
    y = jnp.array([[-4., 0.], [4., 0.]])
    samples = jnp.array([[-5., 1.], [5., 1.], [-3., -1.], [3., -1.]])
    updated, scores, weights, valid = mode_score_and_update(
        y, samples, jnp.zeros(4), jnp.array([0, 1, 0, 1]), .8, .4)
    for mode, indices in enumerate(([0, 2], [1, 3])):
        expected = score_and_update(y[mode], samples[jnp.array(indices)], jnp.zeros(2), .8, .4)
        np.testing.assert_allclose(updated[mode], expected[0], atol=1e-7)
        np.testing.assert_allclose(scores[mode], expected[1], atol=1e-7)
    np.testing.assert_allclose(weights, .5)
    assert np.all(np.asarray(valid))
    assert float(updated[0, 0]) < -2 and float(updated[1, 0]) > 2


def test_empty_mode_does_not_contaminate_another_modes_score():
    y = jnp.array([[-2., 0.], [2., 0.]])
    samples = jnp.array([[jnp.nan, 0.], [3., 1.]])
    updated, scores, weights, valid = mode_score_and_update(
        y, samples, jnp.array([-jnp.inf, 0.]), jnp.array([0, 1]), .8, .4)
    np.testing.assert_array_equal(valid, [False, True])
    np.testing.assert_array_equal(updated[0], y[0])
    np.testing.assert_array_equal(scores[0], [0., 0.])
    np.testing.assert_array_equal(weights, [0., 1.])
    assert np.all(np.isfinite(updated))


def test_extremely_small_finite_likelihood_keeps_positive_normalized_weight():
    updated, scores, weights, valid = mode_score_and_update(
        jnp.zeros((1, 2)), jnp.array([[2., 3.], [4., 5.]]),
        jnp.array([-1e35, -jnp.inf]), jnp.zeros(2, dtype=int), .8, .4)
    np.testing.assert_array_equal(weights, [1., 0.])
    np.testing.assert_array_equal(valid, [True])
    assert np.all(np.isfinite(updated)) and np.all(np.isfinite(scores))


def test_search_retains_both_route_modes_and_every_reverse_equation():
    cfg = settings(proposals=25, cosine_end=.2, temperature=.15)
    seeds = jnp.array([[-3., 0.], [3., 0.]])
    evaluate = lambda theta: jnp.array([(jnp.abs(theta[0])-2.4)**2+theta[1]**2])
    result = search(evaluate, seeds, jnp.zeros(2), jax.random.PRNGKey(41), cfg, 1., 0., True)
    assert result.finalists.shape == (5, 2)
    np.testing.assert_allclose(result.finalist_field_costs,jax.vmap(evaluate)(result.finalists),rtol=1e-6,atol=1e-7)
    assert float(result.finalists[0, 0]) < -2
    assert float(result.finalists[1, 0]) > 2
    assert np.all(np.asarray(jax.vmap(evaluate)(result.finalists[:2])) < .03)
    alpha, abar = cosine_schedule(cfg.iterations, cfg.cosine_end)
    assignment = np.asarray(result.proposal_modes)
    assert np.bincount(assignment).tolist() == [13, 12]
    for step in range(cfg.iterations):
        i = cfg.iterations-1-step
        for mode in range(2):
            keep = assignment == mode
            expected = score_and_update(
                result.before[step, mode], result.samples[step, keep],
                -result.sample_costs[step, keep]/cfg.temperature, alpha[i], abar[i])
            np.testing.assert_allclose(result.after[step, mode], expected[0], atol=3e-6)
            # Masked versus compact float32 reductions differ slightly; the
            # final score amplifies that difference by 1/(1-cumulative_alpha).
            np.testing.assert_allclose(result.scores[step, mode], expected[1], atol=8e-5)
            np.testing.assert_allclose(result.weights[step, keep], expected[2], atol=1e-7)
    np.testing.assert_array_equal(result.evaluation_counts, [3, cfg.iterations*cfg.proposals, 2])
    assert int(result.evaluations) == 3+cfg.iterations*cfg.proposals+2


def test_no_positive_weight_restarts_within_the_same_bounded_mode():
    cfg = settings(iterations=3, proposals=8, latent_bound=.001, cosine_end=.8)
    seeds = jnp.array([[-.0005, 0.], [.0005, 0.]])
    evaluate = lambda theta: jnp.array([jnp.sum(theta**2)])
    result = search(evaluate, seeds, jnp.zeros(2), jax.random.PRNGKey(7), cfg, 1., .4, True)
    np.testing.assert_array_equal(np.asarray(result.mode_trace)[:, :, 6], np.ones((3, 2)))
    np.testing.assert_array_equal(result.weights, np.zeros((3, 8)))
    np.testing.assert_array_equal(result.scores, np.zeros((3, 2, 2)))
    np.testing.assert_allclose(result.mode_finals, seeds, atol=1e-9)
    assert np.all(np.isfinite(result.finalists))
    assert np.max(np.abs(result.mode_finals)) <= cfg.latent_bound


def test_nonfinite_seed_recovery_uses_its_own_sanitized_seed():
    cfg = settings(iterations=1, proposals=4, latent_bound=.01, cosine_end=.8)
    seeds = jnp.array([[0., 0.], [jnp.nan, .005]])
    result = search(lambda theta: jnp.array([jnp.sum(theta**2)]), seeds, jnp.zeros(2),
                    jax.random.PRNGKey(3), cfg, 1., 0., True)
    assert np.all(np.isfinite(result.mode_finals))
    np.testing.assert_allclose(result.mode_finals[1], [0., .005])
    assert np.max(np.abs(result.mode_finals)) <= cfg.latent_bound


def test_entirely_undefined_objective_returns_a_bounded_recovered_decision():
    cfg = settings(iterations=2, proposals=4, latent_bound=.01, cosine_end=.8)
    seeds = jnp.array([[jnp.nan, .005], [jnp.inf, -.005]])
    result = search(lambda theta: jnp.array([jnp.nan]), seeds, jnp.zeros(2),
                    jax.random.PRNGKey(3), cfg, 1., 0., True)
    assert np.all(np.isfinite(result.final))
    assert np.all(np.isfinite(result.mode_finals))
    assert np.all(np.isfinite(result.finalists))
    assert np.max(np.abs(result.final)) <= cfg.latent_bound
    np.testing.assert_array_equal(np.asarray(result.mode_trace)[:, :, 6], np.ones((2, 2)))


def test_corrected_archive_density_is_local_to_each_mode():
    cfg = settings(iterations=7, proposals=32, cosine_end=.45, temperature=.5)
    seeds = jnp.array([[-1., 0.], [1., 0.]])
    evaluate = lambda theta: jnp.array([jnp.sum((theta-1.)**2), jnp.sum((theta+2.)**2)])
    result = search(evaluate, seeds, jnp.zeros(2), jax.random.PRNGKey(12), cfg, .5, .4, True)
    _, abar = cosine_schedule(cfg.iterations, cfg.cosine_end)
    assert np.any(np.asarray(result.mode_trace)[:, :, 3] > 0)
    for step in range(cfg.iterations):
        sigma = float(result.trace[step, 5])
        for mode in range(2):
            mask = np.asarray(result.proposal_modes) == mode
            samples = np.asarray(result.samples[step])[mask]
            center = np.asarray(result.before[step, mode])/np.sqrt(float(abar[-1-step]))
            archive = np.asarray(result.archive_centers[step, mode])
            nu = np.asarray(result.archive_probabilities[step, mode])
            rho = float(result.mode_trace[step, mode, 3])
            # Common Gaussian normalizer cancels, including when rho=0.
            ordinary = -.5*np.sum(((samples-center)/sigma)**2, axis=1)
            local = -.5*np.sum(((samples[:, None]-archive)/sigma)**2, axis=2)
            with np.errstate(divide='ignore'):
                mixture = logsumexp(local+np.log(nu)[None], axis=1)
                denominator = np.logaddexp(np.log1p(-rho)+ordinary,
                                           np.log(rho)+mixture)
            expected = ordinary-denominator
            np.testing.assert_allclose(result.log_importance_ratios[step, mask], expected,
                                       atol=6e-6)
            cost = np.asarray(result.sample_costs[step])[mask]
            logw = -cost/cfg.temperature+expected
            weight = np.exp(logw-logsumexp(logw))
            np.testing.assert_allclose(result.weights[step, mask], weight, atol=2e-6)


def test_exact_duplicate_finalists_have_one_validation_entry():
    rows = jnp.array([[1., 2.], [3., 4.], [1., 2.], [1., 2.+1e-5], [3., 4.]])
    np.testing.assert_array_equal(unique_rows_mask(rows), [True, True, False, True, False])
    cfg = settings(iterations=1, proposals=4, latent_bound=1e-5, cosine_end=.8)
    result = search(lambda theta: jnp.array([jnp.sum(theta**2)]), jnp.zeros((2, 2)),
                    jnp.zeros(2), jax.random.PRNGKey(4), cfg, 1., 0., True)
    assert int(jnp.sum(result.finalist_unique)) == 1


def test_full_hundred_coordinate_search_is_finite_and_objective_budgets_match():
    # H=10 s, dt=.2 s, two independent controls at each interval.
    dim = 2*round(10/.2)
    cfg = settings(iterations=6, proposals=24, cosine_end=.35, batch_size=24)
    target = jnp.linspace(-.3, .3, dim)
    evaluate = lambda theta: jnp.array([jnp.sum((theta-target)**2),
                                        jnp.sum((theta+target)**2)+.5])
    seeds = jnp.stack((jnp.zeros(dim), jnp.full(dim, -.5),
                       jnp.full(dim, .5), jnp.sin(jnp.arange(dim))*.2))
    args = (evaluate, seeds, jnp.zeros(dim), jax.random.PRNGKey(8), cfg)
    optimistic = jax.jit(lambda: search(*args, .5, .4, True))()
    expected = jax.jit(lambda: search(*args, 1., 0., True))()
    assert optimistic.final.shape == (dim,)
    assert optimistic.mode_finals.shape == (4, dim)
    assert optimistic.samples.shape == (6, 24, dim)
    assert np.all(np.isfinite(optimistic.final))
    assert np.all(np.isfinite(optimistic.mode_finals))
    assert np.all(np.std(np.asarray(optimistic.samples).reshape(-1, dim), axis=0) > .05)
    np.testing.assert_array_equal(optimistic.evaluation_counts, expected.evaluation_counts)
    np.testing.assert_array_equal(optimistic.evaluations, expected.evaluations)


def test_batch_size_changes_neither_samples_nor_selection():
    cfg = settings(iterations=3, proposals=16, batch_size=4)
    evaluate = lambda theta: jnp.array([jnp.sum((theta-.2)**2), jnp.sum((theta+.3)**2)])
    args = (evaluate, jnp.array([[-1., 0.], [1., 0.]]), jnp.zeros(2), jax.random.PRNGKey(9))
    first = search(*args, cfg, .5, .4, True)
    second_cfg = settings(**{**vars(cfg), 'batch_size': 16})
    second = search(*args, second_cfg, .5, .4, True)
    for name in first._fields:
        np.testing.assert_allclose(getattr(first, name), getattr(second, name),
                                   rtol=2e-6, atol=2e-6, err_msg=name)


def test_more_modes_than_proposals_is_rejected():
    with pytest.raises(ValueError, match='at least one proposal'):
        search(lambda theta: jnp.array([0.]), jnp.zeros((3, 2)), jnp.zeros(2),
               jax.random.PRNGKey(0), settings(proposals=2), 1., 0., True)


@pytest.mark.parametrize('eta,rho', [(1., 0.), (.5, .4)])
def test_cached_initial_fields_preserve_every_search_result_and_count_once(eta, rho):
    cfg = settings(iterations=3, proposals=8, batch_size=4)
    seeds = jnp.array([[-1., .3], [1., -.2]])
    previous = jnp.array([.2, -.1])
    def evaluate(theta):
        return jnp.array([jnp.sum((theta-.2)**2), jnp.sum((theta+.3)**2)])
    key = jax.random.PRNGKey(123)
    initial = jnp.concatenate((seeds, previous[None]))
    cached = jax.lax.map(evaluate, initial, batch_size=cfg.batch_size)
    ordinary = search(evaluate, seeds, previous, key, cfg, eta, rho, True)
    reused = search(evaluate, seeds, previous, key, cfg, eta, rho, True,
                    initial_field_values=cached)
    for name in ordinary._fields:
        np.testing.assert_array_equal(getattr(reused, name), getattr(ordinary, name), err_msg=name)
    assert int(reused.evaluations) == len(initial)+cfg.iterations*cfg.proposals+len(seeds)


def test_cached_initial_fields_retain_parameter_validity_checks():
    cfg = settings(iterations=1, proposals=4, latent_bound=.1)
    seeds = jnp.array([[0., .01], [2., .02]])
    previous = jnp.zeros(2)
    evaluate = lambda theta: jnp.array([0.])
    args = (evaluate, seeds, previous, jax.random.PRNGKey(3), cfg, 1., 0., True)
    ordinary = search(*args)
    reused = search(*args, initial_field_values=jnp.zeros((3, 1)))
    for name in ordinary._fields:
        np.testing.assert_array_equal(getattr(reused, name), getattr(ordinary, name), err_msg=name)
    with pytest.raises(ValueError, match='Cached initial fields'):
        search(*args, initial_field_values=jnp.zeros((2, 1)))
