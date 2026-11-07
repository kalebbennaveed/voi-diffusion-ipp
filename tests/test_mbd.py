from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.planning.mbd import optimize, schedule, score, weights


def test_score_matches_analytic_gaussian_convolution():
    # p0=N(target, lambda I); convolved p_i=N(sqrt(ab)*target, ab*lambda+1-ab).
    ab = 0.65
    lam = 0.4
    y = jnp.array([0.8, -0.4])
    target = jnp.array([-0.2, 0.7])
    candidates = y / jnp.sqrt(ab) + jnp.sqrt((1 - ab) / ab) * jax.random.normal(
        jax.random.PRNGKey(2), (80000, 2)
    )
    cost = 0.5 * jnp.sum((candidates - target) ** 2, axis=1)
    s, w = score(y, candidates, cost, ab, lam)
    expected = -(y - jnp.sqrt(ab) * target) / (ab * lam + 1 - ab)
    mean = w @ candidates
    se = (
        jnp.sqrt(jnp.sum((w[:, None] * (candidates - mean)) ** 2, axis=0)) * jnp.sqrt(ab) / (1 - ab)
    )
    assert np.all(np.abs(s - expected) < 5 * np.asarray(se))


def test_schedule_optimization_budget_and_invalid():
    a, ab = schedule(8, 0.005, 0.2)
    np.testing.assert_allclose(ab, np.cumprod(a), atol=1e-7)
    assert np.all(np.diff(ab) < 0)
    cfg = replace(Config(), diffusion_steps=14, proposals=80, temperature=0.015, candidate_chunk=8)
    f = lambda y: jnp.sum((y - jnp.array([0.5, -0.3])) ** 2)
    for method in ("mbd", "cem", "mppi"):
        r = jax.jit(lambda k, method=method: optimize(f, jnp.array([-1.0, 1.0]), k, cfg, method))(
            jax.random.PRNGKey(7)
        )
        assert r.cost < 0.025
        assert int(r.evaluations) == 1122
        assert np.all(np.diff(r.trace[:, 0]) <= 1e-7)
    np.testing.assert_array_equal(weights(jnp.array([jnp.inf, jnp.nan]), 0.1), [0, 0])
    r = optimize(
        lambda y: jnp.inf, jnp.zeros(2), jax.random.PRNGKey(1), replace(cfg, diffusion_steps=2)
    )
    assert np.isfinite(r.vector).all() and np.isinf(r.cost)


def test_one_dimensional_quadratic_reproducible():
    cfg = replace(Config(), diffusion_steps=12, proposals=64, temperature=0.02, candidate_chunk=8)
    f = lambda y: (y[0] - 0.7) ** 2
    run = jax.jit(lambda k: optimize(f, jnp.array([-1.0]), k, cfg))
    a = run(jax.random.PRNGKey(33))
    b = run(jax.random.PRNGKey(33))
    np.testing.assert_array_equal(a.vector, b.vector)
    assert a.cost < 0.01 and a.iterates.shape == (12, 1)


def test_cem_elites_exclude_infeasible_candidates():
    cfg = replace(Config(), diffusion_steps=5, proposals=12, candidate_chunk=1)
    f = lambda y: jnp.where((y[0] >= 0) & (y[0] <= 0.1), (y[0] - 0.07) ** 2, jnp.inf)
    result = optimize(f, jnp.array([0.05]), jax.random.PRNGKey(8), cfg, "cem")
    assert np.isfinite(result.cost)
    assert np.all((np.asarray(result.iterates) >= 0) & (np.asarray(result.iterates) <= 0.1))


def test_recorded_diffusion_is_the_actual_score_and_reverse_update():
    cfg = replace(Config(), diffusion_steps=4, proposals=12, candidate_chunk=1)
    objective = lambda y: jnp.sum((y - 0.4) ** 2)
    r = optimize(objective, jnp.zeros(3), jax.random.PRNGKey(81), cfg)
    h = r.history
    np.testing.assert_allclose(h.costs, jax.vmap(jax.vmap(objective))(h.candidates), atol=1e-7)
    np.testing.assert_allclose(h.weights.sum(axis=1), 1, atol=1e-6)
    for i in range(cfg.diffusion_steps):
        s, w = score(h.noisy_before[i], h.candidates[i], h.costs[i], h.alphabar[i], cfg.temperature)
        # Separate eager/JIT evaluations round the cancellation numerator in
        # float32 differently; division by 1-alphabar amplifies that error.
        np.testing.assert_allclose(h.score[i], s, atol=2e-5, rtol=1e-5)
        np.testing.assert_allclose(h.weights[i], w, atol=2e-6)
        expected = (h.noisy_before[i] + (1 - h.alphabar[i]) * s) / jnp.sqrt(h.alpha[i])
        np.testing.assert_allclose(h.noisy_after[i], expected, atol=2e-6)
    np.testing.assert_array_equal(h.noisy_before[1:], h.noisy_after[:-1])


def test_multistart_is_exact_independent_mbd_and_preserves_modes():
    from voi_diffusion_ipp.planning.mbd import optimize_multistart
    cfg=replace(Config(),diffusion_steps=4,proposals=16,candidate_chunk=1)
    starts=jnp.array([[-1.2,0.], [1.1,0.]])
    def objective(v):
        # Disconnected feasible basins. Averaging the two modes is infeasible.
        return jnp.where(jnp.abs(v[0])>.6,jnp.minimum(jnp.sum((v-jnp.array([-1.,.2]))**2),.1+jnp.sum((v-jnp.array([1.,-.2]))**2)),jnp.inf)
    key=jax.random.PRNGKey(8127)
    result=optimize_multistart(objective,starts,key,cfg)
    independent=[optimize(objective,start,k,replace(cfg,proposals=8)) for start,k in zip(starts,jax.random.split(key,2))]
    winner=min(independent,key=lambda r:float(r.cost))
    np.testing.assert_allclose(result.vector,winner.vector,atol=1e-7)
    np.testing.assert_allclose(result.cost,winner.cost,atol=1e-7)
    assert int(result.evaluations)==sum(int(r.evaluations) for r in independent)
    for i,r in enumerate(independent):
        np.testing.assert_allclose(result.history.candidates[i],r.history.candidates,atol=1e-7)
        np.testing.assert_allclose(result.history.noisy_after[i],r.history.noisy_after,atol=1e-7)
    assert abs(float(result.vector[0]))>.6
