"""Independent matrix identities for route-targeted POAM survey reduction."""
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/airflow'))
from survey_information import (prepare_targets, target_variance_reduction,
                                prepare_full_targets, full_target_variance_reduction)


def fixture():
    rng = np.random.default_rng(42)
    n, m = 17, 6
    features = rng.normal(size=(n, m)) * .2
    root = rng.normal(size=(m, m))
    covariance = root @ root.T / m + np.eye(m) * .3
    residual = np.linspace(.1, .6, n)
    b = SimpleNamespace(
        basis=SimpleNamespace(kernel_centers=lambda p: jnp.asarray(features)[p[:, 0].astype(int)]),
        inverse=jnp.eye(m), old_K=jnp.eye(m), covariance=jnp.asarray(covariance), noise_std=.2,
    )
    def predict(p):
        indices = p[:, 0].astype(int)
        selected = jnp.asarray(features)[indices]
        variance = jnp.einsum('ni,ij,nj->n', selected, b.covariance, selected) + jnp.asarray(residual)[indices]
        return jnp.zeros((len(p), 2)), variance[:, None, None] * jnp.eye(2)[None]
    b.predict = predict
    def kernel(p, r):
        i, j = p[:, 0].astype(int), r[:, 0].astype(int)
        f = jnp.asarray(features)
        return f[i] @ f[j].T + (i[:, None] == j[None, :]) * jnp.asarray(residual)[i, None]
    def joint(p):
        i = p[:, 0].astype(int)
        f = jnp.asarray(features)[i]
        covariance = f @ b.covariance @ f.T + (i[:, None] == i[None, :]) * jnp.asarray(residual)[i, None]
        return jnp.zeros((len(p), 2)), covariance
    b.kernel, b.joint = kernel, joint
    return b, features, covariance, residual


def locations(indices):
    return jnp.c_[jnp.asarray(indices), jnp.zeros(len(indices))]


def test_matches_independent_precision_update_and_preserves_residual():
    b, A, S, residual = fixture()
    observations = np.array([1, 2, 6, 7, 7, 10])
    target_ids = np.array([0, 3, 8, 11, 15])
    weights = np.array([.1, 0, .3, .5, .1])
    prepared = prepare_targets(b, locations(target_ids), weights)
    actual = target_variance_reduction(b, locations(observations), prepared)
    posterior = np.linalg.inv(np.linalg.inv(S) + A[observations].T @ A[observations] / b.noise_std**2)
    before = residual[target_ids] + np.einsum('ni,ij,nj->n', A[target_ids], S, A[target_ids])
    after = residual[target_ids] + np.einsum('ni,ij,nj->n', A[target_ids], posterior, A[target_ids])
    expected = 2 * weights @ (before - after)
    np.testing.assert_allclose(actual.absolute, expected, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(prepared.total_variance, 2 * weights @ before, rtol=2e-5)
    assert float(actual.fraction_total) < float(actual.fraction_reducible)


def test_irrelevant_target_weights_give_exactly_zero_exploration_benefit():
    b, *_ = fixture()
    targets = prepare_targets(b, locations([0, 3, 8, 11]), jnp.zeros(4))
    reduction = target_variance_reduction(b, locations([1, 2, 6, 7, 10]), targets)
    np.testing.assert_array_equal(np.asarray(reduction), np.zeros(3))


def test_repeated_measurements_have_diminishing_benefit_and_known_latents_give_none():
    b, *_ = fixture()
    targets = prepare_targets(b, locations([0, 3, 8, 11]), jnp.ones(4) / 4)
    gain = [float(target_variance_reduction(b, locations([1] * n), targets).absolute) for n in (1, 2, 3)]
    assert 0 < gain[0] < gain[1] < gain[2] < float(targets.reducible_variance)
    assert gain[2] - gain[1] < gain[1] - gain[0]
    b.covariance = jnp.zeros_like(b.covariance)
    targets = prepare_targets(b, locations([0, 3, 8, 11]), jnp.ones(4) / 4)
    reduction = target_variance_reduction(b, locations([1, 2, 6, 7]), targets)
    assert float(targets.total_variance) > 0  # The physical GP residual remains.
    assert float(reduction.absolute) == 0


def test_batched_paths_equal_individual_reduction():
    b, *_ = fixture()
    targets = prepare_targets(b, locations([0, 3, 8, 11]), jnp.ones(4) / 4)
    paths = jnp.stack([locations([1, 2, 6, 7]), locations([4, 4, 4, 4])])
    batch = jax.jit(jax.vmap(lambda p: target_variance_reduction(b, p, targets)))(paths)
    individual = [target_variance_reduction(b, p, targets) for p in paths]
    np.testing.assert_allclose(batch.absolute, [v.absolute for v in individual], rtol=2e-5)


def test_matches_actual_fixed_inducing_poam_update_with_numerical_tolerance():
    from voi_diffusion_ipp.wind.poam import POAM
    rng = np.random.default_rng(31)
    centers = rng.uniform([0, 0], [8, 4.625], size=(16, 2))
    points = jnp.asarray(rng.uniform([0, 0], [8, 4.625], size=(10, 2)))
    observations = jnp.asarray(rng.uniform([0, 0], [8, 4.625], size=(12, 2)))
    belief = POAM(centers, jax.random.PRNGKey(4), prior_std=1.8).planning_belief()
    targets = prepare_targets(belief, points, jnp.ones(len(points)) / len(points))
    reduction = target_variance_reduction(belief, observations, targets)
    updated = belief.condition(observations, jnp.zeros_like(observations), relocate=False)
    _, before = belief.predict(points)
    _, after = updated.predict(points)
    direct = 2 * jnp.mean(before[:, 0, 0] - after[:, 0, 0])
    # Unchanged POAM uses a 1e-6 jitter in kernel solves and statistic transport.
    np.testing.assert_allclose(reduction.absolute, direct, rtol=2e-4, atol=2e-5)


def test_full_predictive_gp_reduction_matches_joint_schur_complement():
    b, *_ = fixture()
    points, obs = locations([0, 2, 8, 11, 15]), locations([1, 2, 6, 7, 7, 10])
    weights = jnp.array([.1, .2, .4, .2, .1])
    targets = prepare_full_targets(b, points, weights)
    reduction = full_target_variance_reduction(b, obs, targets)
    _, covariance = b.joint(jnp.concatenate((points, obs)))
    covariance = np.asarray(covariance)
    count = len(points)
    cross = covariance[:count, count:]
    innovation = covariance[count:, count:] + b.noise_std**2 * np.eye(len(obs))
    expected = 2 * np.asarray(weights) @ np.diag(cross @ np.linalg.solve(innovation, cross.T))
    np.testing.assert_allclose(reduction.absolute, expected, rtol=2e-5)
    zero = full_target_variance_reduction(b, obs, prepare_full_targets(b, points, jnp.zeros(len(points))))
    np.testing.assert_array_equal(np.asarray(zero), np.zeros(2))


def test_full_gp_objective_can_value_residual_information_without_claiming_poam_gain():
    b, *_ = fixture()
    b.covariance = jnp.zeros_like(b.covariance)
    points = locations([0, 2, 8, 11])
    weights = jnp.ones(4) / 4
    fixed = target_variance_reduction(b, points, prepare_targets(b, points, weights))
    full = full_target_variance_reduction(b, points, prepare_full_targets(b, points, weights))
    assert float(fixed.absolute) == 0
    assert float(full.absolute) > 0
    assert 0 < float(full.fraction_total) < 1


def test_full_gain_uses_stable_solves_for_concentrated_inducing_support():
    """Compare float32 evaluation to an independent float64 Schur complement.

    Concentrated observed support produced a condition number above four
    million in the failed physical pilot. This finite Gaussian fixture has a
    comparable condition number and repeated noisy observations. Its stored
    inverse is deliberately unusable; the objective must derive projections
    from the covariance factor rather than that rounded matrix.
    """
    rng = np.random.default_rng(812)
    m, nt, nx = 24, 12, 25
    orthogonal = np.linalg.qr(rng.normal(size=(m, m)))[0]
    K = (orthogonal * np.geomspace(2e-5, 80., m)) @ orthogonal.T
    F = rng.normal(size=(nt+nx, m)) * .15
    F[nt:] = F[nt] + rng.normal(size=(nx, m)) * 1e-4
    cross = F @ K
    full_prior = F @ K @ F.T + np.eye(nt+nx)*.003
    S = np.eye(m)*.002
    point_ids = np.arange(nt)
    observation_ids = np.arange(nt, nt+nx)
    b = SimpleNamespace(
        old_K=jnp.asarray(K, dtype=jnp.float32),
        inverse=jnp.full((m, m), jnp.nan, dtype=jnp.float32),
        covariance=jnp.asarray(S, dtype=jnp.float32), noise_std=.1,
        basis=SimpleNamespace(kernel_centers=lambda p: jnp.asarray(cross, dtype=jnp.float32)[p[:, 0].astype(int)]),
        kernel=lambda p, r: jnp.asarray(full_prior, dtype=jnp.float32)[p[:, 0].astype(int)[:, None], r[:, 0].astype(int)[None, :]],
    )
    weights = jnp.ones(nt, dtype=jnp.float32)/nt
    prepared = prepare_full_targets(b, locations(point_ids), weights)
    got = full_target_variance_reduction(b, locations(observation_ids), prepared)
    A = np.linalg.solve(K+1e-6*np.eye(m), cross.T).T
    covariance = full_prior-A@cross.T+A@S@A.T
    C = covariance[nt:, nt:]+.01*np.eye(nx)
    TX = covariance[:nt, nt:]
    expected = 2*np.mean(np.diag(TX@np.linalg.solve(C, TX.T)))
    assert np.isfinite(np.asarray(got)).all()
    np.testing.assert_allclose(got.absolute, expected, rtol=.02, atol=2e-6)
