"""Independent algebra and solver checks for the active airflow experiment.

These check implementation, not calibration or a mission safety guarantee.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/airflow'))
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import linprog, minimize

import rmse_monitoring as rm
from robot_review import FixedWind, load_wind
from waypoint_poam import condition, train
from observed_poam import _double
from voi_diffusion_ipp.environment import Environment
from voi_diffusion_ipp.wind.poam import POAM, attentive_kernel
from voi_diffusion_ipp.safety.qp import solve


def small_belief():
    z = jnp.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    return _double(POAM(z, jax.random.PRNGKey(412), prior_std=1.8,
                        prior_support=z, spectral_count=8).planning_belief())


def numpy_kernel(params, x, y, lengths):
    """Independent NumPy AK formula from POAM section III-B."""
    p = {k: np.asarray(v) for k, v in params.items()}
    def weights(x):
        z = np.tanh(np.tanh(x @ p['w1'] + p['b1']) @ p['w2'] + p['b2'])
        logits = z @ p['w3'] + p['b3']
        w = np.exp(logits - logits.max(axis=1, keepdims=True))
        return w / np.linalg.norm(w, axis=1, keepdims=True)
    wx, wy = weights(x), weights(y)
    d2 = ((x[:, None] - y[None]) ** 2).sum(axis=2)
    out = np.zeros_like(d2)
    for k, length in enumerate(lengths):
        out += wx[:, k, None] * wy[None, :, k] * np.exp(-d2 / (2 * length**2))
    return np.exp(p['log_amplitude']) * (wx @ wy.T) * out


def test_spatial_only_field_revisits_and_rejected_time_inputs(tmp_path):
    b = small_belief()
    x = jnp.array([[.15, .26], [.7, .4], [.15, .26]])
    f = FixedWind(jnp.arange(24., dtype=jnp.float64).reshape(3, 4, 2), .5)
    np.testing.assert_array_equal(f.at(x)[0], f.at(x)[2])
    np.testing.assert_array_equal(f.at(x[::-1])[::-1], f.at(x))
    c = b.sample_field(jax.random.PRNGKey(42), 1)[0]
    draw = b.basis.evaluate(x, c)
    np.testing.assert_array_equal(draw[0], draw[2])
    np.testing.assert_array_equal(b.basis.evaluate(x[::-1], c)[::-1], draw)
    mask = jnp.zeros(4, bool)
    for fn in (lambda: f.at(jnp.zeros((1, 3))),
               lambda: b.predict(jnp.zeros((1, 3))),
               lambda: condition(b, jnp.zeros((1, 3)), jnp.zeros((1, 2)), mask),
               lambda: condition(b, jnp.zeros((1, 2)), jnp.zeros((1, 1)), mask)):
        with pytest.raises(ValueError, match='time|Spatial|observations'):
            fn()
    import json
    root = tmp_path / 'data/scenario1'
    root.mkdir(parents=True)
    (root / 'metadata.json').write_text(json.dumps({'time_dependence': 'periodic'}))
    with pytest.raises(ValueError, match='fixed spatial'):
        load_wind(1, tmp_path)


def test_active_repeated_observations_match_independent_batch_sparse_gp():
    b = small_belief()
    # Repeated measurements of the same fixed field, without clock or forgetting.
    z = np.array(b.basis.centers)
    x = z[[0, 1, 0, 3, 2, 3, 1, 0]]
    y = np.c_[np.sin(x[:, 0]) + .2 * x[:, 1], np.cos(x[:, 1]) - .3 * x[:, 0]]
    mask = jnp.zeros(4, bool)
    before = b
    for xi, yi in zip(x, y):
        b, mask = condition(b, jnp.asarray(xi[None]), jnp.asarray(yi[None]), mask)
    z = np.asarray(b.basis.centers)
    K = numpy_kernel(b.basis.parameters, z, z, np.asarray(b.basis.lengths))
    Kuf = numpy_kernel(b.basis.parameters, z, x, np.asarray(b.basis.lengths))
    np.testing.assert_allclose(b.B, Kuf @ Kuf.T, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(b.a, Kuf @ y, rtol=1e-11, atol=1e-10)
    C = K + Kuf @ Kuf.T / .01 + 1e-6 * np.eye(len(z))
    mean = K @ np.linalg.solve(C, Kuf @ y) / .01
    covariance = K @ np.linalg.solve(C, K)
    np.testing.assert_allclose(b.mean, mean, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(b.covariance, covariance, rtol=1e-8, atol=1e-9)
    query = np.array([[.2, .5], [.8, .8], [1.4, -.1]])
    Kxz = numpy_kernel(b.basis.parameters, query, z, np.asarray(b.basis.lengths))
    A = np.linalg.solve(K + 1e-6 * np.eye(4), Kxz.T).T
    expected = numpy_kernel(b.basis.parameters, query, query, np.asarray(b.basis.lengths))
    expected += A @ covariance @ A.T - A @ Kxz.T
    mu, cov = b.joint(jnp.asarray(query))
    np.testing.assert_allclose(mu, A @ mean, atol=1e-9)
    np.testing.assert_allclose(cov, expected, atol=1e-9)
    assert np.all(np.diag(cov) < np.diag(before.joint(query)[1]))


def test_active_relocation_and_kernel_training_keep_full_statistics_and_caches():
    b = small_belief()
    mask = jnp.zeros(4, bool)
    b, mask = condition(b, jnp.array([[.2, .1]]), jnp.array([[.3, -.2]]), mask)
    x, y = jnp.array([[3., 2.]]), jnp.array([[.6, .4]])
    out, _ = condition(b, x, y, mask)
    assert not np.array_equal(np.sort(out.basis.centers, axis=0), np.sort(b.basis.centers, axis=0))
    Kcross = numpy_kernel(b.basis.parameters, np.array(b.basis.centers),
                          np.array(out.basis.centers), np.array(b.basis.lengths))
    P = np.linalg.solve(np.array(b.old_K) + 1e-6 * np.eye(4), Kcross)
    Knewdata = numpy_kernel(b.basis.parameters, np.array(out.basis.centers), np.array(x), np.array(b.basis.lengths))
    np.testing.assert_allclose(out.a, P.T @ b.a + Knewdata @ y, atol=1e-9)
    np.testing.assert_allclose(out.B, P.T @ b.B @ P + Knewdata @ Knewdata.T, atol=1e-9)
    moments = jax.tree.map(jnp.zeros_like, out.basis.parameters)
    trained, _, _, step, loss = train(out, x, y, jnp.array(2.), moments, moments, jnp.array(0))
    assert int(step) == 4 and np.isfinite(loss).all()
    z = trained.basis.centers
    np.testing.assert_allclose(trained.old_K, trained.kernel(z, z), atol=1e-12)
    query = jnp.array([[.2, .5], [2., 2.]])
    np.testing.assert_allclose(trained.basis.kernel_centers(query), trained.kernel(query, z), atol=1e-12)
    np.testing.assert_allclose(trained.predict(query)[1][:, 0, 0], np.diag(trained.joint(query)[1]), atol=1e-11)
    assert np.linalg.eigvalsh(trained.covariance).min() > 0


def test_active_aerodynamic_policy_value_gradient():
    env = Environment(jnp.zeros((1, 2)), jnp.ones(1),
        jnp.array([[2.34,4.72,.98],[3.59,4.45,.98],[2.8,3.65,.98],
                   [3.6,2.6,.855],[5.355,1.1625,.855]]),
        jnp.array([[0.,0.],[8.,6.]]), jnp.zeros(2), jnp.zeros(2), jnp.zeros(2))
    basis, pts, shape = rm.table_grid(env.bounds)
    mu = jnp.stack((.15 + .03*pts[:, 0], -.1 + .02*pts[:, 1]), axis=1).reshape(shape)
    mean = rm.TableMean(basis, mu)
    fields = rm.InnerFields(basis, jnp.stack((mu+jnp.array([.12,-.05]), mu-jnp.array([.12,-.05]))))
    cfg = rm.settings()
    value = jax.jit(lambda x: rm.values(x, mean, fields, env, cfg))
    derivative = jax.jit(jax.jacfwd(value))
    for x in (jnp.array([4.7,3.3,.2,-.1]), jnp.array([1.2,.8,.4,.2])):
        step = 1e-5
        finite = np.stack([(value(x+jnp.eye(4)[k]*step)-value(x-jnp.eye(4)[k]*step))/(2*step)
                           for k in range(4)], axis=1)
        np.testing.assert_allclose(derivative(x), finite, rtol=3e-4, atol=3e-5)


def test_hard_qp_and_minimum_slack_against_independent_solvers():
    rng = np.random.default_rng(19011)
    solver = jax.jit(lambda u, A, b: solve(u, A, b, 1.6, jnp.zeros(2)))
    for i in range(60):
        A = rng.normal(size=(13, 2))
        # Include duplicate/zero inequalities, feasible and infeasible polytopes.
        if i % 3 == 0:
            A[1] = A[0]
            A[2] = 0
        anchor = rng.uniform(-1.2, 1.2, size=2)
        b = A @ anchor + rng.uniform(.02, 1., size=13)
        if i % 2:
            b -= 2.
        u = rng.normal(size=2)*2
        result = solver(jnp.asarray(u), jnp.asarray(A), jnp.asarray(b))
        lp = linprog([0., 0., 1.], A_ub=np.c_[A, -np.ones(13)], b_ub=b,
                     bounds=[(-1.6, 1.6)]*2+[(0, None)], method='highs')
        assert lp.success
        np.testing.assert_allclose(result.slack, lp.x[2], atol=3e-5)
        assert bool(result.feasible) == (lp.x[2] < 1e-8)
        assert np.isfinite(result.control).all()
        assert np.max(np.abs(result.control)) <= 1.6+2e-5
        assert np.max(A @ result.control - b) <= result.slack+6e-5
        if result.feasible:
            reference = minimize(lambda v: .5*np.sum((v-u)**2), lp.x[:2], jac=lambda v:v-u,
                constraints={'type':'ineq', 'fun':lambda v:b-A@v, 'jac':lambda v:-A},
                bounds=[(-1.6,1.6)]*2, method='SLSQP', options={'ftol':1e-11,'maxiter':100})
            assert reference.success
            np.testing.assert_allclose(result.control, reference.x, atol=3e-5)
