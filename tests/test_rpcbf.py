from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.environment import Environment
from voi_diffusion_ipp.safety.qp import solve
from voi_diffusion_ipp.safety.rpcbf import filter_control, values
from voi_diffusion_ipp.safety.spline import maximum
from voi_diffusion_ipp.wind import prior


def fixture():
    env = Environment(
        jnp.zeros((1, 2)),
        jnp.ones(1),
        jnp.array([[0.0, 1.0, 0.5]]),
        jnp.array([[-3.0, -3.0], [1.0, 3.0]]),
        jnp.zeros(2),
        jnp.zeros(2),
        jnp.zeros(2),
    )
    b = prior([[0, 0], [1, 0]], std=0.2)
    cfg = replace(Config(), backup_steps=12)
    inn = b.inner_fields(jax.random.normal(jax.random.PRNGKey(1), (4, 2, 2)))
    return env, b, cfg, inn


def test_spline_matches_scipy_and_stopping_gradient():
    t = jnp.arange(9) * 0.23
    f = lambda v: maximum(-0.7 + v * t - 0.5 * t * t)
    np.testing.assert_allclose(f(0.83), -0.7 + 0.83**2 / 2, atol=2e-6)
    np.testing.assert_allclose(jax.grad(f)(0.83), 0.83, atol=2e-5)
    rng = np.random.default_rng(3)
    for _ in range(8):
        y = rng.normal(size=9).astype("float32")
        spl = CubicSpline(np.arange(9), y)
        roots = spl.derivative().roots(extrapolate=False)
        expected = max(y.max(), np.max(spl(roots), initial=-np.inf))
        np.testing.assert_allclose(maximum(jnp.array(y)), expected, atol=2e-5)


def test_qp_independent_solver_and_infeasible():
    rng = np.random.default_rng(2)
    for _ in range(15):
        A = rng.normal(size=(4, 2))
        b = rng.uniform(0.1, 1, size=4)
        u = rng.normal(size=2) * 2
        sol = solve(jnp.array(u), jnp.array(A), jnp.array(b), 1.0, jnp.zeros(2))
        ref = minimize(
            lambda x, u=u: 0.5 * np.sum((x - u) ** 2),
            np.zeros(2),
            jac=lambda x, u=u: x - u,
            constraints={
                "type": "ineq",
                "fun": lambda x, b=b, A=A: b - A @ x,
                "jac": lambda x, A=A: -A,
            },
            bounds=[(-1, 1)] * 2,
            tol=1e-10,
        )
        assert sol.feasible and ref.success
        np.testing.assert_allclose(sol.control, ref.x, atol=3e-5)
    sol = solve(
        jnp.zeros(2),
        jnp.array([[1.0, 0.0], [-1.0, 0.0]]),
        jnp.array([-2.0, -2.0]),
        1.0,
        jnp.zeros(2),
    )
    assert not sol.feasible and float(sol.slack) > 1.99
    assert np.max(np.abs(sol.control)) <= 1.0001


def test_rpcbf_gradient_and_intervention():
    env, b, cfg, inn = fixture()
    x = jnp.array([0.45, 0.0, 0.8, 0.05])
    f = jax.jit(lambda z: values(z, b, inn, env, cfg))
    grad = jax.jacfwd(f)(x)
    eps = 2e-3
    fd = np.stack(
        [(f(x + jnp.eye(4)[i] * eps) - f(x - jnp.eye(4)[i] * eps)) / (2 * eps) for i in range(4)],
        axis=1,
    )
    np.testing.assert_allclose(grad, fd, atol=2e-3, rtol=0.02)
    u, info = jax.jit(lambda x: filter_control(x, jnp.array([1.6, 0]), b, inn, env, cfg))(x)
    assert np.isfinite(u).all() and u[0] < 1.5 and info.correction > 0.1
    if info.feasible:
        assert info.residual < 5e-5
