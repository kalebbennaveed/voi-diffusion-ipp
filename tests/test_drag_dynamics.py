from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.dynamics import derivative, step
from voi_diffusion_ipp.environment import Environment
from voi_diffusion_ipp.planning.rollout import rollout
from voi_diffusion_ipp.safety.rpcbf import constraints, values
from voi_diffusion_ipp.wind import WorldField, prior


def test_drag_matches_independent_closed_form_and_zero_drag_limit():
    x = jnp.array([0.2, -0.4, 0.7, 0.3])
    u = jnp.array([0.4, -0.2])
    d = jnp.array([-0.1, 0.35])
    drag, dt, n = 0.9, 0.05, 40
    actual = jax.lax.fori_loop(0, n, lambda _, z: step(z, u, lambda p: d, dt, drag), x)
    t = dt * n
    equilibrium = np.array(u + d) / drag
    exponential = np.exp(-drag * t)
    velocity = equilibrium + (x[2:] - equilibrium) * exponential
    position = x[:2] + equilibrium * t + (x[2:] - equilibrium) * (1 - exponential) / drag
    np.testing.assert_allclose(actual, np.r_[position, velocity], atol=2e-6)
    zero = step(x, u, lambda p: d, dt)
    np.testing.assert_array_equal(zero, step(x, u, lambda p: d, dt, 0.0))
    np.testing.assert_allclose(
        zero,
        np.r_[x[:2] + x[2:] * dt + (u + d) * dt**2 / 2, x[2:] + (u + d) * dt],
        atol=1e-7,
    )


def test_rpcbf_drag_drift_matches_directional_derivative_and_rollout():
    cfg = replace(Config(), linear_drag=0.9, backup_steps=4, stage_steps=2, inner_samples=2)
    env = Environment(
        jnp.zeros((1, 2)),
        jnp.ones(1),
        jnp.array([[0.0, 1.0, 0.5]]),
        jnp.array([[-3.0, -3.0], [1.0, 3.0]]),
        jnp.zeros(2),
        jnp.zeros(2),
        jnp.zeros(2),
    )
    belief = prior([[0.0, 0.0], [1.0, 0.0]], std=0.2)
    inner = belief.inner_fields(jax.random.normal(jax.random.PRNGKey(2), (2, 2, 2)))
    x = jnp.array([0.45, 0.0, 0.8, 0.05])
    A, b, B, gradient = jax.jit(lambda x: constraints(x, belief, inner, env, cfg))(x)
    u = jnp.array([0.3, -0.4])
    f = jax.jit(lambda x: values(x, belief, inner, env, cfg))
    slopes = []
    for coeff in inner.coefficients:
        field = lambda p, c=coeff: inner.basis.evaluate(p, c)[0]
        dx = derivative(x, u, field, cfg.linear_drag)
        eps = 2e-3
        slopes.append((f(x + eps * dx) - f(x - eps * dx)) / (2 * eps))
    expected = np.max(slopes, axis=0) + cfg.cbf_alpha * B
    np.testing.assert_allclose(A @ u - b, expected, atol=2e-3, rtol=0.02)
    fd = np.stack(
        [(f(x + jnp.eye(4)[i] * 0.002) - f(x - jnp.eye(4)[i] * 0.002)) / 0.004 for i in range(4)],
        axis=1,
    )
    np.testing.assert_allclose(gradient, fd, atol=2e-3, rtol=0.02)
    truth = WorldField(belief.basis, inner.coefficients[0])
    result = jax.jit(
        lambda: rollout(
            x,
            jnp.array([0.5]),
            jnp.tile(u, (2, 1)),
            belief,
            inner,
            truth,
            jnp.zeros((2, 2)),
            env,
            cfg,
        )
    )()
    # Fine independent reintegration checks the applied-control path uses drag.
    actual = x
    for control in result.controls:
        actual = jax.lax.fori_loop(
            0,
            32,
            lambda _, z, control=control: step(
                z, control, lambda p: truth.at(p)[0], cfg.dt / 32, 0.9
            ),
            actual,
        )
    np.testing.assert_allclose(actual, result.x[-1], atol=2e-5)


@pytest.mark.parametrize("drag", [-1.0, float("nan"), float("inf")])
def test_invalid_drag_rejected(drag):
    with pytest.raises(ValueError, match="drag"):
        replace(Config(), linear_drag=drag)
