"""Independent LP oracle and degenerate cases for common-slack elimination."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import linprog

from voi_diffusion_ipp.safety.qp import minimum_slack, solve


@pytest.mark.parametrize('rows', [1, 2, 3, 6, 9])
def test_minimax_slack_matches_independent_linear_program(rows):
    rng = np.random.default_rng(471+rows)
    matrices = rng.normal(size=(48, rows, 2)).astype(np.float32)
    bounds = rng.normal(size=(48, rows)).astype(np.float32)
    # Constant, duplicate, opposite, and nearly parallel residual planes.
    matrices[::7, 0] = 0.
    if rows>1:
        matrices[1::7, 1] = matrices[1::7, 0]
        matrices[2::7, 1] = -matrices[2::7, 0]
        matrices[3::7, 1] = matrices[3::7, 0]+1e-7
    got = np.asarray(jax.jit(jax.vmap(minimum_slack, in_axes=(0, 0, None)))(
        jnp.asarray(matrices), jnp.asarray(bounds), 2.2))
    expected = []
    for A, b in zip(matrices, bounds):
        result = linprog([0., 0., 1.], A_ub=np.c_[A, -np.ones(rows)], b_ub=b,
                         bounds=[(-2.2, 2.2), (-2.2, 2.2), (0., None)], method='highs')
        assert result.success
        expected.append(result.fun)
    np.testing.assert_allclose(got, expected, atol=8e-5, rtol=2e-5)


def test_fallback_retains_actuator_box_and_reports_minimal_slack():
    A = jnp.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.], [0., 0.], [1., 1.]])
    b = jnp.array([-3., -1., -1., -2., -.3, -.1])
    result = jax.jit(solve)(jnp.array([7., -7.]), A, b, 2.2, jnp.array([.5, -.5]))
    assert not bool(result.feasible)
    np.testing.assert_allclose(result.slack, 2., atol=2e-5)
    assert float(jnp.max(jnp.abs(result.control)))<=2.2+2e-5
    assert float(jnp.max(A@result.control-b))<=float(result.slack)+8e-5
