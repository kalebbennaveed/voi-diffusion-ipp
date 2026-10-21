"""Independent numerical checks for the fused two-dimensional Newton solve."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from voi_diffusion_ipp.quarter.filter import newton_direction, soft_projection


def test_fused_direction_matches_double_precision_linear_solve():
    rng = np.random.default_rng(83)
    matrices = rng.normal(size=(128, 6, 2)).astype(np.float32)
    matrices[:32, :, 1] = matrices[:32, :, 0] * 1.001
    matrices[32:64] *= 30
    masks = rng.random((128, 6)) > .3
    gradients = rng.normal(size=(128, 2)).astype(np.float32)
    actual = np.asarray(jax.jit(jax.vmap(newton_direction, in_axes=(0, 0, 0, None)))(
        jnp.array(matrices), jnp.array(masks), jnp.array(gradients), 50.))
    for A, active, g, result in zip(matrices, masks, gradients, actual):
        A = A.astype(float)
        H = np.eye(2) + 50 * A.T @ (A * active[:, None])
        np.testing.assert_allclose(result, np.linalg.solve(H, g), atol=4e-6, rtol=4e-5)


def legacy_projection(nominal, A, b, limit, weight):
    """Previous eight-iteration algorithm, retaining its generic matrix solve."""
    objective = lambda v: .5*jnp.sum((v-nominal)**2) + .5*weight*jnp.sum(jax.nn.relu(A@v-b)**2)
    def improve(_, u):
        residual = A@u-b
        grad = u-nominal + weight*(A.T@jax.nn.relu(residual))
        H = jnp.eye(2) + weight*(A.T@(A*(residual>0)[:, None]))
        directions = jnp.stack((jnp.linalg.solve(H, grad), grad/(1+weight*jnp.sum(A*A))))
        scales = 2.**(-jnp.arange(7))
        trials = jnp.clip(u[None,None]-directions[:,None,:]*scales[None,:,None], -limit, limit).reshape(-1,2)
        trials = jnp.r_[u[None], trials]
        return trials[jnp.argmin(jax.vmap(objective)(trials))]
    return jax.lax.fori_loop(0, 8, improve, jnp.clip(nominal, -limit, limit))


def test_fused_projection_preserves_fixed_budget_solver_accuracy():
    # Eight projected Newton iterations are not a convergence guarantee on a
    # box face. Compare with the previous solver AND an independent optimum;
    # the fused solve must not worsen that existing approximation error.
    rng = np.random.default_rng(91)
    solve = jax.jit(soft_projection)
    previous = jax.jit(legacy_projection)
    for _ in range(40):
        nominal = rng.normal(size=2) * 3
        A = rng.normal(size=(6, 2))
        b = rng.normal(size=6)
        objective = lambda u: .5*np.sum((u-nominal)**2) + 25*np.sum(np.maximum(A@u-b, 0)**2)
        gradient = lambda u: u-nominal + 50*A.T@np.maximum(A@u-b, 0)
        reference = minimize(objective, np.clip(nominal, -2.2, 2.2), jac=gradient,
                             bounds=[(-2.2, 2.2)]*2, method='L-BFGS-B',
                             options={'ftol': 1e-13, 'gtol': 1e-10, 'maxiter': 1000})
        control, _ = solve(jnp.array(nominal), jnp.array(A), jnp.array(b), 2.2, 50.)
        old = np.asarray(previous(jnp.array(nominal), jnp.array(A), jnp.array(b), 2.2, 50.))
        actual = np.asarray(control)
        np.testing.assert_allclose(actual, old, atol=2e-5, rtol=2e-5)
        assert np.max(np.abs(actual)) <= 2.2 + 1e-6
        assert objective(actual) - reference.fun <= objective(old) - reference.fun + 1e-4
        if np.max(np.abs(reference.x)) < 2.19:
            assert objective(actual) <= reference.fun + 2e-3
