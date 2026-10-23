"""Physical controller projection agrees with independent constrained optimization."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,minimize
from voi_diffusion_ipp.safety.qp import project


def test_parallel_repeated_and_zero_constraints_preserve_box_projection():
    A=jnp.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.],[1.,0.],[0.,0.]])
    b=jnp.array([1.,1.,2.,2.,1.,0.])
    u,feasible=jax.jit(project)(jnp.array([3.,4.]),A,b)
    assert feasible
    np.testing.assert_allclose(u,[1.,2.],atol=2e-6)
    u,feasible=project(jnp.array([.2,-.3]),A,b)
    assert feasible
    np.testing.assert_array_equal(u,jnp.array([.2,-.3]))


def test_inconsistent_parallel_and_constant_constraints_are_infeasible():
    for A,b in [(jnp.array([[1.,0.],[-1.,0.]]),jnp.array([-1.,-1.])),
                (jnp.array([[0.,0.],[1.,0.]]),jnp.array([-1.,1.]))]:
        u,feasible=project(jnp.array([0.,0.]),A,b)
        assert not feasible
        assert np.isfinite(u).all()


def test_projection_matches_independent_scipy_qp():
    rng=np.random.default_rng(84);count=32
    matrices=rng.normal(size=(count,8,2));known=rng.uniform(-1.5,1.5,(count,2))
    bounds=np.einsum('nki,ni->nk',matrices,known)+rng.uniform(.05,2,(count,8))
    box=np.array([[1,0],[-1,0],[0,1],[0,-1]])
    A=np.concatenate((matrices,np.broadcast_to(box,(count,4,2))),axis=1)
    b=np.c_[bounds,np.full((count,4),2.2)]
    targets=rng.normal(size=(count,2))*6
    result,feasible=jax.jit(jax.vmap(project))(jnp.array(targets),jnp.array(A),jnp.array(b))
    assert feasible.all()
    for i in range(count):
        ref=minimize(lambda x:.5*np.sum((x-targets[i])**2),known[i],
            jac=lambda x:x-targets[i],constraints=LinearConstraint(matrices[i],-np.inf,bounds[i]),
            bounds=Bounds([-2.2,-2.2],[2.2,2.2]),method='SLSQP',options={'ftol':1e-10,'maxiter':200})
        assert ref.success,ref.message
        np.testing.assert_allclose(result[i],ref.x,atol=2e-4,rtol=2e-4)
        assert np.max(A[i]@np.asarray(result[i])-b[i])<3e-5
