"""Independent LP oracle for the batched execution-filter fallback."""
import numpy as np
import jax,jax.numpy as jnp
from scipy.optimize import linprog
from voi_diffusion_ipp.safety.qp import minimum_slack,solve

def test_common_slack_against_highs_linear_program():
    rng=np.random.default_rng(941)
    matrices=rng.normal(size=(100,6,2));bounds=rng.normal(size=(100,6))*3
    expected=[]
    for A,b in zip(matrices,bounds):
        result=linprog([0.,0.,1.],A_ub=np.c_[A,-np.ones(6)],b_ub=b,bounds=[(-2.2,2.2),(-2.2,2.2),(0.,None)],method='highs')
        assert result.success;expected.append(result.fun)
    got=np.asarray(jax.jit(jax.vmap(lambda A,b:minimum_slack(A,b,2.2)))(jnp.array(matrices),jnp.array(bounds)))
    np.testing.assert_allclose(got,expected,atol=3e-5,rtol=3e-5)
    result=jax.jit(jax.vmap(lambda A,b:solve(jnp.array([1.,-1.]),A,b,2.2,jnp.zeros(2))))(jnp.array(matrices),jnp.array(bounds))
    assert np.max(np.abs(result.control))<=2.2+1e-5
    assert np.max(np.asarray(result.residual-result.slack))<2e-4
