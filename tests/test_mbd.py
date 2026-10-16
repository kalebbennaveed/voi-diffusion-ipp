from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.planning.mbd import score, schedule, optimize, weights


def test_score_matches_analytic_gaussian_convolution():
    # p0=N(target, lambda I); convolved p_i=N(sqrt(ab)*target, ab*lambda+1-ab).
    ab=.65; lam=.4; y=jnp.array([.8,-.4]); target=jnp.array([-.2,.7])
    candidates = y/jnp.sqrt(ab)+jnp.sqrt((1-ab)/ab)*jax.random.normal(jax.random.PRNGKey(2),(80000,2))
    cost=.5*jnp.sum((candidates-target)**2,axis=1)
    s,w=score(y,candidates,cost,ab,lam)
    expected=-(y-jnp.sqrt(ab)*target)/(ab*lam+1-ab)
    mean=w@candidates
    se=jnp.sqrt(jnp.sum((w[:,None]*(candidates-mean))**2,axis=0))*jnp.sqrt(ab)/(1-ab)
    assert np.all(np.abs(s-expected) < 5*np.asarray(se))


def test_schedule_optimization_budget_and_invalid():
    a,ab=schedule(8,.005,.2)
    np.testing.assert_allclose(ab,np.cumprod(a),atol=1e-7)
    assert np.all(np.diff(ab)<0)
    cfg=replace(Config(),diffusion_steps=14,proposals=80,temperature=.015,candidate_chunk=8)
    f=lambda y:jnp.sum((y-jnp.array([.5,-.3]))**2)
    for method in ('mbd','cem','mppi'):
        r=jax.jit(lambda k:optimize(f,jnp.array([-1.,1.]),k,cfg,method))(jax.random.PRNGKey(7))
        assert r.cost<.025
        assert int(r.evaluations)==1122
        assert np.all(np.diff(r.trace[:,0])<=1e-7)
    np.testing.assert_array_equal(weights(jnp.array([jnp.inf,jnp.nan]),.1),[0,0])
    r=optimize(lambda y:jnp.inf,jnp.zeros(2),jax.random.PRNGKey(1),replace(cfg,diffusion_steps=2))
    assert np.isfinite(r.vector).all() and np.isinf(r.cost)
