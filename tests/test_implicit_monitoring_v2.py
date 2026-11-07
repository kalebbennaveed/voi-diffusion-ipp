"""Regression checks for conservative bounds and faithful held-out evaluation."""
import inspect
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import jax
import jax.numpy as jnp
import numpy as np
import implicit_monitoring as previous
import implicit_monitoring_v2 as revised


def test_tighter_bound_still_encloses_full_disk():
    rng=np.random.default_rng(508)
    angles=np.linspace(0,2*np.pi,3001)
    radii=np.array([0.,.25,.5,.75,1.])
    unit=(radii[:,None,None]*np.c_[np.cos(angles),np.sin(angles)][None]).reshape(-1,2)
    for _ in range(25):
        v=rng.normal(size=2)*2;mu=rng.normal(size=2)*4;radius=rng.uniform(0,8)
        gradients=jnp.asarray(rng.normal(size=(9,2)))
        winds=mu+radius*unit
        actual=np.asarray(gradients)@np.asarray(revised.m.physical.acceleration(v,winds)).T
        old=np.asarray(previous.force_support(gradients,v,mu,radius))
        new=np.asarray(revised.force_support(gradients,v,mu,radius))
        assert np.all(actual<=new[:,None]+1e-8)
        assert np.all(new<=old+1e-8)


def test_safety_and_performance_are_not_given_hidden_wind():
    for fn in [revised.optimizer,revised.controller,revised.field_builder,revised.execution_forecast]:
        assert not any('hidden' in k or 'truth' in k for k in inspect.signature(fn).parameters)
    assert revised.settings()==previous.settings()
    for fn in ['initial_belief','planning_fields','information_support']:
        assert inspect.getsource(getattr(revised,fn))==inspect.getsource(getattr(previous,fn))


def test_new_fields_are_independent_and_do_not_reduce_safety_spread():
    from voi_diffusion_ipp.wind.poam import POAM
    pts=jnp.array([[0.,0.],[.2,0.],[0.,.2],[.2,.2]])
    b=revised._double(POAM(pts,jax.random.PRNGKey(1),prior_std=1.8,spectral_count=8).planning_belief())
    build=revised.field_builder(pts,(2,2,2),revised.settings())
    cold=build(b,jnp.array(0),jax.random.PRNGKey(15),jax.random.PRNGKey(25))
    np.testing.assert_array_equal(cold.planning,0)
    np.testing.assert_array_equal(cold.validation,0)
    assert float(cold.std.max())>1.7
    assert float(jnp.abs(cold.safety).max())>0
    b,_=revised.condition_supported(b,pts[:1],jnp.array([[2.,-.5]]),jnp.zeros(4,bool))
    post=build(b,jnp.array(1),jax.random.PRNGKey(15),jax.random.PRNGKey(25))
    assert not np.allclose(post.planning,post.validation)
    np.testing.assert_allclose(post.planning.mean(0),post.mean,atol=1e-8)
    np.testing.assert_allclose(post.validation.mean(0),post.mean,atol=1e-8)


def test_two_pass_forecast_matches_original_execution_numerics():
    from dataclasses import replace
    from voi_diffusion_ipp.environment import Environment
    env=Environment(jnp.array([[.5,.5],[1.,1.],[1.5,.5]]),jnp.array([.7,.35,.7]),
                    jnp.array([[1.,1.5,.2]]),jnp.array([[0.,0.],[2.,2.]]),jnp.ones(2),jnp.zeros(2),jnp.zeros(2),jnp.array([.16,.02,.16]))
    basis,points,shape=revised.table_grid(env.bounds)
    mu=jnp.zeros(shape);std=jnp.full((*shape[:2],1),.1)
    w=jnp.broadcast_to(jnp.array([.4,-.2]),shape)
    tables=revised.Tables(mu,std,jnp.ones_like(std),jnp.stack((w,-w)),jnp.stack((w,-w)),jnp.stack((-w,w)))
    cfg=replace(revised.settings(),horizon=.5,commit=.5)
    theta=jnp.array([.4,.2,-.2,.3]);q=jnp.array([.7,.5,.4])
    for aware,x in [(True,jnp.array([.5,.5,.3,0.])),(False,jnp.array([.002,.5,-2.,0.]))]:
        fn=jax.jit(lambda x:revised.execution_forecast(theta,x,q,tables,w,basis,env,cfg,aware))
        reference=jax.jit(lambda x:revised.execution_forecast_reference(theta,x,q,tables,w,basis,env,cfg,aware))(x)
        actual=fn(x)
        for field in ['x','q','u','monitor','penalty','max_slack','min_clearance','terminated']:
            np.testing.assert_allclose(getattr(actual,field),getattr(reference,field),atol=1e-9,rtol=1e-9,err_msg=field)
        if not aware:assert bool(actual.terminated)
        import implicit_monitoring_v3 as fast
        old=jax.jit(lambda x:revised.rollout(theta,x,q,tables,w,basis,env,cfg,aware))(x)
        new=jax.jit(lambda x:fast.rollout(theta,x,q,tables,w,basis,env,cfg,aware))(x)
        for field in old._fields:
            np.testing.assert_allclose(getattr(new,field),getattr(old,field),atol=1e-9,rtol=1e-9,err_msg='relaxed '+field)


def test_execution_only_refresh_matches_full_posterior_sampler():
    import implicit_monitoring_v3 as immediate
    from voi_diffusion_ipp.wind.poam import POAM
    pts=jnp.array([[0.,0.],[.2,0.],[0.,.2],[.2,.2]])
    b=revised._double(POAM(pts,jax.random.PRNGKey(1),prior_std=1.8,spectral_count=8).planning_belief())
    cfg=revised.settings();shape=(2,2,2);key=jax.random.PRNGKey(99)
    full=revised.field_builder(pts,shape,cfg)
    old=full(b,jnp.array(0),jax.random.PRNGKey(4),key)
    b,_=revised.condition_supported(b,pts[:1],jnp.array([[2.,-.5]]),jnp.zeros(4,bool))
    expected=full(b,jnp.array(1),jax.random.PRNGKey(5),key)
    refreshed=immediate.safety_builder(pts,shape,cfg)(b,old,key)
    for name in ['mean','std','safety']:
        np.testing.assert_allclose(getattr(refreshed,name),getattr(expected,name),atol=1e-10)
    np.testing.assert_array_equal(refreshed.planning,old.planning)
    np.testing.assert_array_equal(refreshed.validation,old.validation)
