"""Caching must leave POAM moments and coherent sampled fields unchanged."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import jax
import jax.numpy as jnp
import numpy as np
import rmse_monitoring as model
from cached_poam_fields import CachedFields
from voi_diffusion_ipp.wind.poam import POAM
from waypoint_poam import condition,train


def test_cache_matches_fresh_build_after_actual_updates_and_kernel_training():
    xx,yy=np.meshgrid(np.linspace(0,2,7),np.linspace(0,2,7));points=jnp.asarray(np.c_[xx.ravel(),yy.ravel()])
    cfg=model.settings();shape=(7,7,2)
    gp=POAM(points[::3],jax.random.PRNGKey(13),lengths=(.2,.5,1.,2.),prior_std=1.8,
            noise_std=.1,attention_bounds=jnp.array([[0.,0.],[2.,2.]]),prior_support=points)
    b=jax.tree.map(lambda a:a.astype(jnp.float64) if hasattr(a,'dtype') and jnp.issubdtype(a.dtype,jnp.floating) else a,gp.planning_belief());cache=CachedFields(points,shape,cfg,points);fresh=model.field_builder(points,shape,cfg)
    mask=jnp.zeros(len(b.mean),bool);x=jnp.array([[.31,.8],[1.4,1.32]]);y=jnp.array([[2.,-1.],[-.4,.6]])
    m=jax.tree.map(jnp.zeros_like,b.basis.parameters);v=m
    for stage in range(3):
        if stage==1:b,mask=condition(b,x,y,mask)
        if stage==2:b,*_=train(b,x,y,jnp.array(2.),m,v,jnp.array(0),.01)
        args=(b,jnp.array(stage*2),jax.random.PRNGKey(21),jax.random.PRNGKey(22))
        actual=cache.build(*args);expected=fresh(*args)
        refreshed=cache.safety(b,actual,args[-1])
        reused=cache.planning(b,refreshed,args[-2])
        for aa,bb in zip(reused,expected):np.testing.assert_allclose(aa,bb,atol=2e-7,rtol=1e-6)
        for aa,bb in zip(actual,expected):np.testing.assert_allclose(aa,bb,atol=2e-7,rtol=1e-6)
        mu,cov=cache.map(b);direct_mu,direct_cov=b.predict(points)
        np.testing.assert_allclose(mu,direct_mu,atol=2e-7,rtol=1e-6)
        np.testing.assert_allclose(cov,direct_cov,atol=2e-7,rtol=1e-6)
    assert cache.rebuilds==2
