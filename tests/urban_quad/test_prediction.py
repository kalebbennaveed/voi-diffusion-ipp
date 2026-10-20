from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.experiments.urban_quad import environment,config,initial_belief,build_cache
from voi_diffusion_ipp.planning.urban import rollout,route_library,reference_predictor
from voi_diffusion_ipp.wind import WorldField


def setup():
    cfg=replace(config(),future_steps=5,backup_steps=4)
    env=environment();b=initial_belief(jax.random.PRNGKey(0))
    eps=jax.random.normal(jax.random.PRNGKey(1),(4,*b.sample_shape));cache=jax.jit(build_cache)(b,eps)
    x=jnp.array([6.,6.,0.,0.]);q=jnp.full(len(env.cells),.4)
    lib=route_library(np.array([6.,6.]),env);theta=jnp.array([1.,1.,-1.,-1.,-1.,1.,0.])
    return cfg,env,b,cache,x,q,lib,theta


def test_first_commitment_does_not_know_the_sampled_physical_field():
    cfg,env,b,cache,x,q,lib,theta=setup()
    mean,_=cache.tabulate(b)
    def simulate(field):
        return rollout(theta,x,q,b,cache,WorldField(mean.basis,field),jnp.zeros((10,2)),lib,env,cfg,True)
    simulate=jax.jit(simulate)
    calm=simulate(jnp.zeros_like(mean.field))
    windy=simulate(jnp.ones_like(mean.field)*.8)
    np.testing.assert_allclose(calm.controls[:5],windy.controls[:5],atol=1e-5)
    assert np.linalg.norm(np.asarray(calm.x[5]-windy.x[5]))>.01
    assert np.linalg.norm(np.asarray(calm.controls[5:]-windy.controls[5:]))>.01
    assert np.isfinite(np.asarray(windy.q)).all()


def test_plan_reference_position_velocity_and_acceleration_are_consistent():
    cfg,env,b,cache,x,q,lib,theta=setup()
    result=reference_predictor(env,cfg,'aware')(theta,x,q,b,cache,lib)
    xs=np.asarray(result.x);aa=np.asarray(result.acceleration)
    np.testing.assert_allclose(xs[1:,:2],xs[:-1,:2]+cfg.dt*xs[:-1,2:]+.5*cfg.dt**2*aa,atol=1e-5)
    np.testing.assert_allclose(xs[1:,2:],xs[:-1,2:]+cfg.dt*aa,atol=1e-5)


def test_cache_field_error_and_full_covariance_are_small():
    cfg,env,b,cache,x,q,lib,theta=setup()
    b=b.condition(jnp.array([[8.,8.],[10.,9.]]),jnp.array([[1.2,-.4],[1.5,-.2]]))
    locations=jax.random.uniform(jax.random.PRNGKey(2),(150,2))*jnp.array([36.,24.])+2
    mean,inner=cache.tabulate(b);exact=cache.condition(b)
    # This also checks the updated inducing correction, not only the prior.
    err=np.asarray(mean.mean_at(locations)-b.mean_at(locations))
    assert np.sqrt(np.mean(err**2))<.03
    approximate=inner.at(locations);continuous=exact.at(locations)
    assert np.sqrt(np.mean(np.asarray(approximate-continuous)**2))<.02
    _,cov=b.joint(locations[:8]);assert np.linalg.eigvalsh(np.asarray(cov)).min()>-1e-5
