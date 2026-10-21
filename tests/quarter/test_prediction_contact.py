"""A physically terminated forecast cannot receive later sensing credit."""
from dataclasses import replace
import numpy as np
import jax,jax.numpy as jnp
from voi_diffusion_ipp.quarter.continuous import ControlSettings,rollout,segment_contact
from voi_diffusion_ipp.quarter.environment import environment,TABLE_SHAPE
from voi_diffusion_ipp.experiments.urban_quad import UrbanEnvironment

def test_swept_footprint_uses_euclidean_corners_and_detects_crossings():
    empty=jnp.zeros(0);env=UrbanEnvironment(jnp.array([[0.,0.,1.,1.,4.]]),jnp.array([[-5.,-5.],[5.,5.]]),jnp.zeros((0,2)),empty,empty,jnp.zeros((0,4)))
    assert bool(segment_contact(jnp.array([-1.,.5]),jnp.array([2.,.5]),env))
    assert bool(segment_contact(jnp.array([-.2,-.1]),jnp.array([-.2,-.1]),env))
    assert not bool(segment_contact(jnp.array([-.2,-.2]),jnp.array([-.2,-.2]),env))
    assert not bool(segment_contact(jnp.array([-1.,-.4]),jnp.array([2.,-.4]),env))

def test_monitoring_stops_after_contact_and_unsensed_decay_continues():
    cfg=ControlSettings(horizon=10.,dt=.1,commit=.2,environment_scale=.35,inspection_scale=.55,terminate_predictions=True)
    env=environment('two',.35,.55);q=jnp.full(84,.6);x=jnp.array([-9.8,-3.5,0.,0.]);theta=jnp.tile(jnp.array([3.,0.]),cfg.steps)
    mean=jnp.zeros(TABLE_SHAPE);fields=jnp.zeros((2,*TABLE_SHAPE))
    evaluate=jax.jit(lambda theta:rollout(theta,x,q,mean,fields,mean,env,cfg,False));stopped=evaluate(theta)
    assert bool(stopped.terminated)
    np.testing.assert_allclose(stopped.q[-1],.6/(1+.014*.6*10),atol=2e-6)
    np.testing.assert_array_equal(stopped.x[-1],stopped.x[-2])
    old=jax.jit(lambda theta:rollout(theta,x,q,mean,fields,mean,env,replace(cfg,terminate_predictions=False),False))(theta)
    assert float(old.q[-1,42:].mean())>float(stopped.q[-1,42:].mean())+.01
