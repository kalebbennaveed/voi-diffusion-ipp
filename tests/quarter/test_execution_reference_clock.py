"""An estimator update must not reset or silently retime the planned reference."""
from dataclasses import replace
import numpy as np
import jax.numpy as jnp
from voi_diffusion_ipp.quarter.continuous import ControlSettings
from voi_diffusion_ipp.quarter.environment import environment,TABLE_SHAPE
from voi_diffusion_ipp.quarter.execution import executor
from voi_diffusion_ipp.quadrotor.config import VehicleConfig
from voi_diffusion_ipp.quadrotor.types import State

def test_chunked_physics_retains_the_original_reference_and_controls():
    cfg=ControlSettings(horizon=1.,commit=.4,dt=.1,environment_scale=.35,inspection_scale=.55)
    env=environment('two',.35,.55);vehicle=VehicleConfig()
    packed=jnp.array(State(np.array([-9.8,2.8,3.2]),np.zeros(3),np.eye(3),np.zeros(3),np.full(4,np.sqrt(vehicle.mass*vehicle.gravity/(4*vehicle.thrust_coefficient)))).pack(),jnp.float32)
    theta=jnp.array(np.random.default_rng(13).normal(0,.1,20),jnp.float32);mean=jnp.zeros(TABLE_SHAPE);fields=jnp.zeros((2,*TABLE_SHAPE));origin=jnp.r_[packed[:2],packed[3:5]]
    load=lambda p:jnp.zeros((len(p),2))
    full=executor(env,cfg,load,False);chunk=executor(env,replace(cfg,commit=.2),load,False)
    _,expected=full(packed,theta,mean,fields)
    middle,first=chunk(packed,theta,mean,fields,origin,jnp.array(0))
    _,last=chunk(middle,theta,mean,fields,origin,jnp.array(2))
    for a,b,c in zip(expected,first,last):np.testing.assert_allclose(np.concatenate((b,c)),a,atol=3e-5,rtol=3e-5)
    shifted=middle.at[0].add(.5)
    _,changed=chunk(shifted,theta,mean,fields,origin,jnp.array(2))
    np.testing.assert_array_equal(changed[1],last[1])
