import numpy as np
import pytest
import jax.numpy as jnp
from voi_diffusion_ipp.quarter.continuous import ControlSettings,nominal_reference
from voi_diffusion_ipp.quarter.environment import environment,TABLE_SHAPE
from voi_diffusion_ipp.quarter.execution import executor
from voi_diffusion_ipp.quadrotor.config import VehicleConfig,ControllerConfig
from voi_diffusion_ipp.quadrotor.controller import allocation_matrix,geometric_wrench
from voi_diffusion_ipp.quadrotor.dynamics import Quadrotor
from voi_diffusion_ipp.quadrotor.reference import flat_reference
from voi_diffusion_ipp.quadrotor.types import State
from voi_diffusion_ipp.quadrotor.aerodynamics import acceleration


@pytest.mark.parametrize('planning_dt',[.1,.2,.25])
def test_sampled_outer_controller_and_nonlinear_plant_against_numpy(planning_dt):
    cfg=ControlSettings(dt=planning_dt,horizon=1.,commit=1.)
    vehicle=VehicleConfig();gains=ControllerConfig(max_horizontal_accel=cfg.umax,limit_total_horizontal_force=True)
    actual=np.array([-2.,.4,0.]);estimate=np.array([-.8,.1])
    initial=State(np.array([-28.,8.,3.2]),np.zeros(3),np.eye(3),np.zeros(3),np.full(4,np.sqrt(vehicle.mass*vehicle.gravity/(4*vehicle.thrust_coefficient))))
    state=State.unpack(initial.pack().copy());plant=Quadrotor(vehicle,actual,np.zeros(3))
    theta=jnp.array(np.random.default_rng(7).normal(0,.3,2*cfg.steps),dtype=jnp.float32)
    mean=jnp.broadcast_to(jnp.array(estimate),TABLE_SHAPE);construction=jnp.zeros((2,*TABLE_SHAPE))
    execute=executor(environment(),cfg,lambda p:jnp.broadcast_to(jnp.array(actual[:2]),(len(p),2)),filter_on=False)
    _,logs=execute(jnp.array(initial.pack(),jnp.float32),theta,mean,construction)
    states,_,_,_,applied,_=map(np.asarray,logs)
    reference,aa=map(np.asarray,nominal_reference(theta,jnp.r_[initial.position[:2],initial.velocity[:2]],cfg))
    allocation=allocation_matrix(vehicle)
    for interval in range(cfg.commit_steps):
        ref=reference[interval];a=aa[interval]
        ticks=round(cfg.dt/.01)
        for i in range(ticks):
            t=i*.01;p=np.r_[ref[:2]+t*ref[2:]+.5*t*t*a,3.2];v=np.r_[ref[2:]+t*a,0.]
            if i%round(cfg.execution_dt/.01)==0:
                u=np.clip(a+cfg.tracking_kp*(p[:2]-state.position[:2])+cfg.tracking_kv*(v[:2]-state.velocity[:2])-acceleration(state.velocity[:2],estimate),-cfg.umax,cfg.umax)
            flat=flat_reference(p,v,np.r_[u,0.])
            wrench=geometric_wrench(state.position,state.velocity,state.rotation,state.angular_velocity,flat,vehicle,gains,horizontal_acceleration=u)
            command=np.sqrt(np.clip(np.linalg.solve(allocation,wrench),0,vehicle.max_motor_speed**2))
            state=plant.step(state,command,.01)
            k=interval*ticks+i
            np.testing.assert_allclose(applied[k],u,atol=4e-5)
            np.testing.assert_allclose(states[k,:18],state.pack()[:18],atol=8e-5,rtol=1e-4)
    assert np.max(np.abs(states[:,6:15]-np.eye(3).ravel()))>.01
    assert np.ptp(states[:,18:])>1
