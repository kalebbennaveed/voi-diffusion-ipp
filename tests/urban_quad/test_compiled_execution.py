import jax.numpy as jnp
import numpy as np
import pytest

from voi_diffusion_ipp.quadrotor.config import VehicleConfig,ControllerConfig
from voi_diffusion_ipp.quadrotor.controller import GeometricController
from voi_diffusion_ipp.quadrotor.dynamics import Quadrotor
from voi_diffusion_ipp.quadrotor.types import State
from voi_diffusion_ipp.quadrotor.reference import QuinticReference,flat_reference
from voi_diffusion_ipp.quadrotor.execution import executor
from voi_diffusion_ipp.wind.cache import TableBasis,TableMean


@pytest.mark.parametrize('limit',[None,1.0])
def test_compiled_plant_matches_numpy_plant(limit):
    vehicle=VehicleConfig();control=GeometricController(vehicle,ControllerConfig(max_horizontal_accel=limit))
    force=np.array([.4,-.2,0.]);plant=Quadrotor(vehicle,np.zeros(3),vehicle.mass*force)
    state=State(np.array([0.,0.,3.]),np.zeros(3),np.eye(3),np.zeros(3),np.full(4,np.sqrt(vehicle.mass*vehicle.gravity/(4*vehicle.thrust_coefficient))))
    ref=QuinticReference(state.position,np.zeros(3),np.zeros(3),np.array([.3,.1,3.]),np.array([.4,0,0]),np.zeros(3),.8)
    table=TableMean(TableBasis(jnp.array([-5.,-5.]),jnp.ones(2)),jnp.broadcast_to(jnp.array(force[:2]),(12,12,2)))
    execute=executor(lambda p:jnp.broadcast_to(jnp.array(force[:2]),(len(p),2)),.8,1,max_horizontal_accel=limit)
    _,logs=execute(jnp.asarray(state.pack(),jnp.float32),jnp.asarray(ref.coefficients[None],jnp.float32),table)
    actual=np.asarray(logs[0])
    for k in range(80):
        p,v,a,j,s=ref.at(k*.01)
        drag=vehicle.linear_drag*np.linalg.norm(state.velocity)*state.velocity/vehicle.mass
        target=flat_reference(p,v,a,j,s,compensation=drag-force)
        state=plant.step(state,control(state,target),.01)
        np.testing.assert_allclose(actual[k,:18],state.pack()[:18],atol=1e-4,rtol=1e-4)


def test_air_velocity_executor_matches_numpy_with_imperfect_estimate():
    from voi_diffusion_ipp.quadrotor.aerodynamics import acceleration
    vehicle=VehicleConfig()
    control=GeometricController(vehicle,ControllerConfig(max_horizontal_accel=2.6,limit_total_horizontal_force=True))
    physical=np.array([-3.8,.6,0.]);estimate=np.array([-2.5,.2,0.])
    plant=Quadrotor(vehicle,physical,np.zeros(3))
    state=State(np.array([0.,0.,3.]),np.zeros(3),np.eye(3),np.zeros(3),np.full(4,np.sqrt(vehicle.mass*vehicle.gravity/(4*vehicle.thrust_coefficient))))
    ref=QuinticReference(state.position,np.zeros(3),np.zeros(3),np.array([1.5,.1,3.]),np.array([2.,0,0]),np.zeros(3),.8)
    table=TableMean(TableBasis(jnp.array([-5.,-5.]),jnp.ones(2)),jnp.broadcast_to(jnp.array(estimate[:2]),(12,12,2)))
    execute=executor(lambda p:jnp.broadcast_to(jnp.array(physical[:2]),(len(p),2)),.8,1,max_horizontal_accel=2.6,wind_velocity=True)
    _,logs=execute(jnp.asarray(state.pack(),jnp.float32),jnp.asarray(ref.coefficients[None],jnp.float32),table)
    for k,actual in enumerate(np.asarray(logs[0])):
        p,v,a,j,s=ref.at(k*.01)
        target=flat_reference(p,v,a,j,s,compensation=-acceleration(state.velocity,estimate))
        state=plant.step(state,control(state,target),.01)
        np.testing.assert_allclose(actual[:18],state.pack()[:18],atol=1e-4,rtol=1e-4)
