import numpy as np
import jax.numpy as jnp

from voi_diffusion_ipp.quadrotor.config import VehicleConfig,ControllerConfig
from voi_diffusion_ipp.quadrotor.controller import GeometricController
from voi_diffusion_ipp.quadrotor.dynamics import Quadrotor
from voi_diffusion_ipp.quadrotor.types import State
from voi_diffusion_ipp.quadrotor.reference import QuinticReference,flat_reference
from voi_diffusion_ipp.experiments.urban_quad import environment,wind_load
from voi_diffusion_ipp.planning.urban import route_library,visible


def test_quintic_end_conditions_and_derivatives():
    p0=np.array([1.,2.,3.]);p1=np.array([3.,4.,3.]);v=np.array([.5,0.,0.]);a=np.zeros(3)
    ref=QuinticReference(p0,v,a,p1,v,a,3.)
    for t,p in [(0,p0),(3,p1)]:
        result=ref.at(t)
        np.testing.assert_allclose(result[0],p,atol=1e-10)
        np.testing.assert_allclose(result[1],v,atol=1e-10)
        np.testing.assert_allclose(result[2],a,atol=1e-10)
    for i in range(4):
        fd=(ref.at(1.5+1e-4)[i]-ref.at(1.5-1e-4)[i])/2e-4
        np.testing.assert_allclose(fd,ref.at(1.5)[i+1],atol=1e-6)


def test_hover_and_nonlinear_tracking():
    v=VehicleConfig();c=GeometricController(v,ControllerConfig())
    model=Quadrotor(v,np.zeros(3),np.array([.4,-.2,0.])*v.mass)
    rpm=np.sqrt(v.mass*v.gravity/(4*v.thrust_coefficient))
    state=State(np.array([0.,0.,3.]),np.zeros(3),np.eye(3),np.zeros(3),np.full(4,rpm))
    segment=QuinticReference(state.position,np.zeros(3),np.zeros(3),np.array([2.,1.,3.]),np.zeros(3),np.zeros(3),4.)
    errors=[]
    for k in range(800):
        p,vel,a,j,s=segment.at(k*.005)
        drag=v.linear_drag*np.linalg.norm(state.velocity)*state.velocity/v.mass
        ref=flat_reference(p,vel,a,j,s,compensation=drag-np.array([.4,-.2,0.]))
        state=model.step(state,c(state,ref),.005)
        errors.append(np.linalg.norm(state.position-p))
    assert max(errors)<.12
    np.testing.assert_allclose(state.rotation.T@state.rotation,np.eye(3),atol=1e-10)
    assert np.all(state.motor_speed>=0)


def test_layout_and_wind():
    env=environment()
    assert tuple(np.asarray(env.bounds[1]))==(40,28)
    for p in [[20,9],[20,25],[20,1.5]]:assert float(env.clearance(jnp.array(p)))>1.4
    x=jnp.linspace(0,40,101)
    B=jnp.c_[x,jnp.full(101,25.)]
    np.testing.assert_array_equal(wind_load(B,"crosswind"),np.zeros((101,2)))
    points=jnp.array([[14.,9.],[20.,9.],[26.,9.]])
    np.testing.assert_allclose(wind_load(points,"forward"),-wind_load(points,"reverse"))
    assert np.min(np.linalg.norm(wind_load(points,"crosswind"),axis=1))>1.3


def test_routes_have_geometric_clearance():
    env=environment()
    for start in [np.array([6.,6.]),np.array([20.,9.]),np.array([35.,18.])]:
        paths,_,sizes=route_library(start,env)
        for path,size in zip(np.asarray(paths),np.asarray(sizes)):
            last=start
            for point in path[:size]:
                assert visible(last,point,np.asarray(env.rectangles),margin=.3),(last,point)
                last=point


def test_real_state_inside_route_inflation_can_reconnect():
    env=environment()
    for start in [np.array([15.9054,6.6142]),np.array([15.7824,6.5930])]:
        routes,_,sizes=route_library(start,env)
        for points,size in zip(np.asarray(routes),np.asarray(sizes)):
            assert visible(start,points[0],np.asarray(env.rectangles),margin=.20)
