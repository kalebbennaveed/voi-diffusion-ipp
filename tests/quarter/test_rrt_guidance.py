"""RRT geometry, dynamics conversion, and the paper's demonstration weights."""
import numpy as np
import jax
import jax.numpy as jnp

from voi_diffusion_ipp.quarter.rrt import connect, parameterize, demonstrations
from voi_diffusion_ipp.quarter.environment import environment
from voi_diffusion_ipp.quarter.continuous import ControlSettings, nominal_reference
from voi_diffusion_ipp.quarter.mbd_core import guided_log_weights, search


def test_rrt_connect_goes_around_obstacle_and_stays_in_bounds():
    rectangles=np.array([[-1.,-2.,1.,2.,10.]])
    bounds=np.array([[-4.,-4.],[4.,4.]])
    path=connect(np.array([-3.,0.]),np.array([3.,0.]),rectangles,bounds,
                 np.random.default_rng(51),margin=.4,step=.6)
    assert path is not None
    np.testing.assert_array_equal(path[[0,-1]],[[-3,0],[3,0]])
    assert len(path)>2
    # Independent dense-point collision test of each returned line segment.
    for a,b in zip(path[:-1],path[1:]):
        p=a+np.linspace(0,1,1001)[:,None]*(b-a)
        inside=(p[:,0]>=-1.4)&(p[:,0]<=1.4)&(p[:,1]>=-2.4)&(p[:,1]<=2.4)
        assert not inside.any()
        assert np.all(p>=bounds[0]+.4) and np.all(p<=bounds[1]-.4)


def test_rrt_time_parameterization_is_full_double_integrator_control():
    cfg=ControlSettings(horizon=20,commit=2,environment_scale=.35,inspection_scale=.55,initialization='rrt')
    state=np.array([-5.,0.,.2,0.]);path=np.array([[-5,0],[-5,3],[3,3.]])
    theta=parameterize(path,state,cfg,2.8)
    reference,acceleration=map(np.asarray,nominal_reference(jnp.array(theta),jnp.array(state),cfg))
    assert theta.shape==(400,)
    assert np.max(np.abs(acceleration))<=cfg.umax
    np.testing.assert_allclose(reference[-1,:2],path[-1],atol=.04)
    np.testing.assert_allclose(reference[-1,2:],0,atol=.008)
    altered=theta.copy();altered[121]+=.05
    changed,_=nominal_reference(jnp.array(altered),jnp.array(state),cfg)
    assert not np.array_equal(changed[-1],reference[-1])


def test_demonstrations_are_reproducible_and_use_only_state_and_geometry():
    cfg=ControlSettings(horizon=30,environment_scale=.35,inspection_scale=.55,initialization='rrt',rrt_paths=2)
    env=environment('two',.35,.55);state=np.array([-9.8,2.8,0,0])
    a,ra=demonstrations(state,env,cfg,8);b,rb=demonstrations(state,env,cfg,8)
    np.testing.assert_array_equal(a,b);assert ra==rb
    assert a.shape==(8,600) and ra['found']==8


def test_long_horizon_initializations_can_cover_two_inspection_visits():
    cfg=ControlSettings(horizon=60,environment_scale=.35,inspection_scale=.55,
                        initialization='rrt',rrt_paths=1,rrt_visits=2)
    env=environment('two',.35,.55);state=np.array([-9.8,2.8,0,0])
    parameters,record=demonstrations(state,env,cfg,8)
    assert parameters.shape==(8,1200)
    goals=np.asarray(env.regions[:,:2]+env.regions[:,2:])/2
    for theta,item in zip(parameters,record['records']):
        if item['visit_order']==[1,0] and item['cruise']==2.8:
            p=np.asarray(nominal_reference(jnp.array(theta),jnp.array(state),cfg)[0])[:,:2]
            east=np.linalg.norm(p-goals[1],axis=1).argmin()
            assert np.linalg.norm(p[east]-goals[1])<.05
            assert np.min(np.linalg.norm(p[east+1:]-goals[0],axis=1))<.05
            break
    else:raise AssertionError('No two-visit initialization generated')


def test_equation_13_and_importance_correction_against_numpy():
    model=np.array([-2.,-6.,-3.]); demo=np.array([-4.,-2.,-3.]);ratio=np.array([.2,-.3,0.])
    expected=np.log(np.maximum(np.exp(model),np.exp(demo)))+ratio
    np.testing.assert_allclose(guided_log_weights(model,demo,ratio),expected,rtol=1e-6)
    cfg=ControlSettings(horizon=.1,commit=.1,iterations=3,proposals=16,batch_size=8,initialization='rrt')
    values=lambda x:jnp.array([jnp.sum(x*x)])
    guide=lambda x,reference,cost:-cost/cfg.temperature-.1*jnp.sum((x-reference)**2)
    result=search(values,jnp.zeros((1,2)),jnp.zeros(2),jax.random.PRNGKey(1),cfg,1.,0.,True,guide)
    for i in range(cfg.iterations):
        logw=np.maximum(-np.asarray(result.sample_costs[i])/cfg.temperature,np.asarray(result.guidance_log_weights[i]))
        expected=np.exp(logw-logw.max());expected/=expected.sum()
        np.testing.assert_allclose(result.weights[i],expected,atol=1e-6)
    # Guidance affects search weights, never the recorded objective values.
    expected=np.sum(np.asarray(result.samples)**2,axis=-1)
    np.testing.assert_allclose(result.sample_costs,expected,rtol=1e-5)
