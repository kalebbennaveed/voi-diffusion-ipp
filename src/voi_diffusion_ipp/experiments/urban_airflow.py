"""Air-velocity revision: the same physical layout, explicit finite authority."""
from dataclasses import replace
import jax.numpy as jnp
from .urban_quad import environment as previous_environment,build_cache,smooth_window,CASE_NAMES,METHOD_NAMES
from .urban_quad import config as previous_config
from ..wind.poam import POAM
import numpy as np

REVISION='airflow_narrow_cycles_v8'

def environment(case='forward'):
    env=previous_environment(case)
    rectangles=env.rectangles.at[0].set(jnp.array([14.,10.2,28.,23.,14.]))
    rectangles=rectangles.at[1].set(jnp.array([14.,2.6,28.,7.8,10.]))
    rectangles=rectangles.at[2,4].set(11.).at[4,4].set(10.)
    return env._replace(rectangles=rectangles)

def config():
    return replace(previous_config(),dt=1.,stage_steps=2,future_steps=118,
                   observation_std=.25,wind_std=1.8)

def wind_velocity(p,case='forward',strength=1.):
    p=jnp.atleast_2d(p);x,y=p[:,0],p[:,1]
    axial=smooth_window(x,9,33,3)
    A=axial*smooth_window(y,7.8,10.2,.35)
    B=axial*smooth_window(y,23,28,.7)
    C=axial*smooth_window(y,0,2.6,.35)
    if case in ('forward','reverse'):
        sign=1. if case=='forward' else -1.
        return strength*sign*jnp.stack((3.8*A,.18*A*jnp.sin((x-10)/3)),axis=-1)
    if case=='crosswind':
        return strength*jnp.stack((-3.65*A-3.55*C,.7*A*jnp.cos((x-13)*.42)+.7*C),axis=-1)
    if case=='commitment':
        return strength*jnp.stack((4.1*(A+B+C),.3*(A-C)*jnp.sin(x/4)),axis=-1)
    raise ValueError(case)

def initial_belief(key):
    xx,yy=np.meshgrid(np.linspace(4,36,7),[1.5,8.8,17,25])
    return POAM(np.c_[xx.ravel(),yy.ravel()].astype(np.float32),key,
                noise_std=.25,lengths=(3.,7.),prior_std=1.8,spectral_count=8).planning_belief()


def preflight_observations(case,seed=8101):
    """Declared noisy anemometer survey on B/C, never on passage A.

    These are sensor data supplied identically to every method, not exact
    field values passed to the planner. The raw observations are exported.
    """
    points=np.array([[x,y] for y in [1.3,25.] for x in [10.,15.,20.,25.,31.]],np.float32)
    rng=np.random.default_rng(seed+71000)
    measured=np.asarray(wind_velocity(jnp.asarray(points),case))+.25*rng.normal(size=points.shape)
    return points,measured.astype(np.float32)
