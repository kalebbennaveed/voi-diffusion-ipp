"""Nonlinear open-space speed calibration, separate from mission results."""
from pathlib import Path
import json
import numpy as np
import jax.numpy as jnp
from voi_diffusion_ipp.quadrotor.config import VehicleConfig
from voi_diffusion_ipp.quadrotor.types import State
from voi_diffusion_ipp.quadrotor.reference import QuinticReference
from voi_diffusion_ipp.quadrotor.execution import executor
from voi_diffusion_ipp.quarter.environment import TABLE_BASIS,TABLE_SHAPE
from voi_diffusion_ipp.wind.cache import TableMean

v=VehicleConfig();s=State(np.array([-35.,0.,3.2]),np.zeros(3),np.eye(3),np.zeros(3),np.full(4,np.sqrt(v.mass*v.gravity/(4*v.thrust_coefficient))))
polys=[]
# Smooth acceleration to 3 m/s, then cruise. The controller has correct wind
# for this isolated authority check, so estimation error cannot explain slowing.
for i in range(250):
    def ref(t):
        a=.6 if t<5 else 0.;vel=min(.6*t,3.);x=.3*t*t if t<5 else 7.5+3*(t-5)
        return np.array([x-35,0,3.2]),np.array([vel,0,0]),np.array([a,0,0])
    p0,v0,a0=ref(i*.1);p1,v1,a1=ref((i+1)*.1);polys.append(QuinticReference(p0,v0,a0,p1,v1,a1,.1).coefficients)
rows=[]
for wind in [0.,-2.5,-3.1,-3.5,3.1]:
    field=lambda p:jnp.broadcast_to(jnp.array([wind,0.]),(jnp.atleast_2d(p).shape[0],2))
    execute=executor(field,.1,250,dt=.01,max_horizontal_accel=2.2,wind_velocity=True)
    mean=jnp.zeros(TABLE_SHAPE).at[:,:,0].set(wind)
    _,log=execute(jnp.array(s.pack(),jnp.float32),jnp.array(np.array(polys),jnp.float32),TableMean(TABLE_BASIS,mean));states=np.asarray(log[0]);assert np.isfinite(states).all(), 'Invalid calibration trajectory';speed=float(states[-500:,3].mean())
    rows.append(dict(wind_mps=wind,requested_cruise_mps=3.,measured_final_5s_mean_ground_speed_mps=speed,minimum_height=float(states[:,2].min()),maximum_height=float(states[:,2].max())))
    print(rows[-1],flush=True)
Path('output/european_quarter/headwind_calibration.json').write_text(json.dumps(dict(description='Open-space nonlinear quadrotor calibration with known uniform wind. This is not a monitoring benchmark or learning result.',results=rows),indent=2))
