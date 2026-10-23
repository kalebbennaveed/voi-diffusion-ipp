"""Metre-scale village for planar MBD and nonlinear quadrotor execution.

The field is a synthetic, fixed spatial *acceleration load*, not CFD or wind
velocity. Its physical force is mass times this load. Unknown vehicle parameters
are not part of the belief. All dimensions below are in metres.
"""
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..config import Config
from ..wind.poam import POAM
from ..wind.cache import TableBasis, DecoupledBasis, PriorCache


class UrbanEnvironment(NamedTuple):
    rectangles: jax.Array  # xmin,ymin,xmax,ymax,height
    bounds: jax.Array
    cells: jax.Array
    targets: jax.Array
    process_noise: jax.Array
    regions: jax.Array

    def violations(self, x, margin=0.):
        p = x[:2]
        mid = (self.rectangles[:, :2] + self.rectangles[:, 2:4])/2
        half = (self.rectangles[:, 2:4] - self.rectangles[:, :2])/2
        d = jnp.abs(p-mid)-half
        # Exact signed distance to a rectangle, smoothed norm at zero only.
        outside = jnp.sqrt(jnp.maximum(jnp.sum(jnp.maximum(d, 0)**2, axis=1), 1e-12))
        sdf = outside + jnp.minimum(jnp.max(d, axis=1), 0.)
        return jnp.r_[margin-sdf, self.bounds[0]+margin-p, p-self.bounds[1]+margin]

    def clearance(self, x):
        return -jnp.max(self.violations(x))


def environment(case="forward"):
    rect = np.array([[14,12,28,22,8.5], [16,3,27,6,5.5],
                     [0.6,12,3.2,22,7], [5,24,7.4,26.5,4.5],
                     [37.5,3.2,39.4,12,6], [38,24,39.5,27.5,5]], np.float32)
    regions = np.array([[3.5,3.5,8.5,8.5],[32.5,15.5,37.5,20.5]], np.float32)
    if case == "commitment":
        regions[1] = [32,11,37,23]
    cells = []
    for a,b,c,d in regions:
        xx,yy=np.meshgrid(np.arange(a+.5,c,.9),np.arange(b+.5,d,.9))
        cells.extend(np.c_[xx.ravel(),yy.ravel()])
    cells=jnp.asarray(np.array(cells),jnp.float32)
    return UrbanEnvironment(jnp.asarray(rect),jnp.array([[0.,0.],[40.,28.]]),cells,
                            jnp.full(len(cells),.75),jnp.full(len(cells),.025),jnp.asarray(regions))


def smooth_window(z, lo, hi, edge):
    a=jnp.clip((z-lo)/edge,0,1); b=jnp.clip((hi-z)/edge,0,1)
    smooth=lambda t:t*t*t*(10-15*t+6*t*t)
    return smooth(a)*smooth(b)


def wind_load(p, case="forward", strength=1.):
    p=jnp.atleast_2d(p); x,y=p[:,0],p[:,1]
    axial=smooth_window(x,9,33,3)
    A=axial*smooth_window(y,6,12,1)
    C=axial*smooth_window(y,0,3,0.5)
    B=axial*smooth_window(y,22,28,1)
    zero=jnp.zeros_like(x)
    if case in ("forward","reverse"):
        sign=1 if case=="forward" else -1
        return strength*jnp.stack((sign*2.15*A, .18*A*jnp.sin((x-10)/3)),axis=-1)*jnp.array([1.,sign])
    if case=="crosswind":
        # Alternating lateral jets fill A; longitudinal loading remains when
        # the lateral component changes sign. C is pushed toward its building.
        return strength*jnp.stack((-1.4*A-1.25*C,
                                    2.6*A*jnp.cos((x-13)*.42)+2.5*C),axis=-1)
    if case=="commitment":
        # Remain within the common inspection command authority. A separate
        # archived 3.1 m/s² stress test exceeds it and was not recoverable.
        return strength*jnp.stack((2.15*(A+B+C), .35*(A-C)*jnp.sin(x/4)),axis=-1)
    raise ValueError(case)


def config():
    return Config(dt=.4,stage_steps=5,future_steps=95, worlds=2,inner_samples=2,
        backup_steps=8,backup_dt=.2,backup_gain=2.,cbf_alpha=1.5,umax=2.6,
        safety_margin=.45,diffusion_steps=4,proposals=8,candidate_chunk=1,
        beta_min=.002,beta_max=.08,temperature=.09,mbd_latent_bound=6.,
        sensor_sigma=2.,sensor_R=.3,observation_std=.18,wind_std=1.4)


def initial_belief(key):
    xx,yy=np.meshgrid(np.linspace(4,36,7),[1.5,8.8,17,25])
    centers=np.c_[xx.ravel(),yy.ravel()].astype(np.float32)
    return POAM(centers,key,noise_std=.18,lengths=(3.,7.),prior_std=1.4,
                spectral_count=8).planning_belief()


def build_cache(b,eps):
    # Padded domain: even a failed rollout has finite field queries so that
    # failure costs remain defined. Grid is 1 m; validation reports error.
    nx,ny=57,45; low=jnp.array([-8.,-8.]); spacing=jnp.ones(2)
    xx,yy=jnp.meshgrid(jnp.arange(nx),jnp.arange(ny))
    loc=low+jnp.stack((xx.ravel(),yy.ravel()),axis=-1)
    m=len(b.mean)
    c=b.sample_coefficients(eps).at[:,-m:].set(0)
    prior=jax.vmap(lambda cc:b.basis.evaluate(loc,cc))(c)
    gz=jax.vmap(lambda cc:b.basis.evaluate(b.basis.centers,cc))(c)
    kernel=b.basis.kernel_centers(loc).reshape(ny,nx,m)
    return PriorCache(DecoupledBasis(TableBasis(low,spacing),prior.reshape(len(eps),ny,nx,2),
                                   b.basis,kernel),gz,eps[:,-m:])


CASE_NAMES={"forward":"1a · Wind in A, forward", "reverse":"1b · Wind in A, reversed",
            "crosswind":"2 · Sheltered B, adverse A and C", "commitment":"3 · The cost of crossing"}
METHOD_NAMES={"proposed":"Proposed", "aware":"Disturbance-aware", "blind":"Disturbance-blind"}
