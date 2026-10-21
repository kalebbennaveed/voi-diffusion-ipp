"""MBD trajectory optimization for monitoring using dynamics rollout.

There are exactly 2*T/dt independent decision coordinates: both nominal
accelerations at every configured integration interval. No route library, target selection,
dwell schedule, waypoint tracker, spline knots, or truncated temporal basis.
"""
from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .. import clarity
from ..wind.cache import TableMean
from ..wind.gp import InnerFields,WorldField
from ..quadrotor.aerodynamics import acceleration as drag
from .environment import Settings,table_basis
from .filter import control
from . import sensing
from .mbd_core import search
from .diffusion import lower_tail


@dataclass(frozen=True)
class ControlSettings(Settings):
    horizon:float=60.
    iterations:int=32
    proposals:int=256
    latent_bound:float=12.
    cosine_end:float=.65
    batch_size:int=64
    chains:int=1
    environment_scale:float=1.
    inspection_scale:float=1.
    initialization:str='gaussian'
    rrt_paths:int=3
    rrt_visits:int=1
    common_construction:bool=False
    rrt_margin:float=.65
    guidance_sigma:float=1.
    planning_clearance:float=.275
    speed_penalty:float=2.
    jerk_limit:float=12.
    jerk_penalty:float=0.
    tracking_kp:float=3/.680
    tracking_kv:float=5/.680

    def __post_init__(self):
        super().__post_init__()
        if self.horizon<=0 or abs(self.horizon/self.dt-round(self.horizon/self.dt))>1e-6:raise ValueError('Horizon must be positive and divisible by dt')
        if self.commit<=0 or self.commit>self.horizon:raise ValueError('Commit must be positive and no longer than horizon')
        if min(self.iterations,self.proposals,self.worlds,self.validation_worlds,self.construction_worlds,self.batch_size,self.chains)<1:raise ValueError('All sampling and iteration counts must be positive')
        if self.temperature<=0 or not 0<self.cosine_end<1 or self.latent_bound<=0:raise ValueError('Invalid temperature, cosine interval, or parameter bound')
        if not .35<=self.environment_scale<=1.5:raise ValueError('Invalid environment scale')
        if not .4<=self.inspection_scale<=1.5:raise ValueError('Invalid inspection region scale')
        if self.initialization not in ('gaussian','zero','rrt'):raise ValueError('Unknown MBD initialization')
        if min(self.rrt_paths,self.rrt_visits)<1 or self.guidance_sigma<=0:raise ValueError('Invalid RRT guidance configuration')


def controls(theta,cfg):
    return cfg.umax*jnp.tanh(theta.reshape(cfg.steps,2))


def nominal_reference(theta,x0,cfg):
    acceleration=controls(theta,cfg)
    def step(x,a):
        xn=jnp.r_[x[:2]+cfg.dt*x[2:]+.5*cfg.dt**2*a,x[2:]+cfg.dt*a]
        return xn,xn
    _,xs=jax.lax.scan(step,x0,acceleration)
    return jnp.concatenate((x0[None],xs)),acceleration


def continuation(theta,cfg):
    # Shift the full sequence. Only the unplanned tail is extended by zero
    # acceleration; every coordinate can change at the next diffusion call.
    a=theta.reshape(cfg.steps,2)
    return jnp.concatenate((a[cfg.commit_steps:],jnp.zeros_like(a[:cfg.commit_steps]))).ravel()


def initial_sequences(x0,cfg):
    zero=jnp.zeros((cfg.steps,2))
    a=jnp.clip(-x0[2:]/2,-.9*cfg.umax,.9*cfg.umax)
    braking=zero.at[:min(cfg.steps,round(2/cfg.dt))].set(jnp.arctanh(a/cfg.umax))
    return jnp.stack((zero.ravel(),braking.ravel()))


def applied_control(x,reference,acceleration,mean,inner,env,cfg,filter_on):
    nominal=acceleration+cfg.tracking_kp*(reference[:2]-x[:2])+cfg.tracking_kv*(reference[2:]-x[2:])-drag(x[2:],mean.mean_at(x[:2])[0],xp=jnp)
    if filter_on:return control(x,nominal,mean,inner,env,cfg)
    return jnp.clip(nominal,-cfg.umax,cfg.umax),jnp.array(0.)


class Rollout(NamedTuple):
    x:jax.Array
    q:jax.Array
    u:jax.Array
    nominal:jax.Array
    monitor:jax.Array
    penalty:jax.Array
    max_slack:jax.Array
    min_clearance:jax.Array


def rollout(theta,x0,q0,mean_array,construction,physical,env,cfg,filter_on=True):
    basis=table_basis(cfg.environment_scale)
    mean=TableMean(basis,mean_array);inner=InnerFields(basis,construction);world=WorldField(basis,physical)
    reference,aa=nominal_reference(theta,x0,cfg)
    def step(carry,inputs):
        x,q,monitor,penalty,max_slack,gap,old_u=carry
        ref,a=inputs
        u,slack=applied_control(x,ref,a,mean,inner,env,cfg,filter_on)
        def f(z):return jnp.r_[z[2:],u+drag(z[2:],world.at(z[:2])[0],xp=jnp)]
        h=cfg.dt;k1=f(x);k2=f(x+h*k1/2);k3=f(x+h*k2/2);k4=f(x+h*k3)
        xn=x+h*(k1+2*k2+2*k3+k4)/6
        qn=sensing.step(q,x[:2],xn[:2],env,cfg,h)
        cost=h*.5*(clarity.deficit(q,env.targets)+clarity.deficit(qn,env.targets))
        clearance=env.clearance(xn)
        soft=cfg.violation_weight*jax.nn.relu(cfg.planning_clearance-clearance)**2
        soft+=cfg.speed_penalty*jax.nn.relu(jnp.linalg.norm(xn[2:])-cfg.speed_max)**2
        soft+=cfg.jerk_penalty*jnp.sum(jax.nn.relu(jnp.abs((u-old_u)/h)-cfg.jerk_limit)**2)
        return (xn,qn,monitor+cost,penalty+h*soft,jnp.maximum(max_slack,slack),jnp.minimum(gap,clearance),u),(xn,qn,u)
    init=(x0,q0,jnp.array(0.),jnp.array(0.),jnp.array(0.),env.clearance(x0),jnp.zeros(2))
    end,(xs,qs,us)=jax.lax.scan(step,init,(reference[:-1],aa))
    return Rollout(jnp.r_[x0[None],xs],jnp.r_[q0[None],qs],us,reference,end[2],end[3],end[4],end[5])


def optimizer(env,cfg,method):
    blind=method in ('blind','blind_filter')
    optimistic=method in ('proposed','optimistic','fading')
    @jax.jit
    def plan(x,q,mean,construction,worlds,validation,previous,key,eta,cold,demonstrations=None):
        if blind:mean=jnp.zeros_like(mean);construction=jnp.zeros_like(construction);worlds=jnp.zeros_like(worlds);validation=jnp.zeros_like(validation)
        def values(theta,bank):
            def one(field):
                r=rollout(theta,x,q,mean,construction,field,env,cfg,not blind)
                return r.monitor+r.penalty
            return jax.vmap(one)(bank)
        fraction=eta if optimistic else 1.
        rho=cfg.mixture if method in ('proposed','fading') else 0.
        initials=initial_sequences(x,cfg)
        guidance=None
        if cfg.initialization=='rrt':
            if demonstrations is None:raise ValueError('RRT guidance requires generated demonstrations')
            initials=jnp.concatenate((initials,demonstrations))
            def guidance(theta,selected,selected_cost):
                points=nominal_reference(theta,x,cfg)[0][:, :2]
                reference=nominal_reference(selected,x,cfg)[0][:, :2]
                # Peak-normalized Gaussian trajectory likelihood. Its full
                # covariance is (H+1)*sigma^2 I; sigma is the RMS position scale.
                return -selected_cost/cfg.temperature-jnp.mean(jnp.sum((points-reference)**2,axis=1))/(2*cfg.guidance_sigma**2)
        result=search(lambda th:values(th,worlds),initials,previous,key,cfg,fraction,rho,cold,guidance)
        costs=jax.lax.map(lambda th:values(th,validation),result.finalists,batch_size=cfg.batch_size)
        scores=lower_tail(costs,fraction)
        valid=jnp.all(jnp.isfinite(costs),axis=1)&jnp.all(jnp.abs(result.finalists)<=cfg.latent_bound,axis=1)
        scores=jnp.where(valid,scores,jnp.inf);i=jnp.argmin(scores)
        return result.finalists[i],scores[i],jnp.mean(costs[i]),result,i
    return plan
