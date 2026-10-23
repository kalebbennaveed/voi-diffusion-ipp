"""MBD trajectory optimization for monitoring using dynamics rollout.

There are exactly 2*T/dt independent decision coordinates: both nominal
accelerations at every configured integration interval. Geometry may initialize
distinct routes; diffusion may change every coordinate. There are no fixed route,
target, or dwell decisions, spline knots, or truncated temporal basis.
"""
from dataclasses import dataclass,replace
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
    validation_batch_size:int=0
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
    execution_dt:float=.05
    execution_margin:float=.55
    execution_filter:str='hard'
    penalty_aggregation:str='mean'
    validate_initials:bool=True
    rrt_diversity:bool=False
    prediction_substeps:int=1
    observation_period:float=5.
    prediction_filter:str='soft'
    guidance_mode:str='nominal'
    online_estimation:bool=False
    wind_support_nodes:int=0
    wind_relocate:bool=False
    terminate_predictions:bool=False
    geometry:str='quarter'
    multimode:bool=False
    mode_count:int=4
    initialization_effort:float|None=None
    optimism_anneal_seconds:float|None=None
    detour_depth:float=2.
    wind_at_start:bool=False
    initial_wind_observations:tuple=()
    wind_inducing_count:int=36
    wind_kernel_lengths:tuple=()

    def __post_init__(self):
        super().__post_init__()
        if self.horizon<=0 or abs(self.horizon/self.dt-round(self.horizon/self.dt))>1e-6:raise ValueError('Horizon must be positive and divisible by dt')
        if self.commit<=0 or self.commit>self.horizon:raise ValueError('Commit must be positive and no longer than horizon')
        if min(self.iterations,self.proposals,self.worlds,self.validation_worlds,self.construction_worlds,self.batch_size,self.chains)<1:raise ValueError('All sampling and iteration counts must be positive')
        if self.validation_batch_size<0:raise ValueError('Validation batch size must be nonnegative; zero selects serial execution')
        if self.temperature<=0 or not 0<self.cosine_end<1 or self.latent_bound<=0:raise ValueError('Invalid temperature, cosine interval, or parameter bound')
        if not .35<=self.environment_scale<=1.5:raise ValueError('Invalid environment scale')
        if not .4<=self.inspection_scale<=1.5:raise ValueError('Invalid inspection region scale')
        if self.initialization not in ('gaussian','zero','rrt','modes'):raise ValueError('Unknown MBD initialization')
        if self.geometry not in ('quarter','compact','compact_passages','compact_detour','suite'):raise ValueError('Unknown geometry preset')
        if not 0<=self.detour_depth<=3:raise ValueError('Detour depth must lie in [0,3] metres')
        if self.initial_wind_observations:
            import numpy as np
            observations=np.asarray(self.initial_wind_observations,float)
            if observations.ndim!=2 or observations.shape[1]!=4 or not np.all(np.isfinite(observations)):
                raise ValueError('Initial wind observations must be finite [x,y,wind_x,wind_y] rows')
        if self.mode_count<1:raise ValueError('At least one trajectory mode is required')
        if self.initialization_effort is not None and not 0<self.initialization_effort<1:
            raise ValueError('Initialization acceleration fraction must lie strictly between zero and one')
        if self.optimism_anneal_seconds is not None and self.optimism_anneal_seconds<=0:
            raise ValueError('Optimism annealing duration must be positive')
        if min(self.rrt_paths,self.rrt_visits)<1 or self.guidance_sigma<=0:raise ValueError('Invalid RRT guidance configuration')
        if self.execution_filter not in ('hard','soft'):raise ValueError('Unknown execution filter')
        if self.execution_dt<=0 or abs(self.execution_dt/.01-round(self.execution_dt/.01))>1e-6 or abs(self.dt/self.execution_dt-round(self.dt/self.execution_dt))>1e-6:raise ValueError('Execution control period must be a physics-step multiple and divide planning dt')
        if self.execution_margin<.275:raise ValueError('Execution margin must contain the aircraft')
        if self.penalty_aggregation not in ('mean','tail'):raise ValueError('Unknown constraint-penalty aggregation')
        if self.prediction_substeps<1 or abs(self.dt/self.prediction_substeps/.01-round(self.dt/self.prediction_substeps/.01))>1e-6:raise ValueError('Prediction control period must be a physics-step multiple')
        if self.observation_period<=0 or abs(self.commit/min(self.observation_period,self.commit)-round(self.commit/min(self.observation_period,self.commit)))>1e-6:raise ValueError('Observation period must divide the commit interval')
        if self.prediction_filter not in ('soft','hard'):raise ValueError('Unknown prediction filter')
        if self.guidance_mode not in ('off','nominal','executed'):raise ValueError('Unknown demonstration likelihood')
        if self.online_estimation and abs(min(self.observation_period,self.commit)/self.dt-round(min(self.observation_period,self.commit)/self.dt))>1e-6:raise ValueError('Online observation period must align with trajectory decisions')
        if self.wind_support_nodes not in (0,) and not 3<=self.wind_support_nodes<=33:raise ValueError('Wind sampler support must be zero or a grid dimension between 3 and 33')


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
    terminated:jax.Array


def segment_contact(p0,p1,env,radius=.275):
    """Swept circular footprint against rectangles and the experiment boundary."""
    rectangles=env.rectangles
    corners=jnp.stack((rectangles[:,:2],rectangles[:,2:4],rectangles[:,jnp.array([0,3])],rectangles[:,jnp.array([2,1])]),axis=1)
    delta=p1-p0;length=jnp.maximum(jnp.sum(delta*delta),1e-20)
    t=jnp.clip(jnp.sum((corners-p0)*delta,axis=-1)/length,0.,1.)
    corner_distance=jnp.min(jnp.linalg.norm(corners-(p0+t[:,:,None]*delta),axis=-1))
    moving=jnp.abs(delta)>1e-10;safe=jnp.where(moving,delta,1.)
    lo=(rectangles[:,:2]-p0)/safe;hi=(rectangles[:,2:4]-p0)/safe
    within=(p0>=rectangles[:,:2])&(p0<=rectangles[:,2:4])
    near=jnp.where(moving,jnp.minimum(lo,hi),jnp.where(within,-jnp.inf,jnp.inf))
    far=jnp.where(moving,jnp.maximum(lo,hi),jnp.where(within,jnp.inf,-jnp.inf))
    intersects=jnp.any(jnp.maximum(jnp.max(near,axis=1),0.)<=jnp.minimum(jnp.min(far,axis=1),1.))
    endpoints=jnp.minimum(env.clearance(jnp.r_[p0,jnp.zeros(2)]),env.clearance(jnp.r_[p1,jnp.zeros(2)]))
    return intersects|(jnp.minimum(endpoints,corner_distance)<radius)


def rollout(theta,x0,q0,mean_array,construction,physical,env,cfg,filter_on=True):
    basis=table_basis(cfg.environment_scale,cfg.geometry)
    mean=TableMean(basis,mean_array);inner=InnerFields(basis,construction);world=WorldField(basis,physical)
    prediction_cfg=replace(cfg,safety_margin=cfg.execution_margin) if cfg.prediction_filter=='hard' else cfg
    reference,aa=nominal_reference(theta,x0,cfg)
    def step(carry,inputs):
        x,q,monitor,penalty,max_slack,gap,old_u,alive=carry
        ref,a=inputs
        if filter_on and cfg.prediction_filter=='hard':
            from ..safety.flow_policy_value import filter_control
            nominal=a+cfg.tracking_kp*(ref[:2]-x[:2])+cfg.tracking_kv*(ref[2:]-x[2:])-drag(x[2:],mean.mean_at(x[:2])[0],xp=jnp)
            u,info=filter_control(x,nominal,mean,inner,env,prediction_cfg)
            slack=info.slack
        else:u,slack=applied_control(x,ref,a,mean,inner,env,prediction_cfg,filter_on)
        def f(z):return jnp.r_[z[2:],u+drag(z[2:],world.at(z[:2])[0],xp=jnp)]
        h=cfg.dt/cfg.prediction_substeps;k1=f(x);k2=f(x+h*k1/2);k3=f(x+h*k2/2);k4=f(x+h*k3)
        xn=x+h*(k1+2*k2+2*k3+k4)/6
        contact=segment_contact(x[:2],xn[:2],env) if cfg.terminate_predictions else jnp.array(False)
        observing=alive&~contact
        clearance=env.clearance(xn)
        # Absorbing failure state: keep the last valid position. Subsequent
        # clarity evolves without observations; no post-contact flight or
        # monitoring is credited by the finite-horizon prediction.
        stopped=jnp.r_[x[:2],jnp.zeros(2)]
        xn=jnp.where(observing,xn,jnp.where(contact,stopped,x))
        qn=sensing.step(q,x[:2],xn[:2],env,cfg,h)
        qn=jnp.where(observing,qn,q/(1+h*env.process_noise*q))
        u=jnp.where(alive,u,jnp.zeros_like(u))
        cost=h*.5*(clarity.deficit(q,env.targets)+clarity.deficit(qn,env.targets))
        soft=cfg.violation_weight*jax.nn.relu(cfg.planning_clearance-clearance)**2
        soft+=cfg.speed_penalty*jax.nn.relu(jnp.linalg.norm(xn[2:])-cfg.speed_max)**2
        soft+=cfg.jerk_penalty*jnp.sum(jax.nn.relu(jnp.abs((u-old_u)/h)-cfg.jerk_limit)**2)
        new_gap=jnp.where(alive,jnp.minimum(gap,jnp.where(contact,jnp.minimum(clearance,.275),clearance)),gap)
        new_slack=jnp.where(alive,jnp.maximum(max_slack,slack),max_slack)
        return (xn,qn,monitor+cost,penalty+h*soft*alive,new_slack,new_gap,u,observing),(xn,qn,u)
    init=(x0,q0,jnp.array(0.),jnp.array(0.),jnp.array(0.),env.clearance(x0),jnp.zeros(2),jnp.array(True))
    offsets=jnp.arange(cfg.prediction_substeps)*cfg.dt/cfg.prediction_substeps
    refs=reference[:-1,None,:]
    pp=refs[:,:,:2]+offsets[None,:,None]*refs[:,:,2:]+.5*offsets[None,:,None]**2*aa[:,None,:]
    vv=refs[:,:,2:]+offsets[None,:,None]*aa[:,None,:]
    fine_reference=jnp.concatenate((pp,vv),axis=-1).reshape(-1,4)
    fine_acceleration=jnp.repeat(aa,cfg.prediction_substeps,axis=0)
    end,(xs,qs,us)=jax.lax.scan(step,init,(fine_reference,fine_acceleration))
    # Keep the public rollout arrays at the decision interval. Substeps only
    # resolve the same controller and sensing between decision coordinates.
    xs=xs[cfg.prediction_substeps-1::cfg.prediction_substeps]
    qs=qs[cfg.prediction_substeps-1::cfg.prediction_substeps]
    us=us[cfg.prediction_substeps-1::cfg.prediction_substeps]
    return Rollout(jnp.r_[x0[None],xs],jnp.r_[q0[None],qs],us,reference,end[2],end[3],end[4],end[5],~end[7])


def optimizer(env,cfg,method):
    blind=method in ('blind','blind_filter')
    optimistic=method in ('proposed','optimistic','fading')
    @jax.jit
    def plan(x,q,mean,construction,worlds,validation,previous,key,eta,cold,demonstrations=None):
        if blind:mean=jnp.zeros_like(mean);construction=jnp.zeros_like(construction);worlds=jnp.zeros_like(worlds);validation=jnp.zeros_like(validation)
        def values(theta,bank):
            def one(field):
                r=rollout(theta,x,q,mean,construction,field,env,cfg,not blind)
                return jnp.array([r.monitor,r.penalty])
            components=jax.vmap(one)(bank)
            return aggregate_costs(components[:,0],components[:,1],cfg.penalty_aggregation)
        fraction=eta if optimistic else 1.
        rho=cfg.mixture if method in ('proposed','fading') else 0.
        initials=initial_sequences(x,cfg)
        guidance=None
        if cfg.initialization in ('rrt','modes'):
            if demonstrations is None:raise ValueError('RRT guidance requires generated demonstrations')
            initials=(demonstrations if cfg.multimode or cfg.initialization=='modes' else jnp.concatenate((initials,demonstrations)))
            def guidance(theta,selected,selected_cost):
                points=(rollout(theta,x,q,mean,construction,mean,env,cfg,not blind).x[:,:2]
                        if cfg.guidance_mode=='executed' else nominal_reference(theta,x,cfg)[0][:,:2])
                reference=nominal_reference(selected,x,cfg)[0][:, :2]
                # Peak-normalized Gaussian trajectory likelihood. Its full
                # covariance is (H+1)*sigma^2 I; sigma is the RMS position scale.
                return -selected_cost/cfg.temperature-jnp.mean(jnp.sum((points-reference)**2,axis=1))/(2*cfg.guidance_sigma**2)
            if cfg.guidance_mode=='off':guidance=None
        if cfg.multimode:
            from .multimode import search as mode_search
            result=mode_search(lambda th:values(th,worlds),initials,previous,key,cfg,fraction,rho,cold,guidance)
        else:result=search(lambda th:values(th,worlds),initials,previous,key,cfg,fraction,rho,cold,guidance)
        candidates=result.finalists
        if cfg.validate_initials and not cfg.multimode:
            # A geometrically distinct initial trajectory must not disappear
            # just because it missed the top four on a tiny training bank.
            candidates=jnp.concatenate((candidates,result.initial_parameters))
            candidate_indices=jnp.concatenate((result.finalist_indices,jnp.arange(len(result.initial_parameters))))
            result=result._replace(finalists=candidates,finalist_indices=candidate_indices)
        if cfg.multimode:
            # Serial execution skips exact repeated finalists. A declared
            # batch can trade that skipped work for parallel array execution;
            # the evaluated costs and selection rule are unchanged.
            costs=jax.lax.map(lambda item:jax.lax.cond(item[1],lambda th:values(th,validation),
                lambda th:jnp.full(cfg.validation_worlds,jnp.inf),item[0]),(candidates,result.finalist_unique),
                batch_size=cfg.validation_batch_size or None)
        else:costs=jax.lax.map(lambda th:values(th,validation),candidates,batch_size=cfg.batch_size)
        scores=lower_tail(costs,fraction)
        valid=jnp.all(jnp.isfinite(costs),axis=1)&jnp.all(jnp.abs(result.finalists)<=cfg.latent_bound,axis=1)
        scores=jnp.where(valid,scores,jnp.inf);i=jnp.argmin(scores)
        return result.finalists[i],scores[i],jnp.mean(costs[i]),result,i
    return plan


def aggregate_costs(monitoring,penalties,mode='mean'):
    """Optimism applies to monitoring; mean soft penalties remain in every score.

    Adding the same expected penalty to each field cost preserves the expected
    total cost exactly and makes the favorable-tail score equal to the favorable
    monitoring tail plus the full mean penalty. 'tail' reproduces old runs.
    """
    return monitoring+(penalties if mode=='tail' else jnp.mean(penalties))
