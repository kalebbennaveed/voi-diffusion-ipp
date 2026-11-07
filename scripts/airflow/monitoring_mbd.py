"""Frozen-current-belief MBD in the existing stationary CFD environments.

This composes the original MBD, full POAM, clarity ODE, full spline RPCBF,
and existing relaxed RPCBF QP. No anticipated updates, optimistic aggregation,
policy feedback parameters, interpolated belief cache, or oracle planning.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp

import robot_review as physical
from voi_diffusion_ipp import clarity
from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.environment import Environment
from voi_diffusion_ipp.planning.mbd import optimize, optimize_multistart, target_objective
from voi_diffusion_ipp.quarter.filter import control as relaxed_rpcbf
from voi_diffusion_ipp.safety.flow_policy_value import filter_control
from voi_diffusion_ipp.wind.gp import InnerFields, WorldField
from voi_diffusion_ipp.wind.poam import POAM


@dataclass(frozen=True)
class Settings(Config):
    dt: float = .2                 # Original MBD decision / forecast-filter interval.
    horizon_seconds: float = 20.
    commit_seconds: float = 2.
    mission_seconds: float = 60.
    execution_dt: float = .05       # Existing low-level RPCBF period.
    physics_dt: float = .01
    worlds: int = 2
    inner_samples: int = 3
    diffusion_steps: int = 4
    proposals: int = 8
    candidate_chunk: int = 1
    temperature: float = .025
    mbd_latent_bound: float = 8.
    backup_steps: int = 8
    backup_dt: float = .15
    backup_gain: float = 2.
    cbf_alpha: float = 1.
    safety_margin: float = .06
    umax: float = 1.6
    linear_drag: float = 0.
    slack_weight: float = 15.       # Original relaxed filter's squared-slack weight.
    spline: str = 'full'
    sensor_sigma: float = .5
    sensor_R: float = .1
    clarity_decay: float = .16
    background_decay: float = .02
    region_target: float = .7
    background_target: float = .35
    initial_q: float = .74
    wind_std: float = 1.8
    observation_std: float = .1
    poam_train_steps: int = 10
    poam_inducing_x: int = 6
    poam_inducing_y: int = 4
    poam_spectral_nodes: int = 32
    wind_backend: str = 'poam'
    whole_field: bool = True
    future_updates: bool = False
    reference_tracking: bool = False
    multimode: bool = False
    tracking_kp: float = 1.2
    tracking_kv: float = 1.8
    legacy_safety_costs: bool = False  # False reproduces the archived clarity-only runs.
    planning_clearance: float = .6
    violation_weight: float = 1.
    speed_max: float = 3.1
    speed_penalty: float = 2.

    @property
    def steps(self): return round(self.horizon_seconds/self.dt)

    @property
    def commit_steps(self): return round(self.commit_seconds/self.dt)

    def __post_init__(self):
        super().__post_init__()
        for duration, interval in [(self.horizon_seconds,self.dt),(self.commit_seconds,self.dt),
                                   (self.dt,self.execution_dt),(self.execution_dt,self.physics_dt),
                                   (self.mission_seconds,self.commit_seconds)]:
            if duration <= 0 or abs(duration/interval-round(duration/interval))>1e-8:
                raise ValueError('Monitoring intervals must be positive exact multiples')
        if self.future_updates or self.spline!='full':
            raise ValueError('This comparison requires frozen forecast belief and full spline RPCBF')


class StillAir(NamedTuple):
    def at(self,p): return jnp.zeros_like(jnp.atleast_2d(p))


class MeanWind(NamedTuple):
    belief: object
    def at(self,p): return self.belief.mean_at(p)


def environment(cfg):
    xx,yy=np.meshgrid((np.arange(64)+.5)*.125,(np.arange(37)+.5)*.125)
    all_cells=np.c_[xx.ravel(),yy.ravel()]
    free=physical.clearance(all_cells)>0
    cells=all_cells[free]
    membership=np.c_[(cells[:,0]>=4.98)&(cells[:,1]>=2.295),
                     (cells[:,0]<=2.9)&(cells[:,1]<=1.755)]
    important=membership.any(axis=1)
    env=Environment(jnp.asarray(cells),jnp.asarray(np.where(important,cfg.region_target,cfg.background_target)),
                    jnp.asarray(physical.OBSTACLES),jnp.asarray(physical.BOUNDS),
                    jnp.array([6.5,3.525]),jnp.array([1.45,.875]),jnp.zeros(2),
                    jnp.asarray(np.where(important,cfg.clarity_decay,cfg.background_decay)))
    return env,dict(grid=[64,37],dx_m=.125,cells=cells,all_cells=all_cells,
                    free_indices=np.flatnonzero(free),membership=membership)


def estimator(cfg,key):
    xx,yy=np.meshgrid(np.linspace(.2,7.8,cfg.poam_inducing_x),np.linspace(.2,4.425,cfg.poam_inducing_y))
    sx,sy=np.meshgrid(np.linspace(0,8,29),np.linspace(0,4.625,23))
    return POAM(np.c_[xx.ravel(),yy.ravel()],key,noise_std=cfg.observation_std,
                prior_std=cfg.wind_std,spectral_count=cfg.poam_spectral_nodes,
                prior_support=np.c_[sx.ravel(),sy.ravel()])


def contact_fraction(a,b,env):
    """Earliest analytic swept point contact with the existing circles/walls."""
    d=b-a
    lo=jnp.where((d<0)&(b<=env.bounds[0]),(env.bounds[0]-a)/jnp.where(d!=0,d,1),jnp.inf)
    hi=jnp.where((d>0)&(b>=env.bounds[1]),(env.bounds[1]-a)/jnp.where(d!=0,d,1),jnp.inf)
    delta=a-env.obstacles[:,:2]
    aa=jnp.sum(d*d);bb=2*jnp.sum(delta*d,axis=1)
    cc=jnp.sum(delta*delta,axis=1)-env.obstacles[:,2]**2
    disc=bb*bb-4*aa*cc
    root=(-bb-jnp.sqrt(jnp.maximum(disc,0)))/(2*jnp.maximum(aa,1e-30))
    valid=(aa>1e-25)&(disc>=0)&(root>=0)&(root<=1)
    root=jnp.where(valid,root,jnp.inf)
    fraction=jnp.min(jnp.r_[lo,hi,root])
    return jnp.where(env.clearance(jnp.r_[a,jnp.zeros(2)])<0,0.,fraction)


def advance(x,q,u,world,env,cfg,alive=True,interval=None):
    """Same RK4 air-force plant, .01 s steps; clarity on the executed motion.

    After contact there is no further motion or observation credit. Clarity
    continues to decay, so a failed mission never receives a shorter cost.
    """
    interval=cfg.dt if interval is None else interval
    n=round(interval/cfg.physics_dt);h=interval/n
    def tick(carry,_):
        x,q,alive,cost,gap,hit_time=carry
        xn=physical.rk4(x,u,world,h)
        fraction=contact_fraction(x[:2],xn[:2],env)
        hit=alive & (fraction<=1)
        f=jnp.minimum(fraction,1.)
        end=x+f*(xn-x)
        end=jnp.where(hit,jnp.r_[end[:2],jnp.zeros(2)],end)
        end=jnp.where(alive,end,x)
        sensed=clarity.step(q,x[:2],end[:2],env.cells,cfg.sensor_sigma,cfg.sensor_R,env.process_noise,h*f)
        sensed=sensed/(1+h*(1-f)*env.process_noise*sensed)
        qn=jnp.where(alive,sensed,q/(1+h*env.process_noise*q))
        cost+=.5*h*(clarity.deficit(q,env.targets)+clarity.deficit(qn,env.targets))
        gap=jnp.minimum(gap,jnp.where(hit,0.,env.clearance(end)))
        hit_time=jnp.where(hit,_[0]*h+f*h,hit_time)
        return (end,qn,alive&~hit,cost,gap,hit_time),None
    initial=(x,q,jnp.asarray(alive),jnp.array(0.),env.clearance(x),jnp.array(jnp.inf))
    end,_=jax.lax.scan(tick,initial,(jnp.arange(n),))
    return end


class Forecast(NamedTuple):
    cost: jax.Array
    states: jax.Array
    clarity: jax.Array
    controls: jax.Array
    min_clearance: jax.Array
    max_slack: jax.Array
    collided: jax.Array
    monitoring_cost: jax.Array
    safety_cost: jax.Array


def safety_cost(state,clearance,cfg):
    """Original study's clearance and speed terms, without optimism/bonuses."""
    if not cfg.legacy_safety_costs:return jnp.array(0.)
    return (cfg.violation_weight*jax.nn.relu(cfg.planning_clearance-clearance)**2
            +cfg.speed_penalty*jax.nn.relu(jnp.linalg.norm(state[2:])-cfg.speed_max)**2)



def nominal_reference(theta,x,cfg):
    accelerations=cfg.umax*jnp.tanh(theta.reshape(cfg.steps,2))
    def step(state,a):
        end=jnp.r_[state[:2]+cfg.dt*state[2:]+.5*cfg.dt**2*a,state[2:]+cfg.dt*a]
        return end,state
    _,references=jax.lax.scan(step,x,accelerations)
    return references,accelerations


def tracking_control(state,reference,acceleration,belief,cfg,aware=True):
    wind=belief.mean_at(state[:2])[0] if aware else jnp.zeros(2)
    nominal=acceleration+cfg.tracking_kp*(reference[:2]-state[:2])+cfg.tracking_kv*(reference[2:]-state[2:])
    nominal-=physical.acceleration(state[2:],wind,xp=jnp)
    return nominal


def forecast(theta,x,q,belief,inner,world,env,cfg,aware):
    refs,nominal=nominal_reference(theta,x,cfg)
    def tick(carry,item):
        nom,ref=item
        x,q,cost,alive,gap,maximum,penalty=carry
        if cfg.reference_tracking:nom=tracking_control(x,ref,nom,belief,cfg,aware)
        if aware:
            u,slack=relaxed_rpcbf(x,nom,belief,inner,env,cfg)
        else:
            u,slack=jnp.clip(nom,-cfg.umax,cfg.umax),jnp.array(0.)
        xn,qn,survived,incurred,gap_step,_=advance(x,q,u,world,env,cfg,alive)
        penalty+=cfg.dt*safety_cost(xn,env.clearance(xn),cfg)*alive
        return (xn,qn,cost+incurred,survived,jnp.minimum(gap,gap_step),jnp.maximum(maximum,slack),penalty),(xn,qn,u)
    end,rows=jax.lax.scan(tick,(x,q,jnp.array(0.),jnp.array(True),env.clearance(x),jnp.array(0.),jnp.array(0.)),(nominal,refs))
    xs,qs,us=rows
    return Forecast(end[2]+end[6],jnp.r_[x[None],xs],jnp.r_[q[None],qs],us,end[4],end[5],~end[3],end[2],end[6])


def planner(env,cfg,aware):
    """Ordinary expected-cost MBD, with the current belief frozen throughout.

    Baseline deliberately accepts no belief or hidden field in its objective.
    Full 2*H nominal controls are independently optimized; seeds are only starts.
    """
    @jax.jit
    def plan(x,q,belief,inner,outer,initials,key):
        def objective(theta):
            return forecast_cost(theta,x,q,belief,inner,outer,env,cfg,aware)
        costs=jax.lax.map(target_objective(objective,cfg),initials,batch_size=cfg.candidate_chunk)
        result=(optimize_multistart(objective,initials,key,cfg) if cfg.multimode
                else optimize(objective,initials[jnp.argmin(costs)],key,cfg))
        return result._replace(evaluations=result.evaluations+len(initials)),costs
    return plan


def forecast_cost(theta,x,q,belief,inner,outer,env,cfg,aware):
    """Exact objective, evaluating clarity only for contact-free trajectories.

    The frozen-belief controller and physical motion do not depend on clarity.
    Solve those first with the original .01 s RK4 and swept contact checks.
    Then apply the identical clarity steps on every saved physical segment.
    A rejected trajectory already has infinite cost, so its clarity need not
    be evaluated. No spatial or temporal model approximation is introduced.
    """
    if not cfg.reject_collisions or cfg.legacy_safety_costs:
        return absorbing_forecast_cost(theta,x,q,belief,inner,outer,env,cfg,aware)
    refs,nominal=nominal_reference(theta,x,cfg)
    worlds=cfg.worlds if aware else 1
    n=round(cfg.dt/cfg.physics_dt);h=cfg.dt/n
    def motion(state,u,world):
        def tick(carry,_):
            state,alive=carry;xn=physical.rk4(state,u,world,h)
            fraction=contact_fraction(state[:2],xn[:2],env)
            hit=alive & (fraction<=1);f=jnp.minimum(fraction,1.)
            end=state+f*(xn-state)
            end=jnp.where(hit,jnp.r_[end[:2],jnp.zeros(2)],end)
            end=jnp.where(alive,end,state)
            return (end,alive&~hit),end[:2]
        (end,alive),positions=jax.lax.scan(tick,(state,jnp.array(True)),None,length=n)
        return end,alive,jnp.r_[state[None,:2],positions]
    initial=(jnp.array(0),jnp.tile(x,(worlds,1)),
             jnp.zeros((cfg.steps,worlds,n+1,2)),jnp.array(True))
    def condition(carry):return (carry[0]<cfg.steps)&carry[3]
    def step(carry):
        i,xs,positions,valid=carry;nom=nominal[i]
        if aware:
            def one(state,coeff):
                command=tracking_control(state,refs[i],nom,belief,cfg,True) if cfg.reference_tracking else nom
                u,_=relaxed_rpcbf(state,command,belief,inner,env,cfg)
                return motion(state,u,WorldField(outer.basis,coeff))
            xn,alive,ps=jax.vmap(one)(xs,outer.coefficients)
        else:
            command=tracking_control(xs[0],refs[i],nom,None,cfg,False) if cfg.reference_tracking else nom
            state,alive,p=motion(xs[0],jnp.clip(command,-cfg.umax,cfg.umax),StillAir())
            xn,alive,ps=state[None],alive[None],p[None]
        return i+1,xn,positions.at[i].set(ps),jnp.all(alive)&jnp.all(jnp.isfinite(xn))
    end=jax.lax.while_loop(condition,step,initial)
    def sensing_cost(positions):
        def one_world(path):
            def period(carry,segment):
                q,total=carry
                def tick(carry,points):
                    q,cost=carry;p0,p1=points
                    qn=clarity.step(q,p0,p1,env.cells,cfg.sensor_sigma,cfg.sensor_R,env.process_noise,h)
                    cost+=.5*h*(clarity.deficit(q,env.targets)+clarity.deficit(qn,env.targets))
                    return (qn,cost),None
                (qn,incurred),_=jax.lax.scan(tick,(q,jnp.array(0.)),(segment[:-1],segment[1:]))
                return (qn,total+incurred),None
            (_,cost),_=jax.lax.scan(period,(q,jnp.array(0.)),path)
            return cost
        cost=jnp.mean(jax.vmap(one_world)(jnp.swapaxes(positions,0,1)))
        return jnp.where(jnp.isfinite(cost),cost,jnp.inf)
    return jax.lax.cond(end[3],sensing_cost,lambda _:jnp.array(jnp.inf),end[2])


def absorbing_forecast_cost(theta,x,q,belief,inner,outer,env,cfg,aware):
    """Exact motion/clarity reordering, including every fractional contact.

    Controller and physics are independent of q. Store each physical segment
    and its pre-contact sensing duration; then integrate the unchanged clarity
    equation. Absorbed worlds skip further motion and retain pure decay cost.
    """
    refs,nominal=nominal_reference(theta,x,cfg);worlds=cfg.worlds if aware else 1
    n=round(cfg.dt/cfg.physics_dt);h=cfg.dt/n
    def motion(state,u,world,alive):
        def tick(carry,_):
            z,living=carry;zn=physical.rk4(z,u,world,h)
            fraction=contact_fraction(z[:2],zn[:2],env);hit=living&(fraction<=1);f=jnp.minimum(fraction,1.)
            end=z+f*(zn-z);end=jnp.where(hit,jnp.r_[end[:2],jnp.zeros(2)],end);end=jnp.where(living,end,z)
            return (end,living&~hit),(end[:2],jnp.where(living,f,0.))
        (end,living),(ps,fs)=jax.lax.scan(tick,(state,alive),None,length=n)
        return end,living,jnp.r_[state[None,:2],ps],fs
    initial=(jnp.array(0),jnp.tile(x,(worlds,1)),jnp.ones(worlds,dtype=bool),
             jnp.zeros((cfg.steps,worlds,n+1,2)),jnp.zeros((cfg.steps,worlds,n)),jnp.zeros(worlds))
    def step(carry):
        i,xs,living,positions,fractions,penalty=carry;nom=nominal[i]
        if aware:
            def one(item):
                state,coeff,alive=item
                def moving(_):
                    command=tracking_control(state,refs[i],nom,belief,cfg,True) if cfg.reference_tracking else nom
                    u,_=relaxed_rpcbf(state,command,belief,inner,env,cfg)
                    return motion(state,u,WorldField(outer.basis,coeff),alive)
                def stopped(_):return state,alive,jnp.tile(state[:2],(n+1,1)),jnp.zeros(n)
                return jax.lax.cond(alive,moving,stopped,None)
            xn,alive,ps,fs=jax.lax.map(one,(xs,outer.coefficients,living))
        else:
            command=tracking_control(xs[0],refs[i],nom,None,cfg,False) if cfg.reference_tracking else nom
            state,alive,p,f=motion(xs[0],jnp.clip(command,-cfg.umax,cfg.umax),StillAir(),living[0])
            xn,alive,ps,fs=state[None],alive[None],p[None],f[None]
        penalty+=cfg.dt*jax.vmap(lambda z:safety_cost(z,env.clearance(z),cfg))(xn)*living
        return i+1,xn,alive,positions.at[i].set(ps),fractions.at[i].set(fs),penalty
    end=jax.lax.while_loop(lambda z:z[0]<cfg.steps,step,initial)
    def one_world(item):
        path,fractions=item
        def period(carry,item):
            q,total=carry;segment,fractions=item
            def tick(carry,item):
                q,cost=carry;p0,p1,f=item
                qs=clarity.step(q,p0,p1,env.cells,cfg.sensor_sigma,cfg.sensor_R,env.process_noise,h*f)
                qn=qs/(1+h*(1-f)*env.process_noise*qs)
                return (qn,cost+.5*h*(clarity.deficit(q,env.targets)+clarity.deficit(qn,env.targets))),None
            (qn,incurred),_=jax.lax.scan(tick,(q,jnp.array(0.)),(segment[:-1],segment[1:],fractions))
            return (qn,total+incurred),None
        (_,cost),_=jax.lax.scan(period,(q,jnp.array(0.)),(path,fractions))
        return cost
    cost=jnp.mean(jax.vmap(one_world)((jnp.swapaxes(end[3],0,1),jnp.swapaxes(end[4],0,1))))
    cost+=jnp.mean(end[5])
    valid=jnp.isfinite(cost)&(jnp.all(end[2]) if cfg.reject_collisions else True)
    return jnp.where(valid,cost,jnp.inf)


def execution_kernel(env,cfg):
    @jax.jit
    def execute(x,q,nominal,belief,inner,truth,alive):
        u,info=filter_control(x,nominal,belief,inner,env,cfg)
        end=advance(x,q,u,truth,env,cfg,alive,interval=cfg.execution_dt)
        return end,u,info
    return execute


def seed_bank(x,q,env,cfg,previous=None,site=physical.SITE):
    """Shared, wind-blind initializations propagated with the full plant.

    Existing C2 route references and their timing initialize transfers. The
    still-air quadratic force is compensated while constructing each seed and
    retained in its .01 s RK4 propagation. The true field and wind belief are
    never consulted. MBD still optimizes every acceleration coordinate.
    """
    x=np.asarray(x);q=np.asarray(q)
    def derivative(state,u):
        return np.r_[state[2:],u+physical.acceleration(state[2:],np.zeros(2))]
    def propagate(state,u):
        n=round(cfg.dt/cfg.physics_dt);h=cfg.dt/n
        for _ in range(n):
            a=derivative(state,u);b=derivative(state+h*a/2,u)
            c=derivative(state+h*b/2,u);d=derivative(state+h*c,u)
            state=state+h*(a+2*b+2*c+d)/6
        return state
    def generate(reference=None,point=None):
        state=x.copy();us=[]
        if reference is not None:
            ts=np.linspace(0,reference.duration,2001)
            start=ts[np.argmin(np.linalg.norm(reference.at(ts)[0]-x[:2],axis=1))]
        for k in range(cfg.steps):
            if reference is not None:
                p,v,a=reference.at(start+k*cfg.dt)
                desired=a+physical.KP*(p-state[:2])+physical.KV*(v-state[2:])
            else:
                delta=point-state[:2];length=np.linalg.norm(delta)
                desired_v=delta/max(length,1e-10)*min(.9,1.3*length)
                desired=1.8*(desired_v-state[2:])
            u=np.clip(desired-physical.acceleration(state[2:],np.zeros(2)),-.95*cfg.umax,.95*cfg.umax)
            us.append(u);state=propagate(state,u)
        return np.arctanh(np.asarray(us)/cfg.umax).ravel()
    seeds=[generate(point=x[:2])]
    for route in ('central','lower','upper'):
        for direction in ('A-B','B-A'):
            seeds.append(generate(reference=physical.Reference(route,direction,site)))
    cells=np.asarray(env.cells);need=np.maximum(np.asarray(env.targets)-q/(1+cfg.horizon_seconds*np.asarray(env.process_noise)*q),0)
    distance=np.sum((cells-x[:2])**2,axis=1);score=need/(.25+distance)
    feasible=physical.clearance(cells)>.15
    for _ in range(2):
        i=int(np.argmax(np.where(feasible,score,-np.inf)));seeds.append(generate(point=cells[i]))
        feasible &= np.linalg.norm(cells-cells[i],axis=1)>.7
    if previous is None:previous=np.zeros(cfg.steps*2)
    else:
        a=np.asarray(previous).reshape(cfg.steps,2)
        previous=np.r_[a[cfg.commit_steps:],np.zeros_like(a[:cfg.commit_steps])].ravel()
    return jnp.asarray(np.stack([previous,*seeds]))
