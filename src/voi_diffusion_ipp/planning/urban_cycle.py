"""Inspection-cycle MBD with spatial wind velocity and chronological POAM.

The route library supplies geometric initializations, not prerecorded motion.
MBD chooses a target, passage, speed and dwell duration. The same parameters
are used across physical-field samples. Only earned observations change the
feedback controller. This restricted policy class is documented in the demo.
"""
import heapq
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .. import clarity
from ..dynamics import step
from ..safety.flow_policy_value import filter_control
from ..wind import InnerFields, WorldField
from ..wind.cache import TableMean
from .mbd import optimize


from .urban import visible, shortest_path
from ..quadrotor.aerodynamics import acceleration as air_acceleration


def route_library(position,env,residence=None):
    """Periodic inspection policies, with independent outward/return passages.

    Each policy first reaches its target, then repeatedly visits both regions.
    The observed residence time is retained across receding-horizon calls.
    """
    roi=np.asarray(env.regions);centers=(roi[:,:2]+roi[:,2:])/2
    residence=np.zeros(2) if residence is None else np.asarray(residence)
    def leg(start,target,y):
        enter=np.array([11. if start[0]<21 else 31.,y])
        leave=np.array([31. if target==1 else 11.,y])
        if (start[0]<14 and target==0) or (start[0]>28 and target==1):
            return list(shortest_path(start,centers[target],env))
        if 10.5<start[0]<31.5 and abs(start[1]-y)<2.2:enter=leave.copy()
        route=list(shortest_path(start,enter,env))
        if np.linalg.norm(leave-enter)>1:route.append(leave)
        route.extend(shortest_path(leave,centers[target],env))
        return route
    paths=[]
    for target in range(2):
        for out_y in [9.,25.,1.3]:
            for back_y in [9.,25.,1.3]:
                route=leg(position,target,out_y);stops=[len(route)-1]
                loop_start=len(route)
                route.extend(leg(centers[target],1-target,back_y));stops.append(len(route)-1)
                route.extend(leg(centers[1-target],target,out_y));stops.append(len(route)-1)
                paths.append((route,stops,loop_start,residence[target]))
    for target in range(2):
        route=list(shortest_path(position,centers[target],env))
        paths.append((route,[len(route)-1],len(route)-1,residence[target]))
    length=32;points=[];pauses=[];sizes=[];loops=[];held=[]
    for route,stops,loop_start,initial_held in paths:
        if len(route)>length:raise ValueError('Route library capacity exceeded')
        sizes.append(len(route));loops.append(loop_start);held.append(initial_held)
        mask=np.zeros(length,bool);mask[stops]=True;pauses.append(mask)
        points.append(np.vstack((route,np.repeat(np.asarray(route[-1])[None],length-len(route),axis=0))))
    return (jnp.asarray(np.array(points),jnp.float32),jnp.asarray(np.array(pauses)),
            jnp.array(sizes),jnp.array(loops),jnp.array(held))


class Prediction(NamedTuple):
    x: jax.Array
    q: jax.Array
    controls: jax.Array
    acceleration: jax.Array
    cost: jax.Array
    min_clearance: jax.Array
    correction: jax.Array
    slack: jax.Array
    reference_x: jax.Array


def decode(theta):
    target=(theta[0]>0).astype(jnp.int32)
    outward=jnp.argmax(theta[1:4]);returning=jnp.argmax(theta[4:7])
    index=jnp.where(theta[7]>1.,18+target,9*target+3*outward+returning)
    speed=.8+1.8*jax.nn.sigmoid(theta[8])
    dwell=3.+12.*jax.nn.sigmoid(theta[9])
    return index,speed,dwell


def initial_parameters():
    seeds=[]
    for target in [-1.,1.]:
        for outward in range(3):
            for returning in range(3):
                t=np.zeros(10,np.float32);t[0]=target;t[1:7]=-.7
                t[1+outward]=.7;t[4+returning]=.7;t[7]=-1.;t[8]=1.
                seeds.append(t)
        t=seeds[-1].copy();t[7]=2.;seeds.append(t)
    return jnp.asarray(np.array(seeds))


def rollout(theta,x0,q0,belief,cache,world,noise,library,env,cfg,anticipate,blind=False,steps=None):
    steps=cfg.stage_steps+cfg.future_steps if steps is None else steps
    routes,pauses,sizes,loops,initial_held=library;index,speed,dwell=decode(theta)
    path=routes[index];pause=pauses[index];size=sizes[index]
    initial_mean,all_inner=cache.tabulate(belief)

    def tick(carry,item):
        x,reference,q,b,mean,fields,waypoint,held,cost,gap=carry
        time,eps=item
        # Match the real two-second commitment: generate a mean-field planar
        # reference, then predict the translation part of the same tracker.
        reference=jnp.where(time%cfg.stage_steps==0,x,reference)
        reached=jnp.linalg.norm(reference[:2]-path[waypoint])<.35
        held=jnp.where(pause[waypoint]&reached,held+cfg.dt,held)
        advance=reached & ((~pause[waypoint])|(held>dwell))
        held=jnp.where(advance,0.,held)
        following=jnp.where(waypoint==size-1,loops[index],waypoint+1)
        waypoint=jnp.where(advance,following,waypoint)
        delta=path[waypoint]-reference[:2]
        distance=jnp.linalg.norm(delta)
        # A finite-authority vehicle must brake before each geometric corner.
        # This shared speed profile is independent of method and true wind.
        braking_speed=jnp.sqrt(2*.65*jnp.maximum(distance-.3,0.)+.25**2)
        desired_speed=jnp.minimum(speed,jnp.minimum(.85*distance,braking_speed))
        desired_v=delta/jnp.maximum(distance,1e-5)*desired_speed
        mu=air_acceleration(reference[2:],mean.mean_at(reference[:2])[0],xp=jnp)
        nominal=jnp.clip(1.7*(desired_v-reference[2:])-mu,-cfg.umax,cfg.umax)
        u,info=filter_control(reference,nominal,mean,fields,env,cfg)
        acceleration=u+mu
        def derivative(z,tau):
            pref=reference[:2]+tau*reference[2:]+.5*tau*tau*acceleration
            vref=reference[2:]+tau*acceleration
            control=acceleration+(3./.680)*(pref-z[:2])+(5./.680)*(vref-z[2:])-air_acceleration(z[2:],mean.mean_at(z[:2])[0],xp=jnp)
            return jnp.r_[z[2:],jnp.clip(control,-cfg.umax,cfg.umax)+air_acceleration(z[2:],world.at(z[:2])[0],xp=jnp)]
        def integration(z,i):
            h=cfg.dt/8;tt=i*h
            k1=derivative(z,tt);k2=derivative(z+h*k1/2,tt+h/2)
            k3=derivative(z+h*k2/2,tt+h/2);k4=derivative(z+h*k3,tt+h)
            return z+h*(k1+2*k2+2*k3+k4)/6,None
        xn,_=jax.lax.scan(integration,x,jnp.arange(8))
        rn=jnp.r_[reference[:2]+cfg.dt*reference[2:]+.5*cfg.dt**2*acceleration,
                  reference[2:]+cfg.dt*acceleration]
        qn=clarity.step(q,x[:2],xn[:2],env.cells,cfg.sensor_sigma,cfg.sensor_R,env.process_noise,cfg.dt)
        incurred=cfg.dt*.5*(clarity.deficit(q,env.targets)+clarity.deficit(qn,env.targets))
        clearance=env.clearance(xn)
        # Record failure honestly; collision penalty only rejects invalid plans.
        incurred+=cfg.dt*500*jnp.maximum(.2-clearance,0.)
        observation=world.at(xn[:2])[0]+cfg.observation_std*eps
        if anticipate:
            # Each update follows the executed step. Its information first
            # affects the NEXT command. State/clarity never restart.
            update=(time%cfg.stage_steps)==cfg.stage_steps-1
            def learn(args):
                bb,_,_=args
                bb=bb.condition(xn[None,:2],observation[None])
                mm,ff=cache.tabulate(bb)
                return bb,mm,InnerFields(ff.basis,ff.coefficients[cfg.worlds:])
            b,mean,fields=jax.lax.cond(update,learn,lambda args:args,(b,mean,fields))
        return (xn,rn,qn,b,mean,fields,waypoint,held,cost+incurred,jnp.minimum(gap,clearance)),(
            xn,qn,u,u+mu,info.correction,info.slack,rn)

    if blind:
        initial_mean=TableMean(initial_mean.basis,jnp.zeros_like(initial_mean.field))
        inner=InnerFields(all_inner.basis,jnp.zeros_like(all_inner.coefficients[cfg.worlds:]))
    else:inner=InnerFields(all_inner.basis,all_inner.coefficients[cfg.worlds:])
    carry=(x0,x0,q0,belief,initial_mean,inner,jnp.array(0),initial_held[index],jnp.array(0.),env.clearance(x0))
    result,logs=jax.lax.scan(tick,carry,(jnp.arange(steps),noise[:steps]))
    xs,qs,us,aa,correction,slack,rs=logs
    return Prediction(jnp.r_[x0[None],xs],jnp.r_[q0[None],qs],us,aa,result[-2],result[-1],correction,slack,jnp.r_[x0[None],rs])


def optimizer(env,cfg,method):
    def plan(x,q,b,cache,noise,library,key):
        mean,bank=cache.tabulate(b)
        outer=bank.coefficients[:cfg.worlds]
        if method=="blind":outer=jnp.zeros_like(outer)
        def objective(theta):
            def one(c,ns):
                return rollout(theta,x,q,b,cache,WorldField(bank.basis,c),ns,library,env,cfg,
                               method=="proposed",method=="blind").cost
            cost=jnp.mean(jax.vmap(one)(outer,noise))
            return jnp.where(jnp.isfinite(cost),cost,jnp.inf)
        seeds=initial_parameters()
        costs=jax.lax.map(objective,seeds)
        initial=seeds[jnp.argmin(costs)]
        result=optimize(objective,initial,key,cfg)
        return result.vector,result.cost,result.trace,costs
    return jax.jit(plan)


def reference_predictor(env,cfg,method):
    def predict(theta,x,q,b,cache,library):
        mean,_=cache.tabulate(b)
        field=jnp.zeros_like(mean.field) if method=="blind" else mean.field
        result=rollout(theta,x,q,b,cache,WorldField(mean.basis,field),jnp.zeros((cfg.stage_steps,2)),
                       library,env,cfg,False,method=="blind",steps=cfg.stage_steps)
        return result._replace(x=result.reference_x)
    return jax.jit(predict)
