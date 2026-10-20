"""Route-parameter MBD using chronological POAM + sampled RPCBF rollouts.

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
from ..safety.rpcbf import filter_control
from ..wind import InnerFields, WorldField
from ..wind.cache import TableMean
from .mbd import optimize


def visible(a,b,rectangles,margin=.8):
    for rect in rectangles:
        lo,hi=rect[:2]-margin,rect[2:4]+margin
        delta=b-a; t0,t1=0.,1.
        for k in range(2):
            if abs(delta[k])<1e-9:
                if a[k]<lo[k] or a[k]>hi[k]: t0,t1=1.,0.;break
            else:
                ts=sorted([(lo[k]-a[k])/delta[k],(hi[k]-a[k])/delta[k]])
                t0=max(t0,ts[0]);t1=min(t1,ts[1])
        if t0<=t1: return False
    return True


def shortest_path(start,goal,env):
    rect=np.asarray(env.rectangles); corners=[]
    for r in rect:
        for x in [r[0]-1.,r[2]+1.]:
            for y in [r[1]-1.,r[3]+1.]:
                if .6<x<39.4 and .6<y<27.4:
                    corners.append([x,y])
    nodes=np.array([start,goal,*corners]); n=len(nodes)
    # A real tracking deviation may place the robot outside an obstacle but
    # inside the nominal .8 m route inflation. Let the first connection leave
    # that band without pretending the physical robot is trapped in a wall.
    # Visibility uses axis-aligned rectangle inflation (L-infinity), whereas
    # physical clearance uses Euclidean corner distance. Match that geometry.
    axis_gap=np.max(np.maximum(rect[:,:2]-start,start-rect[:,2:4]),axis=1).min()
    start_margin=min(.8,max(0.,float(axis_gap)-.02))
    graph=[[] for _ in nodes]
    for i in range(n):
        for j in range(i):
            if visible(nodes[i],nodes[j],rect,margin=start_margin if j==0 else .8):
                d=np.linalg.norm(nodes[i]-nodes[j]); graph[i].append((j,d));graph[j].append((i,d))
    queue=[(0.,0)];dist={0:0.};prev={}
    while queue:
        d,i=heapq.heappop(queue)
        if i==1:break
        if d>dist[i]:continue
        for j,w in graph[i]:
            nd=d+w
            if nd<dist.get(j,np.inf):dist[j]=nd;prev[j]=i;heapq.heappush(queue,(nd,j))
    if 1 not in dist:raise ValueError(f"No geometric connection from {start} to {goal}")
    path=[1]
    while path[-1]!=0:path.append(prev[path[-1]])
    return nodes[path[::-1]][1:]


def route_library(position,env):
    """Six cross-region routes + two local inspection patterns, all connected."""
    roi=np.asarray(env.regions); centers=(roi[:,:2]+roi[:,2:])/2
    paths=[]
    for target in range(2):
        for route_y in [9.,25.,1.5]:
            # Enter the chosen passage from the side occupied at planning time.
            side=position[0]<21
            enter=np.array([11.,route_y]) if side else np.array([31.,route_y])
            leave=np.array([31.,route_y]) if target==1 else np.array([11.,route_y])
            if 10.5<position[0]<31.5 and abs(position[1]-route_y)<2.2:
                enter=leave.copy()
            p=list(shortest_path(position,enter,env))
            if np.linalg.norm(leave-enter)>1:p.append(leave)
            p.extend(shortest_path(leave,centers[target],env))
            # A full return is inside the horizon when time permits; no reset.
            other=1-target
            p.extend(shortest_path(centers[target],leave,env))
            p.append(np.array([11.,route_y]) if target==1 else np.array([31.,route_y]))
            p.extend(shortest_path(p[-1],centers[other],env))
            paths.append((p,len(shortest_path(position,enter,env))+(int(np.linalg.norm(leave-enter)>1))+len(shortest_path(leave,centers[target],env))-1))
    for target in range(2):
        p=list(shortest_path(position,centers[target],env)); end=len(p)-1
        paths.append((p,end))
    length=18; points=[]; ends=[]; sizes=[]
    for p,e in paths:
        if len(p)>length:raise ValueError("Route library capacity exceeded")
        sizes.append(len(p));ends.append(e)
        points.append(np.vstack((p,np.repeat(np.asarray(p[-1])[None],length-len(p),axis=0))))
    return jnp.asarray(np.array(points),jnp.float32),jnp.array(ends),jnp.array(sizes)


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
    # Two target logits; three passage logits; local-stay switch.
    target=(theta[0]>0).astype(jnp.int32)
    route=jnp.argmax(theta[1:4])
    index=jnp.where(theta[4]>1.,6+target,3*target+route)
    speed=.8+1.8*jax.nn.sigmoid(theta[5])
    dwell=3.+9.*jax.nn.sigmoid(theta[6])
    return index,speed,dwell


def rollout(theta,x0,q0,belief,cache,world,noise,library,env,cfg,anticipate,blind=False,steps=None):
    steps=cfg.stage_steps+cfg.future_steps if steps is None else steps
    routes,ends,sizes=library;index,speed,dwell=decode(theta)
    path=routes[index];end=ends[index];size=sizes[index]
    initial_mean,all_inner=cache.tabulate(belief)

    def tick(carry,item):
        x,reference,q,b,mean,fields,waypoint,held,cost,gap=carry
        time,eps=item
        # Match the real two-second commitment: generate a mean-field planar
        # reference, then predict the translation part of the same tracker.
        reference=jnp.where(time%cfg.stage_steps==0,x,reference)
        reached=jnp.linalg.norm(reference[:2]-path[waypoint])<.9
        held=jnp.where((waypoint==end)&reached,held+cfg.dt,held)
        advance=reached & ((waypoint!=end)|(held>dwell))
        waypoint=jnp.minimum(waypoint+advance.astype(jnp.int32),size-1)
        delta=path[waypoint]-reference[:2]
        distance=jnp.linalg.norm(delta)
        desired_v=delta/jnp.maximum(distance,1e-5)*jnp.minimum(speed,1.3*distance)
        mu=mean.mean_at(reference[:2])[0]
        nominal=jnp.clip(1.7*(desired_v-reference[2:])-mu,-cfg.umax,cfg.umax)
        u,info=filter_control(reference,nominal,mean,fields,env,cfg)
        acceleration=u+mu
        def derivative(z,tau):
            pref=reference[:2]+tau*reference[2:]+.5*tau*tau*acceleration
            vref=reference[2:]+tau*acceleration
            control=acceleration+(3./.680)*(pref-z[:2])+(5./.680)*(vref-z[2:])-mean.mean_at(z[:2])[0]
            return jnp.r_[z[2:],jnp.clip(control,-cfg.umax,cfg.umax)+world.at(z[:2])[0]]
        def integration(z,i):
            h=cfg.dt/4;tt=i*h
            k1=derivative(z,tt);k2=derivative(z+h*k1/2,tt+h/2)
            k3=derivative(z+h*k2/2,tt+h/2);k4=derivative(z+h*k3,tt+h)
            return z+h*(k1+2*k2+2*k3+k4)/6,None
        xn,_=jax.lax.scan(integration,x,jnp.arange(4))
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
    carry=(x0,x0,q0,belief,initial_mean,inner,jnp.array(0),jnp.array(0.),jnp.array(0.),env.clearance(x0))
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
        seeds=[]
        for target in [-1.,1.]:
            for route in range(4):
                t=np.zeros(7,np.float32);t[0]=target;t[1:4]=-.7
                t[1+min(route,2)]=.7;t[4]=2 if route==3 else -1.;t[5]=1.
                seeds.append(t)
        seeds=jnp.asarray(np.array(seeds))
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
