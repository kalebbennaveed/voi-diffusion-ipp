"""ARCHIVED restricted-route planner. Not the requested MBD trajectory optimizer.

Retained only to reproduce and audit the withdrawn v1-v5 recordings.

The finite geometric library exposes all passages to every method. Diffusion
optimizes target/passage logits, inspection dwell, speed and lateral offsets.
All trajectories, sensing and costs are produced by numerical rollouts.
"""
import heapq
import itertools
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from .. import clarity
from . import sensing
from ..planning.urban import visible
from ..wind.cache import TableMean
from ..wind.gp import InnerFields,WorldField
from ..quadrotor.aerodynamics import acceleration as drag
from .environment import TABLE_BASIS,PASSAGES
from .filter import control
from .diffusion import search,lower_tail


def shortest(start,goal,env):
    rect=np.asarray(env.rectangles);bounds=np.asarray(env.bounds)
    corners=[]
    for r in rect:
        corners.extend([[x,z] for x in [r[0]-.9,r[2]+.9] for z in [r[1]-.9,r[3]+.9] if np.all(np.array([x,z])>bounds[0]+.5) and np.all(np.array([x,z])<bounds[1]-.5)])
    points=np.array([start,goal,*corners]);adj=[[] for p in points]
    distance=np.max(np.maximum(rect[:,:2]-start,np.asarray(start)-rect[:,2:4]),axis=1).min()
    for i in range(len(points)):
        for j in range(i):
            if visible(points[i],points[j],rect,min(.7,max(.02,distance-.03)) if j==0 else .7):
                w=float(np.linalg.norm(points[i]-points[j]));adj[i].append((j,w));adj[j].append((i,w))
    best={0:0};prev={};queue=[(0.,0)]
    while queue:
        d,i=heapq.heappop(queue)
        if i==1:break
        if d>best[i]:continue
        for j,w in adj[i]:
            if d+w<best.get(j,np.inf):best[j]=d+w;prev[j]=i;heapq.heappush(queue,(d+w,j))
    if 1 not in best:raise ValueError(f'No connection: {start} -> {goal}')
    ids=[1]
    while ids[-1]!=0:ids.append(prev[ids[-1]])
    return list(points[ids[::-1]][1:])


def sensing_points(env):
    centers=(np.asarray(env.regions)[:,:2]+np.asarray(env.regions)[:,2:])/2
    return centers[:,None,:]+np.array([[-2.25,-2.6],[2.25,-2.6],[2.25,2.6],[-2.25,2.6]])[None]


def library(position,env,residence,inspection_phase=None,continuation=None):
    sites=sensing_points(env);centers=sites.mean(axis=1);N=len(sites)
    def leg(start,goal,passage):
        # Nearby points on the same side never make an artificial crossing.
        if np.sign(start[0])==np.sign(goal[0]) and abs(start[0])>20 and abs(goal[0])>20:return shortest(start,goal,env)
        # Entry side is opposite the destination. The current x sign changes
        # midway through a crossing and must not reverse an unfinished transfer.
        y=PASSAGES[passage];sgn=-1 if goal[0]>0 else 1
        enter=np.array([sgn*24,y]);leave=np.array([-sgn*24,y])
        if abs(start[0])<24 and abs(start[1]-y)<2.6:enter=leave.copy()
        route=shortest(start,enter,env)
        if np.linalg.norm(leave-enter)>.1:route.append(leave)
        route.extend(shortest(route[-1],goal,env));return route
    def inspect(start,site):
        # Preserve completed sensing stops across replans. Re-selecting the
        # nearest stop at every call can reverse the sweep midway between stops.
        nearest=int(inspection_phase[site]) if inspection_phase is not None else int(np.argmin(np.linalg.norm(sites[site]-start,axis=1)))
        ids=[(nearest+i)%4 for i in range(4)]
        return sites[site,ids],ids
    paths=[]
    for target in range(N):
        for outward,back in itertools.product(range(3),repeat=2):
            order=[target];remaining=set(range(N))-{target}
            while remaining:
                nxt=min(remaining,key=lambda n:np.linalg.norm(centers[n]-centers[order[-1]]));order.append(nxt);remaining.remove(nxt)
            route=[];stops=[];hold=[];start=np.asarray(position)
            # Close the loop at the first site's first sensing point.
            for it,site in enumerate(order+[target]):
                pts,ids=inspect(start,site)
                if it==N:pts=pts[:1];ids=ids[:1]
                route.extend(leg(start,pts[0],outward if it%2==0 else back))
                if it==0:loop_start=len(route)-1
                stops.append(len(route)-1);hold.append(residence[site,ids[0]] if it==0 else 0.)
                for pt in pts[1:]:route.append(pt);stops.append(len(route)-1);hold.append(0.)
                start=route[-1]
            paths.append((route,stops,hold,loop_start))
    for site in range(N):
        pts,ids=inspect(np.asarray(position),site);route=shortest(position,pts[0],env);loop_start=len(route)-1
        stops=[len(route)-1];holds=[residence[site,ids[0]]]
        for pt in list(pts[1:])+[pts[0]]:route.append(pt);stops.append(len(route)-1);holds.append(0.)
        paths.append((route,stops,holds,loop_start))
    capacity=64;points=[];pauses=[];sizes=[];loops=[];held=[]
    for route,stops,holds,loop in paths:
        if len(route)>capacity:raise ValueError('Library capacity exceeded')
        mask=np.zeros(capacity,bool);mask[stops]=True
        points.append(np.vstack((route,np.repeat(np.array(route[-1])[None],capacity-len(route),axis=0))))
        pauses.append(mask);sizes.append(len(route));loops.append(loop);held.append(holds[0])
    values=[np.array(points,dtype=np.float32),np.array(pauses),np.array(sizes),np.array(loops),np.array(held,dtype=np.float32),np.zeros(len(points),np.int32)]
    # The selected policy is a stateful route-following controller. Retaining
    # its actual reference cursor prevents every MPC call from promising a
    # fresh inspection sweep before departure. Other route choices start from
    # the current vehicle state, using the same library for every method.
    if continuation is not None:
        selected,old_paths,cursor,dwell_elapsed=continuation
        for k in range(4):values[k][selected]=np.asarray(old_paths[k])[selected]
        values[4][selected]=float(dwell_elapsed)
        values[5][selected]=int(cursor)
    return tuple(jnp.asarray(v) for v in values)


def decode(theta,N,cfg):
    target=jnp.argmax(theta[:N]);outward=jnp.argmax(theta[N:N+3]);back=jnp.argmax(theta[N+3:N+6])
    index=jnp.where(theta[N+6]>1.,N*9+target,target*9+outward*3+back)
    speed=.7+(cfg.speed_max-.7)*jax.nn.sigmoid(theta[N+7]);dwell=1.+5.*jax.nn.sigmoid(theta[N+8])
    offsets=.65*jnp.tanh(theta[N+9:N+12])
    return index,speed,dwell,offsets


def seeds(N):
    result=[]
    for target in range(N):
        for outward,back in itertools.product(range(3),repeat=2):
            x=np.full(N+12,0.,np.float32);x[:N+6]=-.8;x[target]=.8;x[N+outward]=.8;x[N+3+back]=.8;x[N+6]=-1.;x[N+7]=1.;result.append(x)
    for target in range(N):
        x=result[9*target].copy();x[N+6]=2.;result.append(x)
    return jnp.array(np.array(result))

class Prediction(NamedTuple):
    x:jax.Array
    reference:jax.Array
    q:jax.Array
    acceleration:jax.Array
    monitor:jax.Array
    penalty:jax.Array
    max_slack:jax.Array
    min_clearance:jax.Array
    cursor:jax.Array
    dwell_elapsed:jax.Array


def rollout(theta,x0,q0,mean_array,construction,physical,paths,env,cfg,filter_on=True,steps=None):
    mean=TableMean(TABLE_BASIS,mean_array);inner=InnerFields(TABLE_BASIS,construction);world=WorldField(TABLE_BASIS,physical)
    routes,pauses,sizes,loops,initial_held,initial_cursor=paths;index,speed,dwell,offsets=decode(theta,len(env.regions),cfg)
    path=routes[index]
    # Geometric deformation of crossing points; no field-dependent route edits.
    for j,y in enumerate(PASSAGES):path=path.at[:,1].add(jnp.where(jnp.abs(path[:,1]-y)<.01,offsets[j],0.))
    pause=pauses[index]
    def tick(carry,t):
        x,ref,q,wp,held,monitor,penalty,max_slack,gap=carry
        ref=jnp.where(t%cfg.commit_steps==0,x,ref)
        reached=jnp.linalg.norm(ref[:2]-path[wp])<.38
        held=jnp.where(pause[wp]&reached,held+cfg.dt,held)
        advance=reached&((~pause[wp])|(held>=dwell));held=jnp.where(advance,0.,held)
        wp=jnp.where(advance,jnp.where(wp==sizes[index]-1,loops[index].astype(wp.dtype),wp+1),wp)
        delta=path[wp]-ref[:2];distance=jnp.linalg.norm(delta)
        desired_speed=jnp.minimum(speed,jnp.minimum(.9*distance,jnp.sqrt(2*.7*jnp.maximum(distance-.25,0.)+.2**2)))
        vel=delta/jnp.maximum(distance,1e-6)*desired_speed
        drift=drag(ref[2:],mean.mean_at(ref[:2])[0],xp=jnp)
        nominal=jnp.clip(1.7*(vel-ref[2:])-drift,-cfg.umax,cfg.umax)
        u,slack=control(ref,nominal,mean,inner,env,cfg) if filter_on else (nominal,jnp.array(0.))
        acc=u+drift
        def derivative(z,tau):
            pref=ref[:2]+tau*ref[2:]+.5*tau*tau*acc;vref=ref[2:]+tau*acc
            requested=acc+(3/.680)*(pref-z[:2])+(5/.680)*(vref-z[2:])-drag(z[2:],mean.mean_at(z[:2])[0],xp=jnp)
            return jnp.r_[z[2:],jnp.clip(requested,-cfg.umax,cfg.umax)+drag(z[2:],world.at(z[:2])[0],xp=jnp)]
        h=cfg.dt;a=derivative(x,0.);b=derivative(x+h*a/2,h/2);c=derivative(x+h*b/2,h/2);d=derivative(x+h*c,h)
        xn=x+h*(a+2*b+2*c+d)/6
        rn=jnp.r_[ref[:2]+h*ref[2:]+.5*h*h*acc,ref[2:]+h*acc]
        qn=sensing.step(q,x[:2],xn[:2],env,cfg,h)
        cost=h*.5*(clarity.deficit(q,env.targets)+clarity.deficit(qn,env.targets))
        clearance=env.clearance(xn)
        # Finite penalties permit constraint slack during planning. Actual
        # collision/failure is recorded by the nonlinear execution runner.
        extra=h*cfg.violation_weight*jax.nn.relu(.25-clearance)**2
        return (xn,rn,qn,wp,held,monitor+cost,penalty+extra,jnp.maximum(max_slack,slack),jnp.minimum(gap,clearance)),(xn,rn,qn,acc)
    carry=(x0,x0,q0,initial_cursor[index],initial_held[index],jnp.array(0.),jnp.array(0.),jnp.array(0.),env.clearance(x0))
    result,logs=jax.lax.scan(tick,carry,jnp.arange(cfg.steps if steps is None else steps))
    xs,rs,qs,aa=logs
    return Prediction(jnp.r_[x0[None],xs],jnp.r_[x0[None],rs],jnp.r_[q0[None],qs],aa,result[5],result[6],result[7],result[8],result[3],result[4])


def optimizer(env,cfg,method):
    blind=method in ('blind','blind_filter');optimistic=method in ('proposed','optimistic','fading')
    @jax.jit
    def plan(x,q,mean,construction,worlds,validation,paths,previous,key,eta):
        if blind:mean=jnp.zeros_like(mean);construction=jnp.zeros_like(construction);worlds=jnp.zeros_like(worlds);validation=jnp.zeros_like(validation)
        def values(theta,bank):
            def one(field):
                r=rollout(theta,x,q,mean,construction,field,paths,env,cfg,not blind)
                return r.monitor+r.penalty
            return jax.vmap(one)(bank)
        tail=eta if optimistic else 1.
        rho=cfg.mixture if method in ('proposed','fading') else 0.
        finalists,trace,best,evaluations=search(lambda th:values(th,worlds),seeds(len(env.regions)),previous,key,cfg,tail,rho)
        vcost=jax.lax.map(lambda th:values(th,validation),finalists,batch_size=2)
        scores=lower_tail(vcost,tail);scores=jnp.where(jnp.all(jnp.isfinite(vcost),axis=1),scores,jnp.inf)
        ii=jnp.argmin(scores)
        def local_fallback(_):
            local=seeds(len(env.regions))[-len(env.regions):]
            costs=jax.lax.map(lambda th:values(th,validation),local,batch_size=2)
            ss=lower_tail(costs,tail);ss=jnp.where(jnp.all(jnp.isfinite(costs),axis=1),ss,jnp.inf);best=jnp.argmin(ss)
            return local[best],ss[best],jnp.mean(costs[best]),evaluations+len(local)
        th,score,expected,count=jax.lax.cond(jnp.any(jnp.isfinite(scores)),lambda _:(finalists[ii],scores[ii],jnp.mean(vcost[ii]),evaluations),local_fallback,operand=None)
        return th,score,trace,expected,count
    return plan


def reference_predictor(env,cfg,method):
    @jax.jit
    def predict(theta,x,q,mean,construction,paths):
        return rollout(theta,x,q,mean,construction,mean,paths,env,cfg,method!='blind',cfg.commit_steps)
    return predict
