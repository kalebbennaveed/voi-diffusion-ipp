"""Finite Bayesian rollout selection over original-MBD candidate prefixes.

One decision at 5 s; full POAM hypothetical conditioning, noisy observations,
and full hard RPCBF. This is a declared finite-policy/Monte Carlo planning
approximation, not a replacement plant or a new wind model.
"""
from dataclasses import replace
import time
import numpy as np
import jax
import jax.numpy as jnp
import legacy_monitoring as legacy
import monitoring_mbd as m
from voi_diffusion_ipp.wind.cache import TableMean
from voi_diffusion_ipp.wind.gp import InnerFields,WorldField
from voi_diffusion_ipp.safety.flow_policy_value import filter_control

def forecast(theta,x0,q0,mean_array,construction,physical,env,cfg):
    """Original plant/clarity forecast with the actual hard filter definition."""
    mean=TableMean(legacy.BASIS,mean_array);inner=InnerFields(legacy.BASIS,construction)
    world=WorldField(legacy.BASIS,physical);reference,aa=legacy.old.nominal_reference(theta,x0,cfg)
    def step(carry,inputs):
        x,q,cost,penalty,max_slack,gap,alive=carry;ref,a=inputs
        nominal=m.tracking_control(x,ref,a,mean,cfg,True)
        u,info=filter_control(x,nominal,mean,inner,env,cfg)
        xn=m.physical.rk4(x,u,world,cfg.dt)
        hit=m.contact_fraction(x[:2],xn[:2],env)<=1
        observing=alive&~hit;xn=jnp.where(observing,xn,jnp.r_[x[:2],jnp.zeros(2)])
        qn=m.clarity.step(q,x[:2],xn[:2],env.cells,cfg.sensor_sigma,cfg.sensor_R,env.process_noise,cfg.dt)
        qn=jnp.where(observing,qn,q/(1+cfg.dt*env.process_noise*q))
        incurred=cfg.dt*.5*(m.clarity.deficit(q,env.targets)+m.clarity.deficit(qn,env.targets))
        clearance=env.clearance(xn)
        soft=cfg.violation_weight*jax.nn.relu(cfg.planning_clearance-clearance)**2
        soft+=cfg.speed_penalty*jax.nn.relu(jnp.linalg.norm(xn[2:])-cfg.speed_max)**2
        gap=jnp.where(alive,jnp.minimum(gap,jnp.where(hit,0.,clearance)),gap)
        return (xn,qn,cost+incurred,penalty+cfg.dt*soft*alive,jnp.maximum(max_slack,info.slack*alive),gap,observing),(xn,qn,u)
    initial=(x0,q0,jnp.array(0.),jnp.array(0.),jnp.array(0.),env.clearance(x0),jnp.array(True))
    end,(xs,qs,us)=jax.lax.scan(step,initial,(reference[:-1],aa))
    return legacy.old.Rollout(jnp.r_[x0[None],xs],jnp.r_[q0[None],qs],us,reference,end[2],end[3],end[4],end[5],~end[6])

def diverse_prefixes(winner,seeds,x,cfg,count=4):
    """Winner, hold, and geometrically distinct full-control alternatives."""
    bank=np.vstack((np.asarray(winner)[None],np.asarray(seeds)))
    # Exact nominal double-integrator recurrence, used only for seed diversity.
    controls=cfg.umax*np.tanh(bank.reshape(len(bank),cfg.steps,2))
    states=np.broadcast_to(np.asarray(x),(len(bank),4)).copy();paths=[states[:,:2].copy()]
    for k in range(cfg.steps):
        states[:,:2]+=cfg.dt*states[:,2:]+.5*cfg.dt**2*controls[:,k]
        states[:,2:]+=cfg.dt*controls[:,k];paths.append(states[:,:2].copy())
    paths=np.stack(paths,axis=1)
    selected=[0,1]
    while len(selected)<count:
        distances=np.mean(np.sum((paths[:,None]-paths[np.asarray(selected)][None])**2,axis=-1),axis=-1)
        score=distances.min(axis=1);score[selected]=-np.inf;selected.append(int(np.argmax(score)))
    return jnp.asarray(bank[selected]),selected

def continuation_seeds(x,q,env,cfg):
    seeds,_,meta=legacy.seed_bank(x,q,env,cfg)
    if 'initial_destinations' not in meta:
        return jnp.tile(seeds[:1],(4,1))
    destinations=np.asarray(meta['initial_destinations'])
    # The initializer's final two paths are alternative passages to the same
    # geometrically discovered valuable distant destination. Keep its direct
    # path and hold as well. No named A/B destination or true wind is used.
    direct=int(np.argmin(np.linalg.norm(destinations[:4]-destinations[-1],axis=1)))
    return seeds[jnp.asarray([0,1+direct,5,6])]

class BayesianSelector:
    def __init__(self,env,cfg,*,learning=True,roots=4,branches=2,conditional_worlds=2):
        self.env,self.cfg,self.learning=env,cfg,learning
        self.roots,self.branches,self.conditional_worlds=roots,branches,conditional_worlds
        self.prefix_cfg=replace(cfg,horizon=cfg.commit)
        self.tail_cfg=replace(cfg,horizon=cfg.horizon-cfg.commit,commit=min(cfg.commit,cfg.horizon-cfg.commit))
        self.prefix=jax.jit(jax.vmap(lambda th,x,q,mu,inner,w:forecast(th,x,q,mu,inner,w,env,self.prefix_cfg),in_axes=(0,None,None,None,None,0)))
        self.tail=jax.jit(jax.vmap(lambda th,x,q,mu,inner,w:forecast(th,x,q,mu,inner,w,env,self.tail_cfg)))
        count=round(cfg.commit/.2);times=jnp.arange(1,count+1)*.2
        indices=jnp.minimum(jnp.floor((times-1e-6)/cfg.dt).astype(int),round(cfg.commit/cfg.dt)-1)
        offsets=times-indices*cfg.dt
        def update(b,path,controls,world,noise,terminated):
            field=WorldField(legacy.BASIS,world)
            positions=jax.vmap(lambda x,u,h:m.physical.rk4(x,u,field,h)[:2])(path[indices],controls[indices],offsets)
            values=field.at(positions)+b.noise_std*noise
            posterior=(jax.lax.cond(terminated,lambda _:b,lambda _:b.condition(positions,values,relocate=True),None) if learning else b)
            return posterior,positions,values
        self.update=jax.jit(jax.vmap(update,in_axes=(None,0,0,0,0,0)))
        def fields(b,key,construction_key,n):
            c=b.sample_field(construction_key,8);w=b.sample_field(key,n)
            arrays=jax.vmap(lambda z:b.basis.evaluate(legacy.POINTS,z))(jnp.concatenate((c,w))).reshape((-1,*legacy.SHAPE))
            return b.mean_at(legacy.POINTS).reshape(legacy.SHAPE),arrays[:8],arrays[8:]
        self.initial_fields=jax.jit(lambda b,k,ck:fields(b,k,ck,branches))
        self.child_fields=jax.jit(jax.vmap(lambda b,k,ck:fields(b,k,ck,conditional_worlds)))

    def select(self,b,x,q,candidates,key,construction_key):
        started=time.perf_counter();timing={};r,n,cw=self.roots,self.branches,self.conditional_worlds
        mu,inner,worlds=self.initial_fields(b,jax.random.fold_in(key,1),construction_key);jax.block_until_ready(worlds)
        timing['root_fields']=time.perf_counter()-started;t=time.perf_counter()
        roots=jnp.repeat(candidates[:,:2*self.prefix_cfg.steps],n,axis=0)
        physical=jnp.tile(worlds,(r,1,1,1));pre=self.prefix(roots,x,q,mu,inner,physical);jax.block_until_ready(pre)
        timing['prefix_rollouts']=time.perf_counter()-t;t=time.perf_counter()
        noise=jax.random.normal(jax.random.fold_in(key,2),(r*n,round(self.cfg.commit/.2),2))
        posterior,positions,observations=self.update(b,pre.x,pre.u,physical,noise,pre.terminated)
        if not self.learning:posterior=jax.tree.map(lambda z:jnp.broadcast_to(z,(r*n,*z.shape)),b)
        cm,ci,world=self.child_fields(posterior,jax.random.split(jax.random.fold_in(key,3),r*n),jnp.tile(construction_key[None],(r*n,1)));jax.block_until_ready(world)
        timing['posterior_and_fields']=time.perf_counter()-t;t=time.perf_counter()
        children=jnp.stack([continuation_seeds(np.asarray(xx[-1]),np.asarray(qq[-1]),self.env,self.tail_cfg) for xx,qq in zip(pre.x,pre.q)])
        timing['continuation_geometry']=time.perf_counter()-t;t=time.perf_counter();k=children.shape[1]
        parameters=jnp.repeat(children,cw,axis=1).reshape((-1,children.shape[-1]))
        repeat=lambda a:jnp.repeat(a,k*cw,axis=0)
        outcomes=self.tail(parameters,repeat(pre.x[:,-1]),repeat(pre.q[:,-1]),repeat(cm),repeat(ci),jnp.tile(world[:,None],(1,k,1,1,1,1)).reshape((-1,*legacy.SHAPE)))
        costs=jnp.where(outcomes.terminated,jnp.inf,outcomes.monitor+outcomes.penalty).reshape(r*n,k,cw)
        chosen=jnp.argmin(jnp.mean(costs,axis=-1),axis=1)
        # Choose using only posterior-predictive costs, then assess that causal
        # choice in the SAME outer physical field that generated observations.
        selected=children[jnp.arange(r*n),chosen]
        evaluated=self.tail(selected,pre.x[:,-1],pre.q[:,-1],cm,ci,physical);jax.block_until_ready(evaluated)
        remaining_times=jnp.arange(self.tail_cfg.steps+1)*self.tail_cfg.dt
        terminal_q=pre.q[:,-1]
        decayed=terminal_q[:,None,:]/(1+remaining_times[None,:,None]*self.env.process_noise[None,None,:]*terminal_q[:,None,:])
        deficits=jnp.mean(jax.nn.relu(self.env.targets[None,None,:]-decayed),axis=-1)
        stopped_cost=self.tail_cfg.dt*jnp.sum((deficits[:,1:]+deficits[:,:-1])*.5,axis=1)
        # Contact remains absorbing across the look-ahead decision boundary.
        continuation_cost=jnp.where(pre.terminated,stopped_cost,evaluated.monitor+evaluated.penalty)
        total=pre.monitor+pre.penalty+continuation_cost
        safe=~pre.terminated&~evaluated.terminated&jnp.isfinite(jnp.min(jnp.mean(costs,axis=-1),axis=1))
        scores=jnp.where(jnp.all(safe.reshape(r,n),axis=1),total.reshape(r,n).mean(axis=1),jnp.inf)
        timing['continuation_rollouts']=time.perf_counter()-t
        timing['total']=time.perf_counter()-started
        return dict(index=int(jnp.argmin(scores)),scores=np.asarray(scores),timing=timing,
                    prefix_positions=np.asarray(pre.x),prefix_contacts=np.asarray(pre.terminated),
                    future_observation_positions=np.asarray(positions),future_observations=np.asarray(observations),
                    conditional_costs=np.asarray(costs),continuation_indices=np.asarray(chosen),
                    costs=np.asarray(total).reshape(r,n),continuation_contacts=np.asarray(evaluated.terminated),continuation_parameters=np.asarray(selected),
                    predicted_states=np.asarray(jnp.concatenate((pre.x,evaluated.x[:,1:]),axis=1)).reshape(r,n,self.cfg.steps+1,4),
                    predicted_min_clearance=np.asarray(jnp.minimum(pre.min_clearance,evaluated.min_clearance)).reshape(r,n),
                    predicted_contacts=np.asarray(pre.terminated|evaluated.terminated).reshape(r,n))
