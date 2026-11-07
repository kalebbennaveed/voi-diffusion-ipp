"""Budgeted Bayesian active sensing and posterior-sampling MBD controls.

The full original MBD proposal/score/reverse equations are reused. The active
method maximizes prefix mutual information within an explicit task-cost budget.
Its GP information objective is a surrogate, not an asserted task value of
information. Both planning and execution use full hard RPCBF, eight samples,
and a .12 m margin; the old simulations retain their original settings.
"""
from typing import NamedTuple
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import legacy_monitoring as legacy
from exploration_monitoring import forecast
from voi_diffusion_ipp.quarter.multimode import search

class Result(NamedTuple):
    theta:jax.Array
    score:jax.Array
    expected:jax.Array
    search:object
    index:jax.Array
    task_costs:jax.Array
    information:jax.Array
    contacts:jax.Array
    budget:jax.Array
    initial_budget:jax.Array

def information_gain(b,positions):
    """I(W; noisy wind vectors) under the full current POAM predictive GP.

    Both independent velocity components contribute: 2 × 1/2 log det.
    This retains spatial covariance and conditional residual uncertainty.
    """
    _,cov=b.joint(positions)
    matrix=jnp.eye(len(positions),dtype=cov.dtype)+cov/(b.noise_std**2)
    root=jnp.linalg.cholesky((matrix+matrix.T)*.5)
    return 2*jnp.sum(jnp.log(jnp.diag(root)))

def inducing_information_gain(b,positions):
    """Exact I(U;Y) under current full POAM joint predictive distribution.

    Y|U retains the full spatially correlated conditional residual plus sensor
    noise. Targeting U avoids rewarding irreducible residual uncertainty as if
    it were removed by an update of q(U). No covariance is dropped in physics,
    belief prediction, sampling, or RPCBF.
    """
    cross=b.basis.kernel_centers(positions)
    A=cross@b.inverse
    residual=b.kernel(positions,positions)-A@cross.T
    noise=jnp.eye(len(positions),dtype=residual.dtype)*b.noise_std**2
    conditional=(residual+residual.T)*.5+noise
    total=conditional+A@b.covariance@A.T
    logdet=lambda mat:2*jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky((mat+mat.T)*.5))))
    # Two independent velocity components cancel the scalar-GP factor 1/2.
    return logdet(total)-logdet(conditional)

def optimizer(env,cfg,mode='active',cost_allowance=.25):
    if mode not in ('active','inducing','passive','thompson'):raise ValueError(mode)
    # 5 Hz sensor times within the committed first five seconds. Evaluate the
    # held-control RK4 forecast at these actual sensor timestamps.
    times=jnp.arange(1,round(cfg.commit/.2)+1)*.2
    indices=jnp.minimum(jnp.floor((times-1e-6)/cfg.dt).astype(int),cfg.commit_steps-1)
    offsets=times-indices*cfg.dt
    from voi_diffusion_ipp.wind.gp import WorldField
    @jax.jit
    def plan(b,x,q,mean,construction,worlds,validation,previous,key,seeds):
        def measures(theta,bank):
            def one(field):
                r=forecast(theta,x,q,mean,construction,field,env,cfg)
                world=WorldField(legacy.BASIS,field)
                positions=jax.vmap(lambda z,u,h:legacy.m.physical.rk4(z,u,world,h)[:2])(r.x[indices],r.u[indices],offsets)
                ig=(inducing_information_gain(b,positions) if mode=='inducing' else information_gain(b,positions)) if mode in ('active','inducing') else jnp.array(0.)
                # A failed forecast earns no information from post-contact samples.
                ig=jnp.where(r.terminated,0.,ig)
                return jnp.array([r.monitor+r.penalty,ig,r.terminated.astype(x.dtype)])
            return jax.vmap(one)(bank)
        initials=jnp.r_[seeds,previous[None]]
        initial_values=jax.lax.map(lambda th:measures(th,worlds),initials,batch_size=cfg.batch_size)
        initial_cost=jnp.mean(initial_values[:,:,0],axis=1)
        initial_safe=jnp.all(initial_values[:,:,2]==0,axis=1)
        initial_budget=(1+cost_allowance)*jnp.min(jnp.where(initial_safe,initial_cost,jnp.inf))
        def evaluate(theta):
            values=measures(theta,worlds);cost=values[:,0];ig=values[:,1]
            safe=jnp.all(values[:,2]==0)
            if mode in ('active','inducing'):
                allowed=safe&(jnp.mean(cost)<=initial_budget)
                return jnp.where(allowed,-ig,jnp.inf)
            return jnp.where(safe,cost,jnp.inf)
        # Identical candidate count, Gaussian score, cosine schedule and reverse
        # updates. Only the explicitly documented scalar objective changes.
        result=search(evaluate,seeds,previous,key,cfg,1.,0.,jnp.array(False))
        candidates=result.finalists
        values=jax.lax.map(lambda th:measures(th,validation),candidates,batch_size=cfg.validation_batch_size or None)
        task=jnp.mean(values[:,:,0],axis=1);info=jnp.mean(values[:,:,1],axis=1)
        safe=jnp.all(values[:,:,2]==0,axis=1)&jnp.isfinite(task)&jnp.isfinite(info)
        budget=(1+cost_allowance)*jnp.min(jnp.where(safe,task,jnp.inf))
        if mode in ('active','inducing'):scores=jnp.where(safe&(task<=budget),-info,jnp.inf)
        else:scores=jnp.where(safe,task,jnp.inf)
        index=jnp.argmin(scores)
        return Result(candidates[index],scores[index],task[index],result,index,task,info,values[:,:,2],budget,initial_budget)
    return plan

def field_sampler(cfg):
    @jax.jit
    def sample(b,key,construction_key):
        coefficients=jnp.r_[b.sample_field(construction_key,8),b.sample_field(key,cfg.worlds+cfg.validation_worlds)]
        arrays=jax.vmap(lambda c:b.basis.evaluate(legacy.POINTS,c))(coefficients).reshape((-1,*legacy.SHAPE))
        return b.mean_at(legacy.POINTS).reshape(legacy.SHAPE),arrays[:8],arrays[8:8+cfg.worlds],arrays[8+cfg.worlds:]
    return sample
