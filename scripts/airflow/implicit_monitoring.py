"""Measurement-supported performance planning, separate full-uncertainty RPCBF.

No true wind is accepted by inference, sampling, optimization, or controller APIs.
The plant adapter alone receives it. Existing MBD, robot, clarity, geometry, and
spline policy-value machinery are reused. No information reward or future update.
"""
from dataclasses import replace
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
import known_monitoring as known
import legacy_monitoring as legacy
import monitoring_mbd as m
from supported_poam import condition_supported
from observed_poam import _double
from voi_diffusion_ipp.wind.poam import POAM, select_inducing
from voi_diffusion_ipp.wind.cache import TableMean
from voi_diffusion_ipp.wind.gp import InnerFields, WorldField
from voi_diffusion_ipp.quarter.environment import ExtendedTableBasis
from voi_diffusion_ipp.quarter.multimode import search
from voi_diffusion_ipp.quarter.filter import soft_projection
from voi_diffusion_ipp.safety.flow_policy_value import values, design_policy
from voi_diffusion_ipp.safety.qp import solve
from voi_diffusion_ipp.safety.rpcbf import FilterInfo


class Tables(NamedTuple):
    mean: jax.Array
    std: jax.Array
    support: jax.Array
    planning: jax.Array
    safety: jax.Array


def settings():
    return replace(known.settings(), worlds=2, construction_worlds=2,
                   validation_worlds=2)


def table_grid(bounds, spacing=.2):
    low=np.asarray(bounds[0])-1.
    high=np.asarray(bounds[1])+1.
    nx,ny=np.ceil((high-low)/spacing).astype(int)+1
    yy,xx=np.meshgrid(np.arange(ny),np.arange(nx),indexing='ij')
    pts=low+spacing*np.c_[xx.ravel(),yy.ravel()]
    return ExtendedTableBasis(jnp.asarray(low),jnp.full(2,spacing)),jnp.asarray(pts),(int(ny),int(nx),2)


def initial_belief(env, points):
    """Same 128-variable POAM and fixed attentive kernel; geometry only."""
    options=dict(noise_std=.1,prior_std=1.8,spectral_count=32,
                 prior_support=np.asarray(points)[::3])
    locations=jnp.asarray(env.cells)
    model=POAM(locations[:128],jax.random.PRNGKey(731),**options)
    centers=select_inducing(model.parameters,locations,model.lengths,128)
    return _double(POAM(centers,jax.random.PRNGKey(731),**options).planning_belief())


def information_support(variance, prior_variance, count):
    """Fractional variance reduction; zero until actual observations exist."""
    return jnp.where(count>0,jnp.clip(1-variance/prior_variance,0.,1.),0.)


def planning_fields(mean, draws, variance, prior_variance, count):
    """Declared planning distribution, NOT the unmodified POAM posterior.

    w_plan = mu + sqrt(r) (w_posterior-mu), r = [1-var/var_prior]_0^1.
    Cov_plan(x,y)=sqrt(r(x)) Cov_post(x,y) sqrt(r(y)), hence spatial coherence
    and positive semidefiniteness are retained. Prior-only spread is suppressed
    for performance evaluation only. Safety never receives this covariance.
    """
    r=information_support(variance,prior_variance,count)
    fields=mean[None]+jnp.sqrt(r)[None,:,None]*(draws-mean[None])
    return jnp.where(count>0,fields,jnp.zeros_like(fields)),r


def field_builder(points, shape, cfg):
    @jax.jit
    def build(b, count, performance_key, safety_key):
        # Antithetic whole-field draws retain the same GP spatial covariance
        # and make the finite bank's pointwise mean equal the posterior mean.
        def coefficients(key,n):
            eps=jax.random.normal(key,(n//2,*b.sample_shape),dtype=b.mean.dtype)
            return b.sample_coefficients(jnp.concatenate((eps,-eps)))
        cp=coefficients(performance_key,cfg.worlds)
        cs=coefficients(safety_key,cfg.construction_worlds)
        mu,cov=b.predict(points)
        variance=cov[:,0,0]
        prior=jnp.exp(b.basis.parameters['log_amplitude'])
        draws=jax.vmap(lambda c:b.basis.evaluate(points,c))(cp)
        perf,r=planning_fields(mu,draws,variance,prior,count)
        safe=jax.vmap(lambda c:b.basis.evaluate(points,c))(cs)
        return Tables(mu.reshape(shape),jnp.sqrt(variance).reshape(*shape[:2],1),
                      r.reshape(*shape[:2],1),perf.reshape(cfg.worlds,*shape),
                      safe.reshape(cfg.construction_worlds,*shape))
    return build


def force_support(gradient, velocity, wind_mean, radius):
    """Analytic upper bound for g·d(w,v), for ||w-mu|| <= radius.

    For d(r)=k||r||r, the Jacobian norm is 2k||r||. Integrating along the
    uncertainty segment bounds ||d(r+e)-d(r)|| by k(2||r||R+R²).
    This is a conservative inequality, not a linearized replacement dynamics.
    """
    drift=m.physical.acceleration(velocity,wind_mean,xp=jnp)
    error=(.1/.680)*(2*jnp.linalg.norm(wind_mean-velocity)*radius+radius**2)
    return gradient@drift+jnp.linalg.norm(gradient,axis=-1)*error


def constraints(x, mean, fields, std_table, basis, env, cfg, kappa=3.):
    def f(z):
        B=values(z,mean,fields,env,cfg)
        return B,B
    grad,B=jax.jacfwd(f,has_aux=True)(x)
    winds=fields.at(x[:2])[:,0]
    drift=m.physical.acceleration(x[2:],winds,xp=jnp)
    sample_bound=jnp.max(grad[:,2:]@drift.T,axis=1)
    sigma=jnp.maximum(basis.evaluate(x[:2],std_table)[0,0],0.)
    # Observation noise is retained as a local confidence floor. The full
    # posterior sigma is used even where the planning support is zero.
    radius=kappa*jnp.sqrt(sigma**2+.1**2)
    confidence_bound=force_support(grad[:,2:],x[2:],mean.mean_at(x[:2])[0],radius)
    bound=-cfg.cbf_alpha*B-grad[:,:2]@x[2:]-jnp.maximum(sample_bound,confidence_bound)
    return grad,B,bound


def control(x, nominal, mean, fields, std_table, basis, env, cfg, hard=False):
    grad,B,bound=constraints(x,mean,fields,std_table,basis,env,cfg)
    if hard:
        sol=solve(nominal,grad[:,2:],bound,cfg.umax,design_policy(x,mean,cfg))
        return sol.control,FilterInfo(B,grad,jnp.linalg.norm(sol.control-nominal),
                sol.feasible,sol.slack,sol.residual,jnp.array(0.))
    u,slack=soft_projection(nominal,grad[:,2:],bound,cfg.umax,cfg.slack_weight)
    return u,jnp.max(slack)


def rollout(theta,x0,q0,tables,world_array,basis,env,cfg,aware=True):
    mean=TableMean(basis,tables.mean if aware else jnp.zeros_like(tables.mean))
    safety_mean=TableMean(basis,tables.mean)
    inner=InnerFields(basis,tables.safety)
    world=WorldField(basis,world_array if aware else jnp.zeros_like(world_array))
    refs,aa=legacy.old.nominal_reference(theta,x0,cfg)
    def step(carry,item):
        x,q,cost,penalty,slack,gap,alive=carry
        ref,a=item
        nominal=m.tracking_control(x,ref,a,mean,cfg,True)
        if aware:
            u,sl=control(x,nominal,safety_mean,inner,tables.std,basis,env,cfg)
        else:
            u,sl=jnp.clip(nominal,-cfg.umax,cfg.umax),jnp.array(0.)
        xn=m.physical.rk4(x,u,world,cfg.dt)
        hit=m.contact_fraction(x[:2],xn[:2],env)<=1
        observing=alive&~hit
        xn=jnp.where(observing,xn,jnp.r_[x[:2],jnp.zeros(2)])
        qn=m.clarity.step(q,x[:2],xn[:2],env.cells,cfg.sensor_sigma,cfg.sensor_R,env.process_noise,cfg.dt)
        qn=jnp.where(observing,qn,q/(1+cfg.dt*env.process_noise*q))
        inc=cfg.dt*.5*(m.clarity.deficit(q,env.targets)+m.clarity.deficit(qn,env.targets))
        clearance=env.clearance(xn)
        soft=cfg.violation_weight*jax.nn.relu(cfg.planning_clearance-clearance)**2
        soft+=cfg.speed_penalty*jax.nn.relu(jnp.linalg.norm(xn[2:])-cfg.speed_max)**2
        gap=jnp.where(alive,jnp.minimum(gap,jnp.where(hit,0.,clearance)),gap)
        return (xn,qn,cost+inc,penalty+cfg.dt*soft*alive,jnp.maximum(slack,sl*alive),gap,observing),(xn,qn,u)
    init=(x0,q0,jnp.array(0.),jnp.array(0.),jnp.array(0.),env.clearance(x0),jnp.array(True))
    end,(xs,qs,us)=jax.lax.scan(step,init,(refs[:-1],aa))
    return legacy.old.Rollout(jnp.r_[x0[None],xs],jnp.r_[q0[None],qs],us,refs,end[2],end[3],end[4],end[5],~end[6])


def optimizer(env,cfg,basis,aware=True):
    @jax.jit
    def plan(x,q,tables,seeds,previous,key):
        def forecasts(theta):
            return jax.vmap(lambda w:rollout(theta,x,q,tables,w,basis,env,cfg,aware))(tables.planning)
        def evaluate(theta):
            r=forecasts(theta)
            return r.monitor+r.penalty
        result=search(evaluate,seeds,previous,key,cfg,jnp.array(1.),jnp.array(0.),jnp.array(False))
        r=forecasts(result.final)
        return result.final,jnp.mean(r.monitor+r.penalty),result,r
    return plan


def controller(env,cfg,basis):
    """No truth argument. Both feedback and hard RPCBF use full measured belief."""
    @jax.jit
    def command(x,reference,acceleration,tables):
        mean=TableMean(basis,tables.mean)
        inner=InnerFields(basis,tables.safety)
        nominal=m.tracking_control(x,reference,acceleration,mean,cfg,True)
        u,info=control(x,nominal,mean,inner,tables.std,basis,env,cfg,True)
        return u,nominal,info
    return command


def execution_block(env,cfg,pcfg,basis,seconds=1.):
    command=controller(env,pcfg,basis)
    @jax.jit
    def execute(x,q,refs,aa,offset,tables,hidden_physics_field,alive):
        def tick(carry,k):
            x,q,alive=carry
            local=offset+k*pcfg.execution_dt
            i=jnp.minimum(jnp.floor((local+1e-8)/cfg.dt).astype(int),cfg.steps-1)
            off=local-i*cfg.dt
            ref,a=refs[i],aa[i]
            reference=jnp.r_[ref[:2]+off*ref[2:]+.5*off**2*a,ref[2:]+off*a]
            u,nominal,info=command(x,reference,a,tables)
            end=m.advance(x,q,u,hidden_physics_field,env,pcfg,alive,interval=pcfg.execution_dt)
            xn,qn,living,cost,gap,hit=end
            return (xn,qn,living),(xn,qn,jnp.where(alive,u,0),nominal,
                jnp.where(alive,info.slack,0),jnp.where(alive,info.correction,0),
                jnp.where(alive,info.feasible,True),gap,cost,living,jnp.where(alive,hit,jnp.inf))
        return jax.lax.scan(tick,(x,q,alive),jnp.arange(round(seconds/pcfg.execution_dt)))
    return execute
