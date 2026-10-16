"""Causal two-stage stochastic evaluator with shared observation-conditioned recourse."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from ..wind import WorldField, information_gain
from ..clarity import deficit
from .rollout import rollout


class ScenarioBuffers(NamedTuple):
    world_coefficients: jax.Array
    observation_noise: jax.Array
    inner_prior_eps: jax.Array
    inner_future_eps: jax.Array


class TwoStageResult(NamedTuple):
    cost: jax.Array
    first: object
    future: object
    posterior: object
    branch: jax.Array
    info_gain: jax.Array


def make_buffers(belief,key,cfg):
    kw,ko,ki,kf=jax.random.split(key,4)
    shape=(cfg.inner_samples,*belief.mean.shape)
    return ScenarioBuffers(belief.sample_field(kw,cfg.worlds),
        jax.random.normal(ko,(cfg.worlds,cfg.stage_steps,2)),
        jax.random.normal(ki,shape),jax.random.normal(kf,shape))


def route(x,q,belief,env,cfg):
    """Only observable state and posterior enter this API. No field or scenario ID."""
    if cfg.shared_future or cfg.branches==1:
        return jnp.array(0,jnp.int32)
    mean,cov=belief.predict(jnp.array([[0.,0.]]))
    crosswind=jnp.abs(mean[0,1])+1.5*jnp.sqrt(cov[0,1,1])
    # Fixed physical crosswind threshold; not trained on outer truths.
    return (crosswind>.45).astype(jnp.int32)


def decode(vector,cfg):
    knots=cfg.umax*jnp.tanh(vector.reshape((1+cfg.branches,cfg.knots,2)))
    def interpolate(k,n):
        t=jnp.linspace(0,1,n); tk=jnp.linspace(0,1,cfg.knots)
        return jax.vmap(lambda v:jnp.interp(t,tk,v),in_axes=1,out_axes=1)(k)
    return interpolate(knots[0],cfg.stage_steps),jax.vmap(lambda k:interpolate(k,cfg.future_steps))(knots[1:])


def evaluate_one(vector,x,q,belief,coeff,noise,inner_eps,future_eps,env,cfg,mode='proposed'):
    first_controls,futures=decode(vector,cfg)
    world=WorldField(belief.basis,coeff)
    inner=belief.inner_fields(inner_eps)
    first=rollout(x,q,first_controls,belief,inner,world,noise,env,cfg)
    posterior=belief.condition(first.x[1:,:2],first.observations) if cfg.measurements else belief
    stage_belief=posterior if mode in ('proposed','fixed') else belief
    branch=route(first.x[-1],first.q[-1],stage_belief,env,cfg)
    future=rollout(first.x[-1],first.q[-1],futures[branch],stage_belief,
                   stage_belief.inner_fields(future_eps),world,jnp.zeros((cfg.future_steps,2)),
                   env,cfg,feedback=(mode=='fixed'))
    ig=information_gain(belief,posterior)
    cost=first.cost+future.cost+cfg.terminal_seconds*deficit(future.q[-1],env.targets)
    if mode=='passive':
        cost=first.cost+cfg.terminal_seconds*deficit(first.q[-1],env.targets)
    elif mode=='weighted_ig':
        cost=cost-cfg.ig_weight*ig
    elif mode=='heuristic':
        # Deliberately noncausal draft baseline, isolated and explicitly labeled.
        replay=rollout(x,q,first_controls,posterior,posterior.inner_fields(future_eps),world,
                        noise,env,cfg)
        cost=first.cost-jnp.maximum(first.cost-replay.cost,0.)
    return TwoStageResult(cost,first,future,posterior,branch,ig)


def objective(vector,x,q,belief,buffers,env,cfg,mode='proposed'):
    def one(args):
        coeff,noise=args
        result=evaluate_one(vector,x,q,belief,coeff,noise,buffers.inner_prior_eps,
                            buffers.inner_future_eps,env,cfg,mode)
        return result.cost
    # Sequential scenario map bounds memory; all candidates share these exact random buffers.
    return jnp.mean(jax.lax.map(one,(buffers.world_coefficients,buffers.observation_noise)))


def matched_no_information(result,vector,old_belief,coeff,future_eps,env,cfg):
    """Matched physical start and truth; branch chosen from OLD available belief.

Same policy-vector library is held fixed. This measures the filter/router effect,
not an independently optimized value of perfect or imperfect information.
"""
    _,futures=decode(vector,cfg)
    x=result.first.x[-1];q=result.first.q[-1]
    b=route(x,q,old_belief,env,cfg)
    return rollout(x,q,futures[b],old_belief,old_belief.inner_fields(future_eps),
                   WorldField(old_belief.basis,coeff),jnp.zeros((cfg.future_steps,2)),env,cfg)
