"""MBD Eqs. (6), (8), (9) over the entire nominal control sequence.

The ordinary mode is the paper's model-only score estimator. The lower-tail
objective and importance-corrected archive mixture are the user's additions.
Optional RRT demonstrations use the paper's Eq. (13) maximum likelihood.
Neither demonstrations nor initializations restrict the decision coordinates.
Gaussian samples are not projected and no temporal dimension is reduced.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .diffusion import cosine_schedule, lower_tail, importance_log_ratio


def proposal_parameters(y, cumulative_alpha):
    return y/jnp.sqrt(cumulative_alpha), jnp.sqrt(1/cumulative_alpha-1)


def score_and_update(y, samples, log_weights, alpha, cumulative_alpha):
    valid=jnp.isfinite(log_weights)
    maximum=jnp.max(jnp.where(valid,log_weights,-1e30))
    raw=jnp.where(valid,jnp.exp(log_weights-maximum),0.)
    weights=raw/jnp.maximum(jnp.sum(raw),1e-30)
    mean=weights@samples
    score=(-y+jnp.sqrt(cumulative_alpha)*mean)/(1-cumulative_alpha)
    updated=(y+(1-cumulative_alpha)*score)/jnp.sqrt(alpha)
    # Zero total weight is an undefined score, not a valid weighted average.
    return (jnp.where(jnp.any(valid),updated,y),
            jnp.where(jnp.any(valid),score,jnp.zeros_like(y)),weights,jnp.any(valid))


def guided_log_weights(model_log_weight, demonstration_log_weight, log_importance_ratio):
    """Paper Eq. (13), followed by the user's proposal-density correction."""
    return jnp.maximum(model_log_weight,demonstration_log_weight)+log_importance_ratio


class SearchResult(NamedTuple):
    finalists:jax.Array
    finalist_indices:jax.Array
    final:jax.Array
    initial_cost:jax.Array
    trace:jax.Array
    samples:jax.Array
    sample_costs:jax.Array
    weights:jax.Array
    before:jax.Array
    after:jax.Array
    scores:jax.Array
    field_costs:jax.Array
    log_importance_ratios:jax.Array
    archive_centers:jax.Array
    archive_probabilities:jax.Array
    evaluations:jax.Array
    guidance_log_weights:jax.Array
    initial_parameters:jax.Array
    initial_field_costs:jax.Array


def search(evaluate,seeds,previous,key,cfg,eta,rho,cold,log_guidance=None):
    initial=jnp.concatenate((seeds,previous[None]))
    n,k,dim=cfg.iterations,cfg.proposals,previous.size
    size=len(initial)+n*k+1
    alpha,abar=cosine_schedule(n,cfg.cosine_end)
    def batch(parameters):
        values=jax.lax.map(evaluate,parameters,batch_size=cfg.batch_size)
        valid=jnp.all(jnp.isfinite(values),axis=1)&jnp.all(jnp.abs(parameters)<=cfg.latent_bound,axis=1)
        return jnp.where(valid,jnp.mean(values,axis=1),jnp.inf),jnp.where(valid,lower_tail(values,eta),jnp.inf),values
    means0,opts0,initial_fields=batch(initial)
    guide_index=jnp.argmin(opts0);guide=initial[guide_index];guide_cost=opts0[guide_index]
    parameters=jnp.zeros((size,dim)).at[:len(initial)].set(initial)
    means=jnp.full(size,jnp.inf).at[:len(initial)].set(means0)
    opts=jnp.full(size,jnp.inf).at[:len(initial)].set(opts0)
    key,noise_key=jax.random.split(key)
    noise=jax.random.normal(noise_key,(dim,))
    # A cold call uses Algorithm 2's Gaussian initialization. MPC continuation
    # is forward-noised according to Eq. (3), never treated as a fixed route.
    # The optional zero initialization is used by the official model-only
    # implementation. Gaussian remains Algorithm 2's default. This affects the
    # initial iterate only, never the proposal, score, or reverse update.
    initial_iterate=jnp.zeros_like(noise) if cfg.initialization=='zero' else noise
    y=jnp.where(cold,initial_iterate,jnp.sqrt(abar[-1])*previous+jnp.sqrt(1-abar[-1])*noise)
    if cfg.initialization=='rrt':
        y=jnp.sqrt(abar[-1])*guide+jnp.sqrt(1-abar[-1])*noise
    def iteration(carry,j):
        y,parameters,means,opts,key=carry;i=n-1-j
        center,sigma=proposal_parameters(y,abar[i])
        incumbent=jnp.argmin(means)
        priorities=jnp.where(jnp.isfinite(opts)&(means-opts>1e-6),jnp.maximum(opts[incumbent]-opts,0.),0.)
        priority,indices=jax.lax.top_k(priorities,min(cfg.archive_size,size))
        nu=priority/jnp.maximum(jnp.sum(priority),1e-30)
        fraction=jnp.where(jnp.sum(priority)>0,rho,0.)
        centers=parameters[indices]
        key,component_key,proposal_key=jax.random.split(key,3)
        probabilities=jnp.r_[1-fraction,fraction*nu]
        component=jax.random.categorical(component_key,jnp.where(probabilities>0,jnp.log(jnp.maximum(probabilities,1e-30)),-jnp.inf),shape=(k,))
        samples=jnp.concatenate((center[None],centers))[component]+sigma*jax.random.normal(proposal_key,(k,dim))
        mm,oo,field_costs=batch(samples)
        ratio=importance_log_ratio(samples,center,centers,nu,sigma,fraction)
        demonstration=jnp.full_like(oo,-jnp.inf)
        if log_guidance is not None:
            demonstration=jax.vmap(lambda th:log_guidance(th,guide,guide_cost))(samples)
            demonstration=jnp.where(jnp.isfinite(oo),demonstration,-jnp.inf)
        logw=guided_log_weights(-oo/cfg.temperature,demonstration,ratio)
        yn,score,weights,ok=score_and_update(y,samples,logw,alpha[i],abar[i])
        start=len(initial)+j*k
        parameters=jax.lax.dynamic_update_slice(parameters,samples,(start,0))
        means=jax.lax.dynamic_update_slice(means,mm,(start,));opts=jax.lax.dynamic_update_slice(opts,oo,(start,))
        # Explicit empty-batch fallback: restart at an evaluated incumbent at
        # the next noise level. Log it; do not claim an estimated MBD score.
        previous_abar=jnp.where(i>0,abar[jnp.maximum(i-1,0)],1.)
        yn=jnp.where(ok,yn,jnp.sqrt(previous_abar)*parameters[jnp.argmin(opts)])
        ess=jnp.where(ok,1/jnp.maximum(jnp.sum(weights**2),1e-30),0.)
        trace=jnp.array([jnp.min(opts),jnp.min(means),ess,fraction,jnp.mean(jnp.isfinite(logw)),sigma,~ok])
        return (yn,parameters,means,opts,key),(trace,samples,oo,weights,y,yn,score,field_costs,ratio,centers,nu,demonstration)
    (final,parameters,means,opts,key),logs=jax.lax.scan(iteration,(y,parameters,means,opts,key),jnp.arange(n))
    fm,fo,_=batch(final[None]);parameters=parameters.at[-1].set(final);means=means.at[-1].set(fm[0]);opts=opts.at[-1].set(fo[0])
    indices=jnp.r_[jnp.argsort(opts)[:4],jnp.argmin(means),len(initial)-1,size-1]
    return SearchResult(parameters[indices],indices,final,jnp.min(opts0),*logs[:-1],jnp.array(size),logs[-1],initial,initial_fields)
