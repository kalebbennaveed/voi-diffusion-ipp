"""Cosine MBD with a favorable-tail objective and corrected archive mixture.

A warm-start cosine interval [0,.75] avoids spending a short finite search in
an almost pure-noise distribution. All iteration budgets share the same final
noise level; the endpoint and the exact cumulative alphas are recorded.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def cosine_schedule(n,end=.75,offset=.008):
    tau=jnp.linspace(0.,end,n+1)
    abar=jnp.cos((tau+offset)/(1+offset)*jnp.pi/2)**2
    abar=abar/abar[0]
    return abar[1:]/abar[:-1],abar[1:]


def lower_tail(costs,eta):
    sorted_cost=jnp.sort(costs,axis=-1)
    count=eta*costs.shape[-1]
    # Fractional last sample, including eta*S<1 and eta=1 endpoints.
    weight=jnp.clip(count-jnp.arange(costs.shape[-1]),0.,1.)
    return jnp.sum(sorted_cost*weight,axis=-1)/count


def log_normal(x,mean,sigma):
    return -.5*jnp.sum(((x-mean)/sigma)**2,axis=-1)-x.shape[-1]*jnp.log(sigma)-.5*x.shape[-1]*jnp.log(2*jnp.pi)


def importance_log_ratio(samples,ordinary_center,centers,priority,sigma,rho):
    log_phi=log_normal(samples,ordinary_center,sigma)
    logs=log_normal(samples[:,None,:],centers[None,:,:],sigma)
    log_psi=logsumexp(logs+jnp.where(priority>0,jnp.log(jnp.maximum(priority,1e-30)),-jnp.inf)[None],axis=1)
    denominator=jnp.logaddexp(jnp.log1p(-rho)+log_phi,jnp.where(rho>0,jnp.log(jnp.maximum(rho,1e-30))+log_psi,-jnp.inf))
    return log_phi-denominator


def search(evaluate,seeds,previous,key,cfg,eta,rho):
    n=cfg.iterations;k=cfg.proposals;dim=seeds.shape[1];size=len(seeds)+1+n*k+1
    alpha,abar=cosine_schedule(n,cfg.cosine_end)
    seeds=jnp.concatenate((seeds,previous[None]))
    def batch(xs):
        values=jax.lax.map(evaluate,xs,batch_size=8)
        finite=jnp.all(jnp.isfinite(values),axis=1)&jnp.all(jnp.abs(xs)<=cfg.latent_bound,axis=1)
        mean=jnp.mean(values,axis=1);opt=lower_tail(values,eta)
        return jnp.where(finite,mean,jnp.inf),jnp.where(finite,opt,jnp.inf)
    mean,opt=batch(seeds);ix=jnp.argmin(opt);initial=seeds[ix]
    theta=jnp.zeros((size,dim)).at[:len(seeds)].set(seeds)
    means=jnp.full(size,jnp.inf).at[:len(seeds)].set(mean)
    opts=jnp.full(size,jnp.inf).at[:len(seeds)].set(opt)
    key,ki=jax.random.split(key)
    y=jnp.sqrt(abar[-1])*initial+jnp.sqrt(1-abar[-1])*jax.random.normal(ki,(dim,))
    def iteration(carry,j):
        y,theta,means,opts,key=carry;i=n-1-j
        center=y/jnp.sqrt(abar[i]);sigma=jnp.sqrt(1/abar[i]-1)
        incumbent=jnp.argmin(means)
        priorities=jnp.where(jnp.isfinite(opts)&(means-opts>1e-6),jnp.maximum(opts[incumbent]-opts,0.),0.)
        ap,idx=jax.lax.top_k(priorities,cfg.archive_size)
        nu=ap/jnp.maximum(ap.sum(),1e-30);rr=jnp.where(ap.sum()>0,rho,0.)
        centers=theta[idx];key,kc,kn=jax.random.split(key,3)
        probabilities=jnp.r_[1-rr,rr*nu]
        component=jax.random.categorical(kc,jnp.log(jnp.maximum(probabilities,1e-30)),shape=(k,))
        mus=jnp.concatenate((center[None],centers))[component]
        samples=mus+sigma*jax.random.normal(kn,(k,dim))
        mm,oo=batch(samples)
        ratio=importance_log_ratio(samples,center,centers,nu,sigma,rr)
        logw=-oo/cfg.temperature+ratio
        finite=jnp.isfinite(logw)
        maximum=jnp.max(jnp.where(finite,logw,-1e30))
        w=jnp.where(finite,jnp.exp(logw-maximum),0.);w=w/jnp.maximum(w.sum(),1e-30)
        avg=jnp.where(jnp.any(finite),w@samples,center)
        score=(-y+jnp.sqrt(abar[i])*avg)/(1-abar[i])
        yn=(y+(1-abar[i])*score)/jnp.sqrt(alpha[i])
        start=len(seeds)+j*k
        theta=jax.lax.dynamic_update_slice(theta,samples,(start,0));means=jax.lax.dynamic_update_slice(means,mm,(start,));opts=jax.lax.dynamic_update_slice(opts,oo,(start,))
        ess=jnp.where(jnp.any(finite),1/jnp.maximum(jnp.sum(w*w),1e-20),0.)
        trace=jnp.array([jnp.min(opts),jnp.min(means),ess,rr,jnp.mean(finite),sigma])
        return (yn,theta,means,opts,key),trace
    (final,theta,means,opts,key),trace=jax.lax.scan(iteration,(y,theta,means,opts,key),jnp.arange(n))
    fm,fo=batch(final[None]);theta=theta.at[-1].set(final);means=means.at[-1].set(fm[0]);opts=opts.at[-1].set(fo[0])
    finalists=theta[jnp.argsort(opts)[:4]]
    finalists=jnp.concatenate((finalists,theta[jnp.argmin(means)][None],previous[None]))
    return finalists,trace,jnp.min(opts),jnp.array(len(seeds)+n*k+1)
