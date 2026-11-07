"""Exact reuse of kernel/features on fixed query points, without belief ageing.

Only deterministic basis evaluations are cached. Every actual posterior update
recomputes the mean, full marginal variance and coherent draw coefficients.
The cache is invalidated by any kernel parameter or inducing-location change.
"""
import jax
import jax.numpy as jnp
import numpy as np
from rmse_monitoring import Tables
from exact_poam_variance import PosteriorStandardDeviation
from voi_diffusion_ipp.wind.poam import attention_weights

@jax.jit
def prepare_moments(b):
    return b.inverse@b.mean, b.inverse@b.covariance@b.inverse.T-b.inverse, jnp.exp(b.basis.parameters['log_amplitude'])

@jax.jit
def moments(K,alpha,correction,amplitude):
    mean=K@alpha
    variance=jnp.maximum(amplitude+jnp.sum((K@correction)*K,axis=1),0.)
    return mean,variance

@jax.jit
def draws(b,phi,Ks,Kz,key,n_eps):
    eps=jax.random.normal(key,n_eps.shape,dtype=b.mean.dtype)
    coefficients=b.sample_coefficients(jnp.concatenate((eps,-eps)))
    f=phi.shape[1];m=len(b.mean)
    return (jnp.einsum('pf,nfc->npc',phi,coefficients[:,:f])+
            jnp.einsum('ps,nsc->npc',Ks,coefficients[:,f:-m])+
            jnp.einsum('pm,nmc->npc',Kz,coefficients[:,-m:]))

@jax.jit
def prior_draw(b,phi,Ks,key,n_eps):
    """Identical random draw's prior part; independent of observed values/z."""
    eps=jax.random.normal(key,n_eps.shape,dtype=b.mean.dtype)
    eps=jnp.concatenate((eps,-eps));m=len(b.mean);ns=len(b.basis.support)
    nf=b.sample_shape[0]-m-ns
    prior_eps,inducing_eps=eps[:,:nf],eps[:,-m:]
    support_eps=eps[:,nf:nf+ns]
    if ns:
        gs=jnp.einsum('sf,nfc->nsc',b.basis.spectral_features(b.basis.support),prior_eps)
        vs=jnp.einsum('st,ntc->nsc',b.basis.support_cholesky,support_eps)
        support_coeff=b.basis.support_correction(vs-gs)
    else:support_coeff=support_eps
    field=(jnp.einsum('pf,nfc->npc',phi,prior_eps)+
           jnp.einsum('ps,nsc->npc',Ks,support_coeff))
    return prior_eps,support_coeff,inducing_eps,field

@jax.jit
def condition_prior_draw(b,Kz,prior):
    prior_eps,support_coeff,inducing_eps,field=prior
    gz=(jnp.einsum('mf,nfc->nmc',b.basis.spectral_features(b.basis.centers),prior_eps)+
        jnp.einsum('ms,nsc->nmc',b.basis.kernel_support(b.basis.centers),support_coeff))
    L=jnp.linalg.cholesky(b.covariance+1e-7*jnp.eye(len(b.mean)))
    u=b.mean[None]+jnp.einsum('ij,njc->nic',L,inducing_eps)
    correction=jnp.einsum('ij,njc->nic',b.inverse,u-gz)
    return field+jnp.einsum('pm,nmc->npc',Kz,correction)

class CachedFields:
    def __init__(self,points,shape,cfg,map_points,exact_variance=False):
        self.exact_variance=exact_variance
        self.points=points;self.shape=shape;self.cfg=cfg;self.map_points=jnp.asarray(map_points)
        self.parameters=None;self.centers=None;self.geometry=None;self.rebuilds=0;self.posterior_b=None;self.map_b=None
        self.safety_prior=None;self.safety_prior_key=None
        self._features=jax.jit(lambda b:(b.spectral_features(points),b.kernel_support(points)))
        self._centers=jax.jit(lambda b:(b.kernel_centers(points),b.kernel_centers(self.map_points)))
    def refresh(self,b):
        params=[np.asarray(v) for v in jax.tree.leaves(b.basis.parameters)]
        geometry=[np.asarray(b.basis.lengths),np.asarray(b.basis.support),np.asarray(b.basis.frequencies)]
        changed=self.parameters is None or any(not np.array_equal(x,y) for x,y in zip(params,self.parameters))
        changed=changed or self.geometry is None or any(not np.array_equal(x,y) for x,y in zip(geometry,self.geometry))
        centers=np.asarray(b.basis.centers)
        if changed:
            self.phi,self.Ks=self._features(b.basis);self.parameters=[p.copy() for p in params];self.geometry=[p.copy() for p in geometry];self.rebuilds+=1
        if changed or self.centers is None:
            self.Kz,self.Kmap=self._centers(b.basis)
        elif not np.array_equal(centers,self.centers):
            # Pivoted Cholesky mostly permutes existing inducing locations.
            # Reuse their exact columns and evaluate only genuinely new points.
            lookup={tuple(p):i for i,p in enumerate(self.centers)}
            new=[i for i,p in enumerate(centers) if tuple(p) not in lookup]
            kz,km=self.Kz,self.Kmap
            if new:
                added=b.basis.centers[jnp.asarray(new)]
                reduced=b.basis._replace(centers=added,center_weights=attention_weights(b.basis.parameters,added))
                az,am=self._centers(reduced);kz=jnp.concatenate((kz,az),axis=1);km=jnp.concatenate((km,am),axis=1)
                lookup.update({tuple(centers[i]):len(self.centers)+k for k,i in enumerate(new)})
            order=jnp.asarray([lookup[tuple(p)] for p in centers]);self.Kz=kz[:,order];self.Kmap=km[:,order]
        self.centers=centers.copy()
        if self.posterior_b is not b:
            self.posterior=prepare_moments(b);self.posterior_b=b
    def map(self,b):
        self.refresh(b)
        # Multiple diagnostics/acquisition calls can request the same immutable
        # posterior. Reuse exactly that result; every actual update changes b.
        if self.map_b is not b:
            mean,var=moments(self.Kmap,*self.posterior)
            self.map_result=(mean,var[:,None,None]*jnp.eye(2)[None]);self.map_b=b
        return self.map_result
    def draw(self,b,key,n):
        return draws(b,self.phi,self.Ks,self.Kz,key,jnp.empty((n//2,*b.sample_shape))).reshape(n,*self.shape)
    def safety_draw(self,b,key):
        # Store only the fixed safety bank, not each new planning draw. Kernel
        # updates invalidate this prior; inducing moves and observations still
        # recompute the complete Matheron posterior correction on every call.
        n=self.cfg.construction_worlds
        identity=(self.rebuilds,tuple(np.asarray(key).ravel()),n,b.sample_shape)
        if identity!=self.safety_prior_key:
            self.safety_prior=prior_draw(b,self.phi,self.Ks,key,jnp.empty((n//2,*b.sample_shape)))
            self.safety_prior_key=identity
        return condition_prior_draw(b,self.Kz,self.safety_prior).reshape(n,*self.shape)
    def standard_deviation(self,b,var):
        grid=jnp.sqrt(var).reshape(*self.shape[:2],1)
        if not self.exact_variance:return grid
        return PosteriorStandardDeviation(grid,b.basis.parameters,b.basis.centers,b.basis.lengths,self.posterior[1],self.posterior[2])
    def build(self,b,count,performance_key,safety_key):
        self.refresh(b);mean,var=moments(self.Kz,*self.posterior)
        return Tables(mean.reshape(self.shape),self.standard_deviation(b,var),
            jnp.ones((*self.shape[:2],1)),self.draw(b,performance_key,self.cfg.worlds),
            self.safety_draw(b,safety_key),
            self.draw(b,jax.random.fold_in(performance_key,104729),self.cfg.validation_worlds))
    def planning(self,b,old,performance_key):
        """Execution already refreshed mean, variance and safety for this belief.

        Only the two new performance banks change at a planning call. The
        runner must call safety() after the most recent actual update first.
        """
        self.refresh(b)
        return old._replace(planning=self.draw(b,performance_key,self.cfg.worlds),
            validation=self.draw(b,jax.random.fold_in(performance_key,104729),self.cfg.validation_worlds))
    def safety(self,b,old,key):
        self.refresh(b);mean,var=moments(self.Kz,*self.posterior)
        return old._replace(mean=mean.reshape(self.shape),std=self.standard_deviation(b,var),
            safety=self.safety_draw(b,key))
