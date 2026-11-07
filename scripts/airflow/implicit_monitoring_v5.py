"""Reviewed unknown-wind planner: supported planning mean and exact POAM identity transport.

Full POAM posterior, covariance and confidence bounds remain untouched in safety.
Only the declared performance distribution tends to zero away from information.
"""
from implicit_monitoring_v4 import *
import implicit_monitoring_v4 as predecessor
from supported_poam_identity import condition_supported


def planning_fields(mean,draws,variance,prior_variance,count):
    r=information_support(variance,prior_variance,count)
    fields=r[None,:,None]*mean[None]+jnp.sqrt(r)[None,:,None]*(draws-mean[None])
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
        cv=coefficients(jax.random.fold_in(performance_key,104729),cfg.validation_worlds)
        mu,cov=b.predict(points)
        variance=cov[:,0,0]
        prior=jnp.exp(b.basis.parameters['log_amplitude'])
        draws=jax.vmap(lambda c:b.basis.evaluate(points,c))(cp)
        perf,r=planning_fields(mu,draws,variance,prior,count)
        safe=jax.vmap(lambda c:b.basis.evaluate(points,c))(cs)
        vd=jax.vmap(lambda c:b.basis.evaluate(points,c))(cv)
        validation,_=planning_fields(mu,vd,variance,prior,count)
        return Tables(mu.reshape(shape),jnp.sqrt(variance).reshape(*shape[:2],1),
                      r.reshape(*shape[:2],1),perf.reshape(cfg.worlds,*shape),
                      safe.reshape(cfg.construction_worlds,*shape),validation.reshape(cfg.validation_worlds,*shape))
    return build

