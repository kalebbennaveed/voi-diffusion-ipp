"""Original POAM pivoted-Cholesky inducing selection for executed learning.

The previous observed-slot allocator is not used. Preserve the full transported
statistics, attentive kernel, residual covariance, noise, and prior amplitude.
"""
from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from voi_diffusion_ipp.wind.poam import (select_inducing_jax,attention_weights,
    chol_solve,transport_statistic_root,variational_from_root,negative_elbo,attentive_kernel,
    spatial_observations)


@partial(jax.jit, static_argnames=("observation_first",))
def condition(b,positions,measurements,mask,observation_first=False):
    positions,measurements=spatial_observations(positions,measurements)
    # The attentive kernel has a constant diagonal: first-pivot candidates
    # are exact ties. Put actual new observations first when requested, so
    # an arbitrary initial-grid ordering cannot permanently exclude the
    # measured location. The greedy criterion and POAM transport are unchanged.
    candidates=jnp.concatenate((positions,b.basis.centers) if observation_first else (b.basis.centers,positions))
    centers=select_inducing_jax(b.basis.parameters,candidates,b.basis.lengths,len(b.mean))
    K=b.kernel(centers,centers)
    # Selecting only old locations can permute them. Apply exact permutation
    # transport instead of introducing repeated numerical regularization.
    equal=jnp.all(b.basis.centers[:,None,:]==centers[None,:,:],axis=-1)
    same_set=jnp.all(jnp.any(equal,axis=0))
    P=jax.lax.cond(same_set,lambda _:equal.astype(K.dtype),
        lambda _:chol_solve(b.old_K,b.kernel(b.basis.centers,centers)),None)
    Kuf=b.kernel(centers,positions)
    a=P.T@b.a+Kuf@measurements;root=transport_statistic_root(b.root_B,P,Kuf)
    mean,cov=variational_from_root(K,a,root,b.noise_std)
    basis=b.basis._replace(centers=centers,center_weights=attention_weights(b.basis.parameters,centers))
    supported=(equal.T@mask.astype(jnp.int32)>0)|jnp.any(jnp.linalg.norm(centers[:,None]-positions[None],axis=-1)<=.15,axis=1)
    return b._replace(basis=basis,mean=mean,covariance=(cov+cov.T)/2,a=a,B=root@root.T,root_B=root,
                      old_K=K,inverse=chol_solve(K,jnp.eye(len(K)))),supported


@jax.jit
def train(b,bx,by,data_count,m,v,step,amplitude_rate=.001):
    """Existing POAM variational EM objective; four Adam steps on actual data.

    Follow the M step with an observation-free transported E step so all saved
    prediction and sampling caches correspond to the updated kernel.
    """
    def update(carry,batch):
        params,m,v,step=carry
        batch_x,batch_y=batch
        loss,g=jax.value_and_grad(negative_elbo)(params,batch_x,batch_y,b.basis.centers,
            b.mean,b.covariance,b.basis.lengths,b.noise_std,data_count)
        step=step+1;m=jax.tree.map(lambda a,g:.9*a+.1*g,m,g)
        v=jax.tree.map(lambda a,g:.999*a+.001*g*g,v,g)
        params={name:p-(amplitude_rate if name=='log_amplitude' else .001)*(m[name]/(1-.9**step))/(jnp.sqrt(v[name]/(1-.999**step))+1e-8)
                for name,p in params.items()}
        return (params,m,v,step),loss
    # The reference draws an independent history minibatch for every Adam step.
    # Keep the old 2D-batch behavior for reproducibility of earlier experiments.
    batches=(jnp.broadcast_to(bx,(4,*bx.shape)),jnp.broadcast_to(by,(4,*by.shape))) if bx.ndim==2 else (bx,by)
    if batches[0].shape[0]!=4 or batches[0].shape!=batches[1].shape:
        raise ValueError('POAM EM needs four matching spatial minibatches')
    (params,m,v,step),loss=jax.lax.scan(update,(b.basis.parameters,m,v,step),batches)
    z=b.basis.centers;kernel=lambda x,y:attentive_kernel(params,x,y,b.basis.lengths)
    K=kernel(z,z);P=chol_solve(b.old_K,K);a=P.T@b.a
    root=transport_statistic_root(b.root_B,P,jnp.zeros((len(z),0)))
    mean,cov=variational_from_root(K,a,root,b.noise_std)
    support=b.basis.support;Ks=kernel(support,support)
    L=jnp.linalg.cholesky((Ks+Ks.T)/2+1e-4*jnp.eye(len(support)))
    inverse=(jnp.zeros((0,0),dtype=L.dtype) if b.basis.support_inverse.size==0
             else jsp.linalg.cho_solve((L,True),jnp.eye(len(support))))
    basis=b.basis._replace(parameters=params,center_weights=attention_weights(params,z),
        support_weights=attention_weights(params,support),support_inverse=inverse,support_cholesky=L)
    return b._replace(basis=basis,mean=mean,covariance=(cov+cov.T)/2,a=a,B=root@root.T,root_B=root,
        old_K=K,inverse=chol_solve(K,jnp.eye(len(K)))),m,v,step,loss
