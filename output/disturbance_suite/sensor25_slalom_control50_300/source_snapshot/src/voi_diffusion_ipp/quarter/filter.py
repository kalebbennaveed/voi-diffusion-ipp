"""State-dependent policy-value filter with penalized constraint slack.

The nominal double integrator and quadratic air drag are unchanged. This is a
finite-sample soft constraint construction; it is not a safety guarantee.
"""
import jax
import jax.numpy as jnp
from ..safety.flow_policy_value import values,design_policy
from ..quadrotor.aerodynamics import acceleration


def newton_direction(A, active, gradient, weight):
    """Solve (I + weight A_active.T A_active) v = gradient in two dimensions.

    The positive Gram determinant avoids subtracting nearly equal diagonal
    products. Scalar LDL elimination lets XLA fuse this tiny solve instead of
    making a batched LAPACK call at every rollout/filter iteration.
    """
    a = A[:, 0] * active
    b = A[:, 1] * active
    aa, bb, ab = jnp.sum(a*a), jnp.sum(b*b), jnp.sum(a*b)
    i, j = jnp.triu_indices(A.shape[0], 1)
    cross = a[i]*b[j] - a[j]*b[i]
    determinant = 1 + weight*(aa+bb) + weight*weight*jnp.sum(cross*cross)
    diagonal = 1 + weight*aa
    lower = weight*ab/diagonal
    second = (gradient[1] - lower*gradient[0])/(determinant/diagonal)
    first = gradient[0]/diagonal - lower*second
    return jnp.stack((first, second))


def soft_projection(nominal,A,b,limit,weight):
    """Box-constrained, squared positive-residual penalty; 2D semismooth solve."""
    u=jnp.clip(nominal,-limit,limit)
    objective=lambda v:.5*jnp.sum((v-nominal)**2)+.5*weight*jnp.sum(jax.nn.relu(A@v-b)**2)
    def improve(_,u):
        residual=A@u-b;active=residual>0
        grad=u-nominal+weight*(A.T@jax.nn.relu(residual))
        # A projected-gradient step is a robust alternative at an active box face.
        newton=newton_direction(A,active,grad,weight)
        lipschitz=1+weight*jnp.sum(A*A)
        directions=jnp.stack((newton,grad/lipschitz))
        scales=2.**(-jnp.arange(7))
        trials=jnp.clip(u[None,None]-directions[:,None,:]*scales[None,:,None],-limit,limit).reshape(-1,2)
        trials=jnp.r_[u[None],trials]
        return trials[jnp.argmin(jax.vmap(objective)(trials))]
    u=jax.lax.fori_loop(0,8,improve,u)
    return u,jax.nn.relu(A@u-b)


def control(x,nominal,mean,fields,env,cfg):
    def f(z):
        value=values(z,mean,fields,env,cfg)
        return value,value
    grad,B=jax.jacfwd(f,has_aux=True)(x)
    winds=fields.at(x[:2])[:,0]
    drift=acceleration(x[2:],winds,xp=jnp)
    bound=-cfg.cbf_alpha*B-grad[:,:2]@x[2:]-jnp.max(grad[:,2:]@drift.T,axis=1)
    u,slack=soft_projection(nominal,grad[:,2:],bound,cfg.umax,cfg.slack_weight)
    return u,jnp.max(slack)
