"""Existing spline policy-value construction for a planar model in air flow.

The nominal plant is a double integrator. Known air drag is evaluated under
belief-derived spatial wind-velocity samples, including velocity dependence.
No physical-field input is accepted by the controller.
"""
from functools import partial
import jax
import jax.numpy as jnp
from ..wind import InnerFields
from ..quadrotor.aerodynamics import acceleration
from .rpcbf import FilterInfo
from .spline import maximum
from .qp import solve


def design_policy(x, belief, cfg):
    drift=acceleration(x[2:],belief.mean_at(x[:2])[0],xp=jnp)
    return jnp.clip(-cfg.backup_gain*x[2:]-drift,-cfg.umax,cfg.umax)


def values(x,belief,inner,env,cfg):
    if not isinstance(inner,InnerFields):
        raise TypeError('Policy-value prediction requires belief-derived inner fields')
    def one(coeff):
        def tick(z,_):
            u=design_policy(z,belief,cfg)
            def f(zz):
                w=inner.basis.evaluate(zz[:2],coeff)[0]
                return jnp.r_[zz[2:],u+acceleration(zz[2:],w,xp=jnp)]
            h=cfg.backup_dt
            a=f(z);b=f(z+h*a/2);c=f(z+h*b/2);d=f(z+h*c)
            zn=z+h*(a+2*b+2*c+d)/6
            return zn,env.violations(zn,cfg.safety_margin)
        _,h=jax.lax.scan(tick,x,None,length=cfg.backup_steps)
        h=jnp.concatenate((env.violations(x,cfg.safety_margin)[None],h))
        return jax.vmap(partial(maximum,mode=cfg.spline),in_axes=1)(h)
    return jnp.max(jax.vmap(one)(inner.coefficients),axis=0)


def filter_control(x,u_nom,belief,inner,env,cfg):
    def f(z):
        value=values(z,belief,inner,env,cfg)
        return value,value
    grad,B=jax.jacfwd(f,has_aux=True)(x)
    winds=inner.at(x[:2])[:,0]
    drift=acceleration(x[2:],winds,xp=jnp)
    worst=grad[:,:2]@x[2:]+jnp.max(grad[:,2:]@drift.T,axis=1)
    A=grad[:,2:];bound=-worst-cfg.cbf_alpha*B
    sol=solve(u_nom,A,bound,cfg.umax,design_policy(x,belief,cfg))
    return sol.control,FilterInfo(B,grad,jnp.linalg.norm(sol.control-u_nom),sol.feasible,
                                 sol.slack,sol.residual,jnp.array(0.))
