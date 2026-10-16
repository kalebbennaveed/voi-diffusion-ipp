from functools import lru_cache
import jax
import jax.numpy as jnp
from .mbd import optimize
from .two_stage import make_buffers,objective,decode,route
from ..dynamics import step


METHODS=('proposed','passive','weighted_ig','heuristic','cem','mppi','oracle','current_belief','fixed')


@lru_cache(None)
def planner(cfg,method):
    if method not in METHODS:
        raise ValueError(method)
    mode='proposed' if method in ('proposed','cem','mppi') else method
    if method=='oracle':
        mode='current_belief'
    alg=method if method in ('cem','mppi') else 'mbd'
    @jax.jit
    def plan(x,q,belief,key,initial,env):
        ks,ko=jax.random.split(key)
        buffers=make_buffers(belief,ks,cfg)
        fn=lambda y:objective(y,x,q,belief,buffers,env,cfg,mode)
        return optimize(fn,initial,ko,cfg,alg)
    return plan


def initial_policy(x,q,belief,env,cfg):
    """Task-only seed for every solver. No posterior outcomes or truths are used."""
    wind=lambda p:(belief.basis.features(p)@belief.mean)[0]
    def toward(x,goal,n):
        def body(x,_):
            u=jnp.clip(1.2*(goal-x[:2])-1.8*x[2:]-wind(x[:2]),-.92*cfg.umax,.92*cfg.umax)
            return step(x,u,wind,cfg.dt),u
        xf,us=jax.lax.scan(body,x,None,length=n)
        idx=jnp.linspace(0,n-1,cfg.knots)
        knots=jax.vmap(lambda u:jnp.interp(idx,jnp.arange(n),u),in_axes=1,out_axes=1)(us)
        return xf,jnp.arctanh(knots/cfg.umax)
    xL,first=toward(x,env.roi_b,cfg.stage_steps)
    _,a=toward(xL,env.roi_a,cfg.future_steps)
    _,b=toward(xL,env.roi_b,cfg.future_steps)
    futures=jnp.stack((a,b))[:cfg.branches]
    return jnp.concatenate((first[None],futures)).ravel()


def next_initial(previous,x,q,belief,env,cfg):
    # Reuse the observation-selected future as a warm first period. Remaining
    # branches are task-only seeds; the whole vector is optimized again.
    base=initial_policy(x,q,belief,env,cfg).reshape((1+cfg.branches,cfg.knots,2))
    prior=previous.reshape(base.shape)
    b=route(x,q,belief,env,cfg)
    # First period only covers the first TL/TF of previous future.
    times=jnp.linspace(0,min(1,cfg.stage_steps/cfg.future_steps),cfg.knots)
    knots=jax.vmap(lambda u:jnp.interp(times,jnp.linspace(0,1,cfg.knots),u),in_axes=1,out_axes=1)(prior[b+1])
    return base.at[0].set(knots).ravel()
