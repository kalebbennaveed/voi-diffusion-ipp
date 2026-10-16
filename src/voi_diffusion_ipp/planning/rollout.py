from typing import NamedTuple
import jax
import jax.numpy as jnp
from .. import clarity
from ..dynamics import step
from ..safety import filter_control


class Rollout(NamedTuple):
    x: jax.Array
    q: jax.Array
    nominal: jax.Array
    controls: jax.Array
    observations: jax.Array
    cost: jax.Array
    correction: jax.Array
    feasible: jax.Array
    clearance: jax.Array
    admissible: jax.Array
    slack: jax.Array


def clarity_controller(x,q,belief,env,cfg):
    """Fixed task-aware continuation approximation; no wind-information reward."""
    need=jnp.maximum(env.targets-q,0)
    attractiveness=need/(.25+jnp.sum((env.cells-x[:2])**2,axis=1))
    goal=env.cells[jnp.argmax(attractiveness)]
    mean=(belief.basis.features(x[:2])@belief.mean)[0]
    return jnp.clip(1.3*(goal-x[:2])-1.8*x[2:]-mean,-cfg.umax,cfg.umax)


def rollout(x0,q0,controls,belief,inner,world,noise,env,cfg,feedback=False):
    """Belief and inner fields remain fixed for this entire horizon."""
    wind=lambda p:world.at(p)[0]
    def body(carry,inputs):
        x,q=carry; u,eps=inputs
        if feedback:
            u=clarity_controller(x,q,belief,env,cfg)
        safe,info=filter_control(x,u,belief,inner,env,cfg)
        xn=step(x,safe,wind,cfg.dt)
        qn=clarity.step(q,x[:2],xn[:2],env.cells,cfg.sensor_sigma,cfg.sensor_R,
                        cfg.clarity_decay,cfg.dt)
        cost=.5*cfg.dt*(clarity.deficit(q,env.targets)+clarity.deficit(qn,env.targets))
        obs=world.at(xn[:2])[0]+belief.noise_std*eps
        # Sample RK4 trajectory at intermediate times for empirical collision diagnostics.
        mids=jax.vmap(lambda t:step(x,safe,wind,t))(cfg.dt*jnp.array([.25,.5,.75,1.]))
        clearance=jnp.min(jax.vmap(env.clearance)(mids))
        return (xn,qn),(xn,qn,u,safe,obs,cost,info.correction,info.feasible,clearance,
                        info.admissible_fraction,info.slack)
    _,out=jax.lax.scan(body,(x0,q0),(controls,noise))
    x,q,u,safe,obs,cost,corr,feasible,clearance,area,slack=out
    return Rollout(jnp.concatenate((x0[None],x)),jnp.concatenate((q0[None],q)),u,safe,
                   obs,jnp.sum(cost),corr,feasible,clearance,area,slack)
