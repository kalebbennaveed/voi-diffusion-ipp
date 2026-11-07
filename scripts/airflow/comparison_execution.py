"""Explicit unfiltered ablation, sharing the original plant and clarity integrator."""
import jax
import jax.numpy as jnp
import full_learning_monitoring as model


def unfiltered_execution(env,cfg,pcfg,basis,seconds=.2):
    @jax.jit
    def execute(x,q,refs,aa,offset,tables,hidden_physics_field,alive):
        still=model.TableMean(basis,jnp.zeros_like(tables.mean))
        def tick(carry,k):
            x,q,alive=carry
            local=offset+k*pcfg.execution_dt
            i=jnp.minimum(jnp.floor((local+1e-8)/cfg.dt).astype(int),cfg.steps-1)
            off=local-i*cfg.dt;ref,a=refs[i],aa[i]
            reference=jnp.r_[ref[:2]+off*ref[2:]+.5*off**2*a,ref[2:]+off*a]
            nominal=model.m.tracking_control(x,reference,a,still,cfg,True)
            u=jnp.clip(nominal,-cfg.umax,cfg.umax)
            xn,qn,living,cost,gap,hit=model.m.advance(x,q,u,hidden_physics_field,env,pcfg,alive,interval=pcfg.execution_dt)
            return (xn,qn,living),(xn,qn,jnp.where(alive,u,0),nominal,jnp.array(0.),jnp.array(0.),jnp.array(True),gap,cost,living,jnp.where(alive,hit,jnp.inf))
        return jax.lax.scan(tick,(x,q,alive),jnp.arange(round(seconds/pcfg.execution_dt)))
    return execute
