"""RPCBF with explicit robust obstacle and wall braking constraints.

The additional inequalities use the exact distance Hessian and the existing
nonlinear wind-force support bound. They do not alter physics, remove RPCBF,
shrink uncertainty, or use true wind. Validity remains conditional on the
wind envelope, feasible controls and appropriate initial barrier conditions.
"""
from rmse_monitoring import *
import rmse_monitoring as original


def obstacle_constraints(x,mean,std_table,basis,env,cfg):
    delta=x[:2]-env.obstacles[:,:2]
    distance=jnp.maximum(jnp.linalg.norm(delta,axis=1),1e-9)
    normal=delta/distance[:,None]
    h=distance-env.obstacles[:,2]-cfg.safety_margin
    normal_speed=normal@x[2:]
    curvature=(jnp.dot(x[2:],x[2:])-normal_speed**2)/distance
    sigma=jnp.maximum(basis.evaluate(x[:2],std_table)[0,0],0.)
    radius=getattr(cfg,'safety_confidence',3.)*jnp.sqrt(sigma**2+.1**2);rate=1.5
    A=-normal
    b=curvature+2*rate*normal_speed+rate**2*h-core.force_support(A,x[2:],mean.mean_at(x[:2])[0],radius)
    return A,b


def control(x,nominal,mean,fields,std_table,basis,env,cfg,hard=False):
    grad,B,bound=core.constraints(x,mean,fields,std_table,basis,env,cfg,kappa=getattr(cfg,'safety_confidence',3.))
    wall_A,wall_b=boundary_constraints(x,mean,std_table,basis,env,cfg)
    obstacle_A,obstacle_b=obstacle_constraints(x,mean,std_table,basis,env,cfg)
    A=jnp.concatenate((grad[:,2:],wall_A,obstacle_A));b=jnp.concatenate((bound,wall_b,obstacle_b))
    if hard:
        sol=solve(nominal,A,b,cfg.umax,design_policy(x,mean,cfg))
        # The polytope solver accepts a 2e-5 numerical feasibility tolerance.
        # Enforce the physical actuator box exactly at the command boundary.
        u=jnp.clip(sol.control,-cfg.umax,cfg.umax);residual=jnp.max(A@u-b)
        feasible=sol.feasible&(residual<=2e-5)
        slack=jnp.where(feasible,0.,jnp.maximum(sol.slack,residual))
        return u,FilterInfo(B,grad,jnp.linalg.norm(u-nominal),feasible,slack,residual,jnp.array(0.))
    u,slack=soft_projection(nominal,A,b,cfg.umax,cfg.slack_weight)
    return u,jnp.max(slack)


def execution_block(env,cfg,pcfg,basis,seconds=.2):
    return core.execution_block(env,cfg,pcfg,basis,seconds,control_fn=control)


def optimizer(env,cfg,basis,aware=True):
    return original.optimizer(env,cfg,basis,aware,control_fn=control)
