"""V3 physics and POAM; bounded two-finalist validation plus explicit fallback."""
from implicit_monitoring_v3 import *
import implicit_monitoring_v3 as previous

def optimizer(env,cfg,basis,aware=True):
    # Compile numerical stages separately: avoid cloning the full physics scan
    # into conditional branches and skip exact duplicate finalists on the host.
    import time
    @jax.jit
    def optimize(x,q,tables,seeds,previous,key):
        def evaluate(theta):
            r=jax.vmap(lambda w:rollout(theta,x,q,tables,w,basis,env,cfg,aware))(tables.planning)
            return jnp.where(r.terminated,jnp.inf,r.monitor+r.penalty)
        return search(evaluate,seeds,previous,key,cfg,jnp.array(1.),jnp.array(0.),jnp.array(False))
    @jax.jit
    def validate(theta,x,q,tables):
        return jax.vmap(lambda w:execution_forecast(theta,x,q,tables,w,basis,env,cfg,aware))(tables.validation)
    def plan(x,q,tables,seeds,previous,key):
        begin=time.perf_counter()
        result=optimize(x,q,tables,seeds,previous,key);jax.block_until_ready(result)
        search_s=time.perf_counter()-begin
        different=np.any(np.asarray(result.initial_parameters)!=np.asarray(result.final)[None],axis=1)
        seedscore=np.asarray(result.initial_field_costs).mean(axis=1)
        second=int(np.argmin(np.where(different,seedscore,np.inf)))
        candidates=jnp.stack((result.final,result.initial_parameters[second]))
        seen={};forecasts=[];begin=time.perf_counter()
        for theta in candidates:
            token=np.asarray(theta).tobytes()
            if token not in seen:
                seen[token]=validate(theta,x,q,tables);jax.block_until_ready(seen[token])
            forecasts.append(seen[token])
        validation_s=time.perf_counter()-begin
        forecasts=jax.tree.map(lambda *z:jnp.stack(z),*forecasts)
        score=jnp.mean(forecasts.monitor+forecasts.penalty,axis=1)
        feasible=jnp.all(~forecasts.terminated & (forecasts.max_slack<1e-4),axis=1)
        if not bool(jnp.any(feasible)):
            # Explicit fallback, evaluated and logged, never presented as safe
            # if its hard QP or physical forecast also fails.
            token=np.asarray(seeds[0]).tobytes();begin=time.perf_counter()
            hold=seen.get(token)
            if hold is None:hold=validate(seeds[0],x,q,tables);jax.block_until_ready(hold)
            validation_s+=time.perf_counter()-begin
            candidates=jnp.concatenate((candidates,seeds[:1]))
            forecasts=jax.tree.map(lambda a,b:jnp.concatenate((a,b[None])),forecasts,hold)
            score=jnp.mean(forecasts.monitor+forecasts.penalty,axis=1)
            feasible=jnp.all(~forecasts.terminated & (forecasts.max_slack<1e-4),axis=1)
            choice=2
        else:
            choice=int(jnp.argmin(jnp.where(feasible,score,jnp.inf)))
        selected=jax.tree.map(lambda z:z[choice],forecasts)
        audit=dict(candidates=candidates,costs=score,feasible=feasible,
                   contacts=forecasts.terminated,slack=forecasts.max_slack,
                   selected=choice,no_feasible=~jnp.any(feasible),
                   unique_evaluations=len(seen),search_s=search_s,validation_s=validation_s)
        return candidates[choice],score[choice],result,selected,audit
    return plan

