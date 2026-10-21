"""Fixed-state optimizer sensitivity; separate from closed-loop mission evidence."""
import json,time
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import jax,jax.numpy as jnp
from voi_diffusion_ipp.quarter.environment import Settings,environment,initial_belief,table_sampler
from voi_diffusion_ipp.quarter.planner import library,optimizer,sensing_points,rollout
from voi_diffusion_ipp import clarity

source=Path('output/european_quarter/pilot/two_forward_blind_8101.json')
data=json.loads(source.read_text());env=environment('two');base=Settings(worlds=4,validation_worlds=6)
b=initial_belief(base);condition=jax.jit(lambda bb,p,y:bb.condition(p,y))
for o in [o for o in data['observations'] if o['t']<=40.001]:b=condition(b,jnp.array([o['position']],jnp.float32),jnp.array([o['measurement']],jnp.float32))
f=min(data['frames'],key=lambda f:abs(f['t']-40.));x=jnp.array(f['p'][:2]+f['velocity'][:2],jnp.float32);q=jnp.array(f['q'],jnp.float32);previous=jnp.array(min(data['plans'],key=lambda p:abs(p['t']-40.))['policy'])
residence=np.zeros((2,4));sites=sensing_points(env)
for f0 in [f0 for f0 in data['frames'] if f0['t']<=40.001]:residence=np.where(np.linalg.norm(sites-np.array(f0['p'][:2]),axis=2)<.4,residence+.1,0)
paths=library(np.asarray(x[:2]),env,residence,np.array(f["inspection_phase"]))
mean,bank=table_sampler(24)(b,jax.random.PRNGKey(949));construction=bank[:2];training=bank[2:6];validation=bank[6:12];holdout=bank[12:]
rows=[];dest=Path('output/european_quarter/parameter_screen.json')
for horizon,n,temp in [(30,4,.15),(60,4,.15),(90,4,.15),(60,8,.15),(60,12,.15),(60,8,.05),(60,8,.4)]:
    cfg=Settings(horizon=horizon,iterations=n,proposals=8,worlds=4,validation_worlds=6,temperature=temp)
    plan=optimizer(env,cfg,'proposed');start=time.perf_counter();theta,opt,trace,expected,count=plan(x,q,mean,construction,training,validation,paths,previous,jax.random.PRNGKey(37),jnp.array(.5));theta.block_until_ready();seconds=time.perf_counter()-start
    @jax.jit
    def evaluate(th):
        def one(field):
            r=rollout(th,x,q,mean,construction,field,paths,env,replace(cfg,horizon=90.))
            valid=jnp.all(jnp.isfinite(r.x),axis=1)&(jax.vmap(env.clearance)(r.x)>=.275)&jnp.all(jnp.isfinite(r.q),axis=1)
            failed=~jnp.all(valid);first=jnp.where(failed,jnp.argmax(~valid),len(valid))
            before=r.q[jnp.maximum(jnp.minimum(first-1,len(valid)-1),0)]
            elapsed=jnp.maximum(jnp.arange(len(valid))-first+1,0)*cfg.dt
            no_sensing=before[None]/(1+env.process_noise[None]*before[None]*elapsed[:,None])
            qs=jnp.where((jnp.arange(len(valid))<first)[:,None],r.q,no_sensing)
            deficits=jax.vmap(lambda z:clarity.deficit(z,env.targets))(qs)
            return cfg.dt*(deficits.sum()-.5*(deficits[0]+deficits[-1])),failed
        return jax.vmap(one)(holdout)
    values,failures=evaluate(theta);costs=np.asarray(values);assert np.isfinite(costs).all();rows.append(dict(horizon=horizon,iterations=n,temperature=temp,training_best=float(trace[-1,0]),selection_validation_optimistic=float(opt),selection_validation_mean=float(expected),holdout_failed_fields=int(np.asarray(failures).sum()),holdout_mean=float(costs.mean()),holdout_std=float(costs.std(ddof=1)),holdout_costs=costs.tolist(),evaluations=int(count),compile_and_plan_seconds=seconds,theta=np.asarray(theta).tolist(),trace=np.asarray(trace).tolist()))
    dest.write_text(json.dumps(dict(state_source=str(source),time=f['t'],description='Fixed state and posterior. Shared training/selection fields; independent holdout fields. Holdout evaluates every selected policy over the same 90 s, with no-sensing mission accounting after any predicted contact or numerical loss. Prediction only, not nonlinear closed-loop evidence.',results=rows),indent=2,allow_nan=False))
    print({k:v for k,v in rows[-1].items() if k not in ('theta','trace','holdout_costs')},flush=True)
    jax.clear_caches()
