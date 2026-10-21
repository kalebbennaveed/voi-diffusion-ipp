import json,time
import numpy as np
import jax,jax.numpy as jnp
from dataclasses import replace
from voi_diffusion_ipp.quarter.environment import Settings,environment,initial_belief,table_sampler
from voi_diffusion_ipp.quarter.planner import library,sensing_points,seeds,rollout,decode
from voi_diffusion_ipp.quarter.diffusion import lower_tail
p='output/european_quarter/pilot/two_forward_proposed_8101.json';d=json.load(open(p));t=72.;f=min(d['frames'],key=lambda f:abs(f['t']-t));t=f['t'];cfg=Settings(**d['config']);env=environment();b=initial_belief(cfg);condition=jax.jit(lambda b,p,y:b.condition(p,y))
for o in d['observations']:
 if o['t']<=t+.001:b=condition(b,jnp.array([o['position']],jnp.float32),jnp.array([o['measurement']],jnp.float32))
x=jnp.array(f['p'][:2]+f['velocity'][:2],jnp.float32);q=jnp.array(f['q']);previous=jnp.array(min(d['plans'],key=lambda v:abs(v['t']-t))['policy']);held=np.zeros((2,4));sites=sensing_points(env)
for ff in d['frames']:
 if ff['t']<=t:held=np.where(np.linalg.norm(sites-np.array(ff['p'][:2]),axis=2)<.4,held+.1,0.)
paths=library(np.asarray(x[:2]),env,held,f['inspection_phase']);mean,fields=table_sampler(10)(b,jax.random.PRNGKey(851));bank=fields[2:];construction=fields[:2];th=seeds(2).at[:,9:].set(previous[9:]);th=jnp.r_[th,previous[None]]
rows=[]
for slack,horizon in [(15.,60.),(1.,60.),(.1,60.),(15.,120.)]:
 c=replace(cfg,slack_weight=slack,horizon=horizon)
 @jax.jit
 def evaluate(th):
  def one(theta):
   def world(field):
    r=rollout(theta,x,q,mean,construction,field,paths,env,c);return jnp.array([r.monitor+r.penalty,r.max_slack,r.min_clearance])
   return jax.vmap(world)(bank)
  return jax.lax.map(one,th,batch_size=8)
 start=time.perf_counter();v=np.asarray(evaluate(th));opt=np.asarray(lower_tail(jnp.array(v[:,:,0]),.5));opt=np.where(np.isfinite(opt),opt,np.inf);idx=np.argsort(opt)[:5]
 row=dict(time=t,slack_weight=slack,horizon=horizon,seconds=time.perf_counter()-start,policies=[dict(index=int(i),cost=float(opt[i]),mean=float(v[i,:,0].mean()),max_slack=float(v[i,:,1].max()),clearance=float(v[i,:,2].min())) for i in idx]);rows.append(row);print(row,flush=True)
open('output/european_quarter/choice_diagnostic.json','w').write(json.dumps(rows,indent=2))
