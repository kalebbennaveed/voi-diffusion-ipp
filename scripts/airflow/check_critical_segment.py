"""Counterfactual safety stress test from a recorded, pre-failure state.

These are diagnostic continuations, not replacements for complete missions.
All POAM updates after t=333 use actual measurements at the new executed path.
"""
import argparse,json,gzip,pickle,signal,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import full_learning_monitoring as m
import efficient_learning_monitoring as efficient
import cluster_environment as ce
from cached_poam_fields import CachedFields
p=argparse.ArgumentParser();p.add_argument('--profiles',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=20);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('300s critical-segment diagnostic budget')));signal.alarm(300)
r=json.load(gzip.open('output/matched_duration_20261105/baseline-pilot-failed/s1-center-baseline1.json.gz'))
cfg=m.settings();pcfg=m.legacy.physical_settings(cfg);env,grid,spec=ce.environment(pcfg,m.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json');basis,points,shape=m.table_grid(env.bounds,.1)
truth=m.m.physical.load_wind(1,m.m.physical.SITE/'revised');seed=next(c['seed'] for c in ce.CASES if c['id']=='center')
with jax.experimental.disable_x64():planner=efficient.optimizer(m.legacy.float32(env),cfg,m.legacy.float32(basis),True,prepare_validation=True)
execute=m.execution_block(env,cfg,pcfg,basis,seconds=.2)
report=[]
for source in sorted(a.profiles.glob('length-*-t333.pkl')):
 start=time.perf_counter()
 with source.open('rb') as f:gp=pickle.load(f)
 b=gp.belief;cache=CachedFields(points,shape,cfg,grid['all_cells'])
 query=jnp.asarray(np.vstack((np.asarray(env.cells)[::30],[[4.8,3.3],[4.7,3.2],[4.9,3.4],[5.,3.4]])))
 _,exact=jax.jit(lambda b,p:b.joint(p))(b,query);_,sampled=jax.jit(lambda b,p:b.sample_moments(p))(b,query);jax.block_until_ready(sampled)
 relative=float(np.linalg.norm(sampled-exact)/np.linalg.norm(exact))
 x=jnp.asarray(next(s[1:5] for s in r['samples'] if s[0]==333));q=jnp.asarray(next(s['q'] for s in r['clarity_frames'] if s['t']==333));alive=jnp.array(True)
 noise=np.random.default_rng(seed).normal(size=(2001,2));previous=None;rowsout=[];plans=[];contact=None;minimum=float('inf');infeasible=0;cost=0.
 tables=cache.build(b,jnp.array(len(gp.positions)),jax.random.PRNGKey(seed+1000+1665),jax.random.PRNGKey(seed+271828))
 for k in range(a.seconds*5):
  clock=round(333+k*.2,8);block=round(clock/.2)
  if k%25==0:
   st=time.perf_counter();tables=cache.planning(b,tables,jax.random.PRNGKey(seed+1000+block));seeds,shifted,meta=m.legacy.seed_bank(x,q,env,cfg,previous)
   with jax.experimental.disable_x64():theta,score,search,forecast,v=planner(*m.legacy.float32((x,q,tables,seeds,shifted)),jax.random.PRNGKey(seed+2000+block));jax.block_until_ready(theta)
   if bool(v['no_feasible']):theta=seeds[0]
   refs,aa=m.legacy.old.nominal_reference(theta.astype(jnp.float64),x,cfg);previous=theta
   plans.append(dict(t=clock,no_feasible=bool(v['no_feasible']),predicted_states=np.asarray(forecast.x).tolist(),seconds=time.perf_counter()-st))
  end,rows=execute(x,q,refs,aa,jnp.array((k%25)*.2),tables,truth,alive);jax.block_until_ready(end)
  xs,qs,us,nom,slack,corr,feasible,gaps,costs,living,hits=map(np.asarray,rows)
  minimum=min(minimum,float(gaps.min()));infeasible+=int((~feasible).sum());cost+=float(costs.sum())
  rowsout.extend([[round(clock+(j+1)*.05,8),*z.tolist(),*us[j].tolist(),float(slack[j]),bool(feasible[j]),float(gaps[j])] for j,z in enumerate(xs)])
  x,q,alive=end
  if not bool(alive):contact=clock+float(hits[np.flatnonzero(np.isfinite(hits))[0]]);break
  newclock=round(clock+.2,8);y=np.asarray(truth.at(x[:2]))[0]+.1*noise[round(newclock/.2)]
  gp.update(x[None,:2],jnp.asarray(y)[None],train_kernel=(len(gp.positions)+1)%50==0);b=gp.belief;tables=cache.safety(b,tables,jax.random.PRNGKey(seed+271828));jax.block_until_ready(tables)
 result=dict(profile=source.name,lengths=np.asarray(b.basis.lengths).tolist(),contact_s=contact,minimum_clearance_m=minimum,infeasible_commands=infeasible,
             completed_seconds=(k+1)*.2,monitoring_cost=cost,relative_sample_covariance_error=relative,wall_seconds=time.perf_counter()-start,plans=plans,samples=rowsout)
 report.append(result);(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('plans','samples')}),flush=True)
signal.alarm(0)
