"""Bounded diagnostic of safety-envelope coverage at an actual failed transition.

Recorded checkpoints are replayed exactly. Counterfactual execution obtains
new readings only at its own robot positions. No truth reaches MBD or RPCBF.
"""
import gzip,json,pickle,signal,time
from pathlib import Path
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import full_learning_monitoring as model
import efficient_learning_monitoring as efficient
import cluster_environment as ce
from cached_poam_fields import CachedFields
from exact_poam_variance import ResolvedVarianceBasis

out=Path('output/matched_duration_20261105/confidence-diagnostic')
r=json.load(gzip.open(out.parent/'resolved-variance-failed/s1-center-baseline1.json.gz'))
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600 s confidence diagnostic limit')));signal.alarm(600)
reports=[]
for beta in (3.,7.):
 start=time.perf_counter();cfg=replace(model.settings(),safety_confidence=beta);pcfg=model.legacy.physical_settings(cfg)
 env,grid,spec=ce.environment(pcfg,model.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
 basis,points,shape=model.table_grid(env.bounds,.1);basis=ResolvedVarianceBasis(basis.lower,basis.spacing)
 cache=CachedFields(points,shape,cfg,grid['all_cells'],exact_variance=True)
 with (out/'posterior-t285.pkl').open('rb') as f:gp=pickle.load(f)
 with jax.experimental.disable_x64():planner=efficient.optimizer(model.legacy.float32(env),cfg,model.legacy.float32(basis),True,prepare_validation=True)
 execute=model.execution_block(env,cfg,pcfg,basis,seconds=.2)
 truth=model.m.physical.load_wind(1,model.m.physical.SITE/'revised');seed=next(c['seed'] for c in ce.CASES if c['id']=='center')
 noise=np.random.default_rng(seed).normal(size=(5001,2))
 x=jnp.asarray(next(s[1:5] for s in r['samples'] if s[0]==285));q=jnp.asarray(next(s['q'] for s in r['clarity_frames'] if s['t']==285));alive=jnp.array(True)
 previous=jnp.asarray(next(p['selected_parameters'] for p in r['plans'] if p['t']==280))
 tables=cache.build(gp.belief,jnp.array(len(gp.positions)),jax.random.PRNGKey(seed+1000+1425),jax.random.PRNGKey(seed+271828))
 samples=[];plans=[];infeasible=0;contact=None;cost=0.;minimum=100.
 for k in range(60*5):
  clock=round(285+k*.2,8);block=round(clock/.2)
  if k%25==0:
   t=time.perf_counter();tables=cache.planning(gp.belief,tables,jax.random.PRNGKey(seed+1000+block))
   seeds,shifted,meta=model.legacy.seed_bank(x,q,env,cfg,previous)
   with jax.experimental.disable_x64():
    theta,score,search,forecast,audit=planner(*model.legacy.float32((x,q,tables,seeds,shifted)),jax.random.PRNGKey(seed+2000+block));jax.block_until_ready(theta)
   if bool(audit['no_feasible']):theta=seeds[0]
   refs,aa=model.legacy.old.nominal_reference(theta.astype(jnp.float64),x,cfg);previous=theta
   plans.append(dict(t=clock,accepted=not bool(audit['no_feasible']),request_seconds=time.perf_counter()-t,expected_cost=float(score)))
  end,rows=execute(x,q,refs,aa,jnp.array((k%25)*.2),tables,truth,alive);jax.block_until_ready(end)
  xs,qs,us,nom,slack,corr,feasible,gaps,costs,living,hits=map(np.asarray,rows)
  infeasible+=int((~feasible).sum());minimum=min(minimum,float(gaps.min()));cost+=float(costs.sum())
  samples.extend([[round(clock+(j+1)*.05,8),*z.tolist(),float(slack[j]),bool(feasible[j]),float(gaps[j])] for j,z in enumerate(xs)])
  x,q,alive=end
  if not bool(alive):contact=clock+float(hits[np.flatnonzero(np.isfinite(hits))[0]]);break
  t=round(clock+.2,8);reading=np.asarray(truth.at(x[:2]))[0]+.1*noise[round(t/.2)]
  gp.update(x[None,:2],jnp.asarray(reading)[None],train_kernel=(len(gp.positions)+1)%50==0)
  tables=cache.safety(gp.belief,tables,jax.random.PRNGKey(seed+271828));jax.block_until_ready(tables)
 result=dict(safety_confidence=beta,seconds=(k+1)*.2,contact=contact,infeasible_commands=infeasible,min_clearance=minimum,cost=cost,wall_seconds=time.perf_counter()-start,plans=plans,samples=samples)
 reports.append(result);(out/'transition-report.json').write_text(json.dumps(reports,indent=2,allow_nan=False))
 print(json.dumps({key:val for key,val in result.items() if key not in ('plans','samples')}),flush=True)
signal.alarm(0)
