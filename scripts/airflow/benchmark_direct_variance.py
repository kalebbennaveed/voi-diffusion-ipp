"""Bounded exact-variance timing check on a preserved recorded posterior."""
from pathlib import Path
import json,gzip,pickle,time,signal
import numpy as np
import jax
import jax.numpy as jnp
import full_learning_monitoring as m
import efficient_learning_monitoring as efficient
import cluster_environment as ce
from cached_poam_fields import CachedFields
from exact_poam_variance import ResolvedVarianceBasis
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s direct-variance benchmark limit')));signal.alarm(180)
folder=Path('output/matched_duration_20261105/shortscale-survey-diagnostic');r=json.load(gzip.open(folder/'s1-center-baseline1.json.gz'))
with (folder/'s1-center-baseline1-final-profile.pkl').open('rb') as f:gp=pickle.load(f)
cfg=m.settings();env,grid,spec=ce.environment(m.legacy.physical_settings(cfg),m.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
basis,points,shape=m.table_grid(env.bounds,.1);basis=ResolvedVarianceBasis(basis.lower,basis.spacing)
cache=CachedFields(points,shape,cfg,grid['all_cells'],exact_variance=True)
with jax.experimental.disable_x64():planner=efficient.optimizer(m.legacy.float32(env),cfg,m.legacy.float32(basis),True,prepare_validation=True)
x=jnp.asarray(r['samples'][-1][1:5]);q=jnp.asarray(r['clarity_frames'][-1]['q']);b=gp.belief;reports=[]
for repeat in range(4):
 start=time.perf_counter();tables=cache.build(b,jnp.array(len(gp.positions)),jax.random.PRNGKey(116+repeat),jax.random.PRNGKey(117));jax.block_until_ready(tables);field_s=time.perf_counter()-start
 start=time.perf_counter();seeds,previous,_=m.legacy.seed_bank(x,q,env,cfg,None);geometry_s=time.perf_counter()-start
 start=time.perf_counter()
 with jax.experimental.disable_x64():theta,cost,search,forecast,v=planner(*m.legacy.float32((x,q,tables,seeds,previous)),jax.random.PRNGKey(118+repeat));jax.block_until_ready(theta)
 row=dict(repeat=repeat,compilation_included=repeat==0,field_s=field_s,geometry_s=geometry_s,mbd_s=time.perf_counter()-start,search_s=v['search_s'],validation_s=v['validation_s'],accepted=not bool(v['no_feasible']))
 row['total_s']=sum(row[k] for k in ['field_s','geometry_s','mbd_s']);reports.append(row);print(json.dumps(row),flush=True)
Path('output/matched_duration_20261105/direct-variance-benchmark.json').write_text(json.dumps(reports,indent=2));signal.alarm(0)
