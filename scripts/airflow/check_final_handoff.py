"""Bounded check of the declared handoff test on saved real-survey posteriors."""
from pathlib import Path
import gzip,json,signal
import numpy as np
import jax.numpy as jnp
import full_learning_monitoring as m
from dataclasses import replace
import cluster_environment as ce
from cached_poam_fields import CachedFields
from benchmark_recorded_planner import restore_inference
from handoff_validation import HandoffValidator
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120s validation budget')));signal.alarm(120)
cfg=m.settings();pcfg=replace(m.legacy.physical_settings(cfg),safety_margin=cfg.safety_margin)
env,grid,spec=ce.environment(pcfg,m.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
basis,points,shape=m.table_grid(env.bounds,.1)
_,extended,_=m.table_grid(env.bounds,.4)
outside=np.any((np.asarray(extended)<np.asarray(env.bounds[0]))|(np.asarray(extended)>np.asarray(env.bounds[1])),axis=1)
support=jnp.asarray(np.vstack((np.asarray(env.cells),np.asarray(extended)[outside])))
cache=CachedFields(points,shape,cfg,grid['all_cells']);validator=HandoffValidator(env,cfg,pcfg,basis,spec)
import jax
out=[]
for s in [1,2]:
 source=Path(f'output/task_learning_20261105/final_baseline/s{s}-center-baseline1.json.gz')
 r=json.load(gzip.open(source));b=restore_inference(source.with_name(source.name.replace('.json.gz','-posterior.npz')),support)
 key=jax.random.PRNGKey(113)
 tables=cache.build(b,jnp.array(len(r['observations'])),key,key)
 truth=m.m.physical.load_wind(s,m.m.physical.SITE/'revised')
 seed=next(c['seed'] for c in ce.CASES if c['id']=='center')
 row=validator.check(jnp.asarray(r['samples'][-1][1:5]),jnp.asarray(r['clarity_frames'][-1]['q']),b,cache,tables,truth,seed)
 row.update(scenario=s,rmse_m_s=r['summary']['final_rmse_m_s']);print(json.dumps(row),flush=True);out.append(row)
Path('output/matched_duration_20261105/handoff-final-validation.json').write_text(json.dumps(out,indent=2))
