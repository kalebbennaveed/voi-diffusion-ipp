"""Isolate numerical/controller prediction error from wind-estimation error.

The true field is supplied only to this offline diagnostic plant rollout, never
to the controller. State, posterior, construction fields and decisions are the
original unrounded recorded data. No trajectory is reoptimized here.
"""
import json,time
from dataclasses import replace
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from voi_diffusion_ipp.quarter.continuous import ControlSettings,rollout
from voi_diffusion_ipp.quarter.environment import environment,wind,TABLE_POINTS,TABLE_SHAPE

root=Path('output/european_quarter/rrt_mbd_execution_300')
out=Path('output/european_quarter/planner_investigation');out.mkdir(parents=True,exist_ok=True)
report=[]
for case in ('crosswind','reverse'):
 d=json.loads((root/f'two_{case}_proposed_8101.json').read_text());base=ControlSettings(**d['config']);env=environment('two',base.environment_scale,base.inspection_scale,case)
 physical=wind(TABLE_POINTS*base.environment_scale,case,base.wind_strength,base.environment_scale).reshape(TABLE_SHAPE)
 selected=[0,2,6,12]
 for label,changes in [('original',dict(prediction_substeps=1,prediction_filter='soft')),('soft_100ms',dict(prediction_substeps=2,prediction_filter='soft')),('matched_hard_100ms',dict(prediction_substeps=2,prediction_filter='hard')),('matched_hard_50ms',dict(prediction_substeps=4,prediction_filter='hard'))]:
  cfg=replace(base,horizon=base.commit,**changes)
  @jax.jit
  def predict(theta,x,q,mean,construction,field):return rollout(theta,x,q,mean,construction,field,env,cfg)
  for i in selected:
   p=d['plans'][i];a=np.load(root/p['execution_audit']);prev=np.load(root/d['plans'][i-1]['execution_audit']) if i else None
   q=prev['clarity'][-1] if prev else np.array(d['frames'][0]['q']);packed=a['initial_state'];x=np.r_[packed[:2],packed[3:5]]
   start=time.perf_counter();r=predict(jnp.array(a['theta'][:2*cfg.steps]),jnp.array(x),jnp.array(q),jnp.array(a['mean']),jnp.array(a['construction']),physical);jax.block_until_ready(r)
   errors=np.linalg.norm(np.array(r.x)[1:,:2]-a['states'][round(cfg.dt/.01)-1::round(cfg.dt/.01),:2],axis=1)
   row=dict(case=case,t=p['t'],variant=label,mean_position_error=float(errors.mean()),endpoint_position_error=float(errors[-1]),predicted_monitor=float(r.monitor),actual_monitor=float(a['cost_increments'].sum()),predicted_penalty=float(r.penalty),elapsed_seconds=time.perf_counter()-start)
   report.append(row);print(row,flush=True)
  (out/'prediction_error.json').write_text(json.dumps(report,indent=2))
  jax.clear_caches()
