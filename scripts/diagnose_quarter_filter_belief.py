"""Isolate fixed controller uncertainty in the recorded initial route ranking.

Both diagnostic plants use the zero prior-mean field. The actual hidden field
is not used. Only the controller's recorded construction-field spread changes.
"""
import json,time
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from voi_diffusion_ipp.quarter.continuous import ControlSettings,rollout
from voi_diffusion_ipp.quarter.environment import environment

root=Path('output/european_quarter/planner_investigation');source=root/'supported300/two_crosswind_proposed_8101.json';d=json.loads(source.read_text());cfg=ControlSettings(**d['config']);env=environment(d['layout'],cfg.environment_scale,cfg.inspection_scale,d['case']);check=json.loads((root/'selection_bank128.json').read_text())
with np.load(source.parent/d['plans'][0]['audit']) as a:values={k:a[k] for k in ['initial_state','clarity','mean','construction','finalists']}
evaluate=jax.jit(lambda theta,construction:rollout(theta,jnp.array(values['initial_state']),jnp.array(values['clarity']),jnp.array(values['mean']),construction,jnp.array(values['mean']),env,cfg,True));rows=[]
for group in check['route_groups']:
 if group['first_crossing'] not in 'ABC' or len(group['first_crossing'])!=1:continue
 for spread in (1.,0.):
  start=time.perf_counter();r=evaluate(jnp.array(values['finalists'][group['best_index']]),jnp.array(values['construction'])*spread);jax.block_until_ready(r);p=np.array(r.x[:,:2]);region=np.array(env.regions[1]);inside=np.all(p>=region[:2],axis=1)&np.all(p<=region[2:],axis=1);arrival=float(np.flatnonzero(inside)[0]*cfg.dt) if inside.any() else None
  row=dict(reference_first_crossing=group['first_crossing'],construction_spread=spread,monitoring_cost=float(r.monitor),soft_penalty=float(r.penalty),first_east_region_entry=arrival,final_position=p[-1].tolist(),terminated=bool(r.terminated),seconds=time.perf_counter()-start);rows.append(row);print(row,flush=True)
out=dict(source=str(source),notes='Recorded nominal reference, initial state, clarity and mean held identical. Zero mean physical field in both cases, not the actual hidden wind. Zero spread is an idealized controller-uncertainty counterfactual, not a claimed reachable posterior.',cases=rows);(root/'filter_belief_counterfactual.json').write_text(json.dumps(out,indent=2))
