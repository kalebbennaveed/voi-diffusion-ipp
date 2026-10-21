"""Oracle route diagnostics, explicitly separate from a learning experiment.

Give a diagnostic controller the exact wind to ask whether the calm detour is
even preferable with the current vehicle, geometry and sensing. The actual MBD
mission planner never receives this information. No hand-drawn route is used.
"""
import json,time
from pathlib import Path
from dataclasses import replace
import numpy as np
import jax,jax.numpy as jnp
from voi_diffusion_ipp.quarter.environment import environment,wind,TABLE_POINTS,TABLE_SHAPE
from voi_diffusion_ipp.quarter.continuous import ControlSettings,rollout
from voi_diffusion_ipp.quarter.rrt import demonstrations
from investigate_quarter_planner import crossings

out=Path('output/european_quarter/planner_investigation');out.mkdir(parents=True,exist_ok=True)
base=ControlSettings(**json.loads(Path('output/european_quarter/rrt_mbd_execution_300/two_forward_proposed_8101.json').read_text())['config'])
rows=[]
for scale,horizon in [(.35,45.),(.45,45.),(.45,60.)]:
 cfg=replace(base,environment_scale=scale,horizon=horizon,rrt_visits=1,rrt_diversity=True,prediction_filter='hard',prediction_substeps=1)
 for case,strength in [('crosswind',3.1),('crosswind',3.9),('reverse',3.1),('calm_a',3.1)]:
  env=environment('two',scale,.55,case);state=jnp.array([-28*scale,8*scale,0.,0.]);q=jnp.full(len(env.cells),.35).at[:42].set(.65)
  controls,record=demonstrations(state,env,cfg,8101)
  selected=[i for i,r in enumerate(record['records']) if r.get('goal_index')==1]
  actual=wind(TABLE_POINTS*scale,case,strength,scale).reshape(TABLE_SHAPE)
  @jax.jit
  def evaluate(th,physical):
   def one(a):
    r=rollout(a,state,q,physical,jnp.repeat(physical[None],2,axis=0),physical,env,cfg)
    region=(env.regions[1,:2]+env.regions[1,2:])/2
    reached=jnp.linalg.norm(r.x[:,:2]-region,axis=1)<1.
    arrival=jnp.where(reached.any(),jnp.argmax(reached)*cfg.dt,jnp.nan)
    return r.monitor,r.penalty,arrival,r.x
   return jax.lax.map(one,th,batch_size=8)
  values=tuple(np.asarray(a) for a in evaluate(jnp.array(controls[selected]),actual))
  for j,i in enumerate(selected):
   r=record['records'][i];cs=crossings(np.array(r['path']),scale)
   row=dict(scale=scale,horizon=horizon,case=case,wind_strength=strength,passage=cs[0] if cs else 'none',cruise=r['cruise'],monitor=float(values[0][j]),penalty=float(values[1][j]),arrival=None if np.isnan(values[2][j]) else float(values[2][j]))
   rows.append(row);print(row,flush=True)
  np.savez_compressed(out/f'oracle_routes_{case}_{strength}_{scale}_{horizon}.npz',positions=values[3],parameters=controls[selected],physical=np.array(actual))
  (out/'oracle_routes.json').write_text(json.dumps(rows,indent=2))
  jax.clear_caches()
