import json,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from voi_diffusion_ipp.quarter.environment import environment,TABLE_SHAPE
from voi_diffusion_ipp.quarter.continuous import ControlSettings,optimizer,rollout
out=Path('output/european_quarter/geometry_diagnostic');out.mkdir(exist_ok=True)
rows=[]
for geometry in ['no_obstacles','original_buildings']:
 for temperature in [.25,1.,10.]:
  cfg=ControlSettings(environment_scale=.5,inspection_scale=.55,horizon=40,iterations=64,proposals=512,batch_size=256,worlds=1,validation_worlds=1,temperature=temperature,initialization='zero')
  env=environment('two',.5,.55)
  if geometry=='no_obstacles':env=env._replace(rectangles=jnp.empty((0,5)))
  x=jnp.array([-14.,4.,0.,0.]);q=jnp.full(84,.35).at[:42].set(.65)
  m=jnp.zeros(TABLE_SHAPE);c=jnp.zeros((2,*TABLE_SHAPE));w=m[None]
  t=time.perf_counter();found=optimizer(env,cfg,'blind')(x,q,m,c,w,w,jnp.zeros(800),jax.random.PRNGKey(8101),jnp.array(1.),jnp.array(True));jax.block_until_ready(found);elapsed=time.perf_counter()-t
  theta,score,_,search,ix=found
  r=jax.jit(lambda th:rollout(th,x,q,m,c,m,env,cfg,False))(theta)
  pts=np.array(r.x[:,:2]);dist=np.linalg.norm(pts-[14,-5],axis=1)
  row=dict(geometry=geometry,temperature=temperature,status='planning_only',cost=float(score),monitor=float(r.monitor),penalty=float(r.penalty),east_distance=float(dist.min()),closest_time=float(dist.argmin()*.1),seconds=elapsed,points=pts.tolist(),initial_cost=float(search.initial_cost),empty_batches=int(np.array(search.trace)[:,-1].sum()))
  print({k:v for k,v in row.items() if k!='points'},flush=True);rows.append(row)
  (out/'results.json').write_text(json.dumps(rows,indent=2))
  np.savez_compressed(out/f'{geometry}_{temperature:g}.npz',**{k:np.asarray(v) for k,v in search._asdict().items()},theta=np.asarray(theta))
  jax.clear_caches()
