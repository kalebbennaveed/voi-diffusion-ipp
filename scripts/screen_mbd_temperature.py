"""Matched MBD temperature diagnostics at a recorded physical state.

This is a planning-only experiment, never a flight or performance comparison.
All temperatures use the same state, clarity, random draws and trajectory space.
"""
import argparse,json,time
from dataclasses import replace
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from voi_diffusion_ipp.quarter.continuous import ControlSettings,optimizer,rollout,continuation
from voi_diffusion_ipp.quarter.environment import environment,TABLE_SHAPE

p=argparse.ArgumentParser()
p.add_argument('--record',required=True)
p.add_argument('--output',required=True)
p.add_argument('--temperature',nargs='+',type=float,default=[.25,1.,4.,10.])
p.add_argument('--iterations',type=int,default=64)
p.add_argument('--proposals',type=int,default=256)
p.add_argument('--seed',type=int,default=7311)
a=p.parse_args()
source=json.loads(Path(a.record).read_text())
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
(out/'initial_record.json').write_text(json.dumps(source,separators=(',',':')))
cfg=ControlSettings(**source['config'])
cfg=replace(cfg,iterations=a.iterations,proposals=a.proposals,worlds=1,validation_worlds=1,batch_size=a.proposals)
env=environment(source['layout'],cfg.environment_scale,cfg.inspection_scale)
frame=source['frames'][-1]
x=jnp.array(frame['p'][:2]+frame['velocity'][:2],jnp.float32)
q=jnp.array(frame['q'],jnp.float32)
mean=jnp.zeros(TABLE_SHAPE)
construction=jnp.zeros((cfg.construction_worlds,*TABLE_SHAPE))
worlds=jnp.zeros((1,*TABLE_SHAPE))
previous=continuation(jnp.array(source['plans'][-1]['policy'],jnp.float32),cfg)
rows=[]
for temperature in a.temperature:
    current=replace(cfg,temperature=temperature)
    start=time.perf_counter()
    found=optimizer(env,current,'blind')(x,q,mean,construction,worlds,worlds,previous,jax.random.PRNGKey(a.seed),jnp.array(1.),jnp.array(False))
    jax.block_until_ready(found)
    elapsed=time.perf_counter()-start
    theta,cost,_,search,index=found
    prediction=jax.jit(lambda th:rollout(th,x,q,mean,construction,mean,env,current,False))(theta)
    points=np.asarray(prediction.x[:,:2]);center=np.asarray(env.regions[1,[0,1]]+env.regions[1,[2,3]])/2
    row=dict(status='planning_only',source_time=frame['t'],temperature=temperature,iterations=current.iterations,proposals=current.proposals,predicted_cost=float(cost),monitor=float(prediction.monitor),penalty=float(prediction.penalty),minimum_east_distance=float(np.linalg.norm(points-center,axis=1).min()),seconds=elapsed,points=points.tolist(),initial_cost=float(search.initial_cost),empty_batches=int(np.asarray(search.trace)[:,-1].sum()))
    rows.append(row)
    np.savez_compressed(out/f'temperature_{temperature:g}.npz',**{k:np.asarray(v) for k,v in search._asdict().items()},chosen_theta=np.asarray(theta),initial_state=np.asarray(x),clarity=np.asarray(q))
    (out/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    print({k:v for k,v in row.items() if k!='points'},flush=True)
    jax.clear_caches()
