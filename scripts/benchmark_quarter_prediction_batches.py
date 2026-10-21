"""Batch-width timing with identical recorded physical prediction work."""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
import numpy as np
import jax,jax.numpy as jnp
from voi_diffusion_ipp.quarter.continuous import ControlSettings,rollout,aggregate_costs
from voi_diffusion_ipp.quarter.environment import environment

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batches',type=int,nargs='+',default=[1,4,8,16]);a=p.parse_args();d=json.loads(a.source.read_text());cfg=ControlSettings(**d['config']);env=environment(d['layout'],cfg.environment_scale,cfg.inspection_scale,d['case']);data=np.load(a.source.parent/d['plans'][0]['audit'])
args=tuple(jnp.array(v) for v in [data['samples'][0],*[data[k] for k in ('initial_state','clarity','mean','construction','performance_fields')]])
rows=[];baseline=None
for width in a.batches:
 @jax.jit
 def evaluate(theta,x,q,mean,construction,fields):
  def one(th):
   def physical(field):
    r=rollout(th,x,q,mean,construction,field,env,cfg,True)
    return jnp.array([r.monitor,r.penalty])
   v=jax.vmap(physical)(fields)
   return aggregate_costs(v[:,0],v[:,1],cfg.penalty_aggregation)
  return jax.lax.map(one,theta,batch_size=width)
 seconds=[]
 for repeat in range(4):
  start=time.perf_counter();result=evaluate(*args);result.block_until_ready();seconds.append(time.perf_counter()-start)
 result=np.array(result)
 if baseline is None:baseline=result
 np.testing.assert_allclose(result,baseline,rtol=3e-5,atol=3e-5)
 rows.append(dict(batch_size=width,compile_and_first=seconds[0],warm_seconds=seconds[1:],max_cost_difference=float(np.max(abs(result-baseline)))))
 print(rows[-1],flush=True);a.output.write_text(json.dumps(rows,indent=2));jax.clear_caches()
