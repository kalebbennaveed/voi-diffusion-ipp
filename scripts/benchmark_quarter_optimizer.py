"""Compare execution settings on the same complete MBD planning call."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from voi_diffusion_ipp.quarter.continuous import ControlSettings,optimizer
from voi_diffusion_ipp.quarter.environment import environment

p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--output',required=True)
a=p.parse_args();data=np.load(a.audit)
cfg=ControlSettings(horizon=60,iterations=4,proposals=64,worlds=len(data['performance_fields']),validation_worlds=len(data['validation_fields']),temperature=10,batch_size=64)
plan=optimizer(environment(),cfg,'proposed')
args=[jnp.asarray(data[k]) for k in ['initial_state','clarity','mean','construction','performance_fields','validation_fields']]
args+=[jnp.zeros(2*cfg.steps),jax.random.PRNGKey(7311),jnp.array(.5),jnp.array(True)]
seconds=[]
for repeat in range(2):
    start=time.perf_counter();result=plan(*args);jax.block_until_ready(result);seconds.append(time.perf_counter()-start)
    print(repeat,'seconds',seconds[-1],'selected cost',float(result[1]),flush=True)
out=Path(a.output);out.parent.mkdir(exist_ok=True,parents=True)
np.savez_compressed(out.with_suffix('.npz'),theta=np.asarray(result[0]),validation_cost=np.asarray(result[1]),**{k:np.asarray(v) for k,v in result[3]._asdict().items()})
out.with_suffix('.json').write_text(json.dumps(dict(xla_flags=os.environ.get('XLA_FLAGS',''),jax=jax.__version__,seconds=seconds,input_audit=a.audit),indent=2))
