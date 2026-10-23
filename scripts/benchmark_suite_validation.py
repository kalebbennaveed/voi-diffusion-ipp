"""Compare exact validation scheduling on an immutable search archive."""
import argparse,json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from voi_diffusion_ipp.quarter.continuous import ControlSettings,rollout
from voi_diffusion_ipp.quarter.benchmark_environments import make_environment
p=argparse.ArgumentParser();p.add_argument('record',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=json.loads(a.record.read_text());cfg=ControlSettings(**d['config']);env=make_environment(d['case'])
with np.load(a.record.parent/d['plans'][0]['audit']) as z:
    args=tuple(jnp.asarray(z[k]) for k in ['finalists','finalist_unique','initial_state','clarity','mean','construction','validation_fields'])
rows=[];baseline=None
for batch in [1,4,27]:
    def make(batch):
        @jax.jit
        def kernel(parameters,unique,x,q,m,c,v):
            def values(theta):
                def one(field):
                    r=rollout(theta,x,q,m,c,field,env,cfg,True)
                    return r.monitor+r.penalty
                return jax.vmap(one)(v)
            return jax.lax.map(lambda item:jax.lax.cond(item[1],values,lambda th:jnp.full(cfg.validation_worlds,jnp.inf),item[0]),(parameters,unique),batch_size=None if batch==1 else batch)
        return kernel
    f=make(batch);start=time.perf_counter();values=f(*args);values.block_until_ready();cold=time.perf_counter()-start;elapsed=[]
    for _ in range(3):
        start=time.perf_counter();values=f(*args);values.block_until_ready();elapsed.append(time.perf_counter()-start)
    value=np.asarray(values)
    if baseline is None:baseline=value
    else:np.testing.assert_allclose(value,baseline,atol=2e-4,rtol=2e-4)
    row=dict(batch=batch,cold=cold,warm=elapsed,finite_unique=int(np.asarray(args[1]).sum()),total=len(value));rows.append(row);print(row,flush=True)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2))
