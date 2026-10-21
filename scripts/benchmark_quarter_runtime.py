"""Compare CPU runtimes on identical MBD rollout inputs, without changing math."""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from voi_diffusion_ipp.quarter.continuous import ControlSettings, rollout
from voi_diffusion_ipp.quarter.environment import environment

p=argparse.ArgumentParser()
p.add_argument('--audit',required=True)
p.add_argument('--output',required=True)
p.add_argument('--batch',type=int,default=8)
p.add_argument('--unroll-backup',type=int,default=1)
p.add_argument('--cost-only',action='store_true')
p.add_argument('--barrier',action='store_true')
p.add_argument('--elementwise',action='store_true')
a=p.parse_args()
if a.elementwise:
    from voi_diffusion_ipp.wind.cache import TableBasis,cubic_weights
    from voi_diffusion_ipp.safety import spline
    def interpolate(self,locations,field):
        xy=(jnp.atleast_2d(locations)-self.lower)/self.spacing
        ij=jnp.floor(xy).astype(jnp.int32);w=cubic_weights(xy-ij)
        offsets=jnp.arange(-1,3);ix=ij[:,0,None]+offsets;iy=ij[:,1,None]+offsets
        patch=field[iy[:,:,None],ix[:,None,:]]
        result=jnp.sum(jnp.sum(patch*w[:,0,None,:,None],axis=2)*w[:,1,:,None],axis=1)
        valid=jnp.all(ij>=1,axis=1)&(ij[:,0]<field.shape[1]-2)&(ij[:,1]<field.shape[0]-2)
        return jnp.where(valid[:,None],result,jnp.nan)
    TableBasis.evaluate=interpolate
    def coefficients(y):
        m=jnp.sum(jnp.asarray(spline.second_derivative_operator(y.shape[0]))*y[None,:],axis=1)
        return jnp.stack(((m[1:]-m[:-1])/6,m[:-1]/2,y[1:]-y[:-1]-(2*m[:-1]+m[1:])/6,y[:-1]),axis=-1)
    spline.coefficients=coefficients
if a.unroll_backup!=1:
    ordinary_scan=jax.lax.scan
    def benchmark_scan(*args,**kwargs):
        if kwargs.get('length')==8:
            kwargs['unroll']=a.unroll_backup
        return ordinary_scan(*args,**kwargs)
    jax.lax.scan=benchmark_scan
source=np.load(a.audit)
cfg=ControlSettings(horizon=60,worlds=len(source['performance_fields']))
env=environment()
theta=np.random.default_rng(371).normal(0,.3,(a.batch,2*cfg.steps)).astype(np.float32)
args=tuple(jnp.asarray(z) for z in (theta,source['initial_state'],source['clarity'],source['mean'],source['construction'],source['performance_fields']))

@jax.jit
def evaluate(theta,x,q,mean,construction,worlds):
    def one(th,w):
        result=rollout(th,x,q,mean,construction,w,env,cfg,True)
        if a.barrier:result=jax.lax.optimization_barrier(result)
        return result.monitor+result.penalty if a.cost_only else result
    return jax.vmap(lambda th:jax.vmap(lambda w:one(th,w))(worlds))(theta)

timings=[]
for repeat in range(4):
    start=time.perf_counter();result=evaluate(*args);jax.block_until_ready(result)
    timings.append(time.perf_counter()-start)
    print('pass',repeat,'seconds',timings[-1],flush=True)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out.with_suffix('.npz'),**({'cost':np.asarray(result)} if a.cost_only else {name:np.asarray(value) for name,value in result._asdict().items()}))
out.with_suffix('.json').write_text(json.dumps(dict(jax=jax.__version__,devices=list(map(str,jax.devices())),xla_flags=os.environ.get('XLA_FLAGS',''),input_audit=a.audit,batch=a.batch,unroll_backup=a.unroll_backup,cost_only=a.cost_only,barrier=a.barrier,elementwise=a.elementwise,compile_and_first=timings[0],warm_seconds=timings[1:]),indent=2))
