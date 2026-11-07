"""Full-size numerical-equivalence and timing check; no simulation or new data."""
import argparse,json,pickle,signal,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import full_learning_monitoring as im
from waypoint_poam import train

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('240s benchmark bound')));signal.alarm(240)
    with (im.m.physical.ROOT/'output/gp_safety_20261104/sampling/s1-dense-profile.pkl').open('rb') as f:gp=pickle.load(f)
    b=gp.belief;fast=b._replace(basis=b.basis._replace(support_inverse=jnp.zeros((0,0),dtype=b.mean.dtype)))
    rng=np.random.default_rng(214);ix=rng.integers(len(gp.positions),size=(4,256))
    bx,by=jnp.asarray(gp.positions[ix]),jnp.asarray(gp.measurements[ix]);n=jnp.array(len(gp.positions),dtype=b.mean.dtype)
    eps=jax.random.normal(jax.random.PRNGKey(214),(4,*b.sample_shape),dtype=b.mean.dtype)
    sample=jax.jit(lambda belief:jax.vmap(lambda c:belief.basis.evaluate(b.basis.support,c))(belief.sample_coefficients(eps)))
    rows=[]
    for repeat in range(3):
        out=[];samples=[];times={}
        for name,belief in [('inverse',b),('cholesky',fast)]:
            start=time.perf_counter();trained,*_=train(belief,bx,by,n,gp.adam_m,gp.adam_v,gp.adam_step,.01);jax.block_until_ready(trained)
            times[name+'_training_s']=time.perf_counter()-start;out.append(trained)
            start=time.perf_counter();value=sample(trained);jax.block_until_ready(value)
            times[name+'_full_support_sampling_s']=time.perf_counter()-start;samples.append(value)
        np.testing.assert_allclose(samples[0],samples[1],rtol=1e-7,atol=1e-7)
        for left,right in zip(out[0].predict(b.basis.support),out[1].predict(b.basis.support)):
            np.testing.assert_allclose(left,right,rtol=1e-9,atol=1e-9)
        row=dict(repeat=repeat,compilation_included=repeat==0,**times,
            max_field_difference=float(jnp.max(jnp.abs(samples[0]-samples[1]))))
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(inducing=len(b.mean),support=len(b.basis.support),fields=4,device=str(jax.devices()),
        timing_conditions='Other simulation processes may be active; equivalence is unaffected. Do not label these timings isolated.',rows=rows)
    a.output.write_text(json.dumps(report,indent=2));signal.alarm(0)

if __name__=='__main__':main()
