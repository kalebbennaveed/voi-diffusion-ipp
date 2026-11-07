"""20 s horizon sensitivity on actual saved beliefs, never simulated wind truth."""
import signal,time,json
from pathlib import Path
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import implicit_monitoring_v5 as im
import cluster_environment as ce


def main():
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('90 s horizon audit limit')));signal.alarm(90)
    root=im.m.physical.ROOT;cfg=replace(im.settings(),horizon=20.)
    pcfg=replace(im.legacy.physical_settings(cfg),safety_margin=cfg.safety_margin)
    env,grid,spec=ce.environment(pcfg,root/'results/airflow/cluster_revision_20261103/v2/environment-spec.json');basis,_,_=im.table_grid(env.bounds)
    with jax.experimental.disable_x64():plan=im.optimizer(im.legacy.float32(env),cfg,im.legacy.float32(basis),True)
    rows=[]
    for ident,cycle in [('s1-center-aware',12),('s1-center-aware',48),('s2-center-aware',12),('s2-center-aware',48)]:
        z=np.load(root/f'results/airflow/implicit_v5_20261104/missions300/{ident}-plan-{cycle:03d}.npz')
        t=im.Tables(*map(jnp.asarray,[z['mean'],z['std'],z['support'],z['planning_fields'],z['safety_fields'],z['validation_fields']]))
        x,q=jnp.asarray(z['initial_state']),jnp.asarray(z['initial_clarity'])
        ts=time.perf_counter();seeds,shifted,metadata=im.legacy.seed_bank(x,q,env,cfg)
        with jax.experimental.disable_x64():out=plan(*im.legacy.float32((x,q,t,seeds,shifted)),jax.random.PRNGKey(2000+cycle));jax.block_until_ready(out)
        elapsed=time.perf_counter()-ts;theta,cost,search,r,validation=out
        lo,hi=np.asarray(spec['regions'][0]['bounds']);xx=np.asarray(r.x)
        row=dict(id=ident,t=cycle*5,horizon=20,observations=int(z['observations']),wall_s=elapsed,compile=len(rows)==0,
                 cost=float(cost),A=np.all((xx[:,:,:2]>=lo)&(xx[:,:,:2]<=hi),axis=2).any(1).tolist(),
                 contacts=np.asarray(r.terminated).tolist(),max_slack=np.asarray(r.max_slack).tolist(),
                 validation={k:np.asarray(v).tolist() for k,v in validation.items() if k!='candidates'},
                 source='Frozen recorded posterior fields; no new or future measurements; no physical wind source')
        rows.append(row);print(json.dumps(row),flush=True)
    (root/'output/diagnostics/implicit_v2_20261104/horizon20-check-v5.json').write_text(json.dumps(rows,indent=2));signal.alarm(0)

if __name__=='__main__':main()
