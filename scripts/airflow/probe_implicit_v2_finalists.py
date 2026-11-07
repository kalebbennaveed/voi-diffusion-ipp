"""Bounded exhaustive geometric-seed diagnosis on recorded beliefs; no truth."""
import os,sys,time,signal,json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import replace
import implicit_monitoring_v2 as im
import cluster_environment as ce


def main():
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('90 s seed audit limit')));signal.alarm(90)
    root=im.m.physical.ROOT;source=root/'results/airflow/implicit_v2_20261104/missions300'
    cfg=im.settings();pcfg=replace(im.legacy.physical_settings(cfg),safety_margin=cfg.safety_margin)
    env,grid,spec=ce.environment(pcfg,root/'results/airflow/cluster_revision_20261103/v2/environment-spec.json');basis,_,_=im.table_grid(env.bounds)
    env,basis=im.legacy.float32((env,basis))
    with jax.experimental.disable_x64():
        @jax.jit
        def run(theta,x,q,tables):
            actual=jax.vmap(lambda w:im.execution_forecast(theta,x,q,tables,w,basis,env,cfg,True))(tables.validation)
            blind=im.execution_forecast(theta,x,q,tables,tables.validation[0],basis,env,cfg,False)
            return actual,blind
    reports=[]
    for ident,cycle in [('s1-center-aware',12),('s1-center-aware',48),('s1-center-aware',59),('s2-center-aware',12),('s2-center-aware',47)]:
        z=np.load(source/f'{ident}-plan-{cycle:03d}.npz')
        t=im.Tables(*map(jnp.asarray,[z['mean'],z['std'],z['support'],z['planning_fields'],z['safety_fields'],z['validation_fields']]))
        x,q,t=im.legacy.float32((jnp.asarray(z['initial_state']),jnp.asarray(z['initial_clarity']),t))
        params=im.legacy.float32(jnp.asarray(np.vstack((z['initial_parameters'],z['theta']))))
        candidates=[];ts=time.perf_counter()
        for i,theta in enumerate(params):
            with jax.experimental.disable_x64():r,blind=run(theta,x,q,t);jax.block_until_ready(r)
            states=np.asarray(r.x);lo,hi=np.asarray(spec['regions'][0]['bounds']);visits=np.all((states[:,:,:2]>=lo)&(states[:,:,:2]<=hi),axis=2).any(1)
            bs=np.asarray(blind.x);blind_A=bool(np.all((bs[:,:2]>=lo)&(bs[:,:2]<=hi),axis=1).any())
            candidates.append(dict(index=i,cost=float(jnp.mean(r.monitor+r.penalty)),feasible=bool(jnp.all(~r.terminated&(r.max_slack<1e-4))),
                A=visits.tolist(),max_slack=np.asarray(r.max_slack).tolist(),contacts=np.asarray(r.terminated).tolist(),
                still_air_unfiltered_cost=float(blind.monitor+blind.penalty),still_air_unfiltered_A=blind_A,still_air_contact=bool(blind.terminated)))
        row=dict(run=ident,t=cycle*5,candidates=candidates,wall_s=time.perf_counter()-ts);reports.append(row);print(json.dumps(row),flush=True)
    (root/'output/diagnostics/implicit_v2_20261104/all-seed-check.json').write_text(json.dumps(reports,indent=2));signal.alarm(0)

if __name__=='__main__':main()
