"""Bounded benchmark before starting missions; no simulation or model tuning."""
import json,time,signal
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import implicit_monitoring as im
import cluster_environment as ce


def main():
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180 s benchmark limit')))
    signal.alarm(180)
    cfg=im.settings();pcfg=replace(im.legacy.physical_settings(cfg),safety_margin=cfg.safety_margin)
    env,grid,spec=ce.environment(pcfg,im.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
    basis,points,shape=im.table_grid(env.bounds)
    b=im.initial_belief(env,points);sample=im.field_builder(points,shape,cfg)
    with jax.experimental.disable_x64():plan=im.optimizer(im.legacy.float32(env),cfg,im.legacy.float32(basis))
    x,q=ce.initial(ce.CASES[0],env,grid,pcfg)
    # No wind source is used in this compute benchmark. Declared synthetic
    # sensor fixture exercises the measured-posterior branch, not mission data.
    rows=[]
    for iteration in range(4):
        ts=time.perf_counter()
        tables=sample(b,jnp.array(iteration>0,dtype=jnp.int32),jax.random.PRNGKey(1),jax.random.PRNGKey(2));jax.block_until_ready(tables)
        sampling=time.perf_counter()-ts
        t0=time.perf_counter();seeds,prev,_=im.legacy.seed_bank(x,q,env,cfg);initialization=time.perf_counter()-t0
        args=im.legacy.float32((x,q,tables,seeds,prev))
        ts=time.perf_counter()
        with jax.experimental.disable_x64():result=plan(*args,jax.random.PRNGKey(8101));jax.block_until_ready(result)
        optimization=time.perf_counter()-ts
        row=dict(iteration=iteration,compilation=iteration==0,sampling_s=sampling,geometry_s=initialization,
                 mbd_forecast_s=optimization,total_s=sampling+initialization+optimization,cost=float(result[1]),device=str(jax.devices()))
        rows.append(row);print(json.dumps(row),flush=True)
        if iteration==0:
            ts=time.perf_counter();b,mask=im.condition_supported(b,x[None,:2],jnp.array([[-.8,.2]]),jnp.zeros(len(b.mean),bool));jax.block_until_ready(b)
            print(json.dumps(dict(first_condition_compile_s=time.perf_counter()-ts)),flush=True)
    out=im.m.physical.ROOT/'results/airflow/implicit_20261104';out.mkdir(exist_ok=True)
    (out/'benchmark.json').write_text(json.dumps(rows,indent=2))
    signal.alarm(0)


if __name__=='__main__':main()
