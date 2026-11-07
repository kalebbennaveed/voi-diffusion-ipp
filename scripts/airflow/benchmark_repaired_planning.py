"""Isolated timings of one recorded-sensor update and the full 15s MBD call."""
import argparse,gzip,json,pickle,signal,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import rmse_monitoring as im
import cluster_environment as ce
from waypoint_poam import condition,train

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s isolated benchmark limit')));signal.alarm(180)
    root=im.m.physical.ROOT;cfg=im.settings()
    env,grid,_=ce.environment(im.legacy.physical_settings(cfg),root/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
    basis,points,shape=im.table_grid(env.bounds,.1);fields=im.field_builder(points,shape,cfg)
    with jax.experimental.disable_x64():planner=im.optimizer(im.legacy.float32(env),cfg,im.legacy.float32(basis),True)
    reports=[]
    for scenario in (1,2):
        with (root/f'output/gp_safety_20261104/sampling/s{scenario}-dense-profile.pkl').open('rb') as f:gp=pickle.load(f)
        record=json.load(gzip.open(root/f'output/gp_safety_20261104/informed300/s{scenario}-center-baseline1.json.gz'))
        observation=jnp.asarray(record['observations'][0]);x,q=ce.initial(ce.CASES[0],env,grid,im.legacy.physical_settings(cfg))
        rng=np.random.default_rng(442);ix=rng.integers(0,len(gp.positions),size=(4,256))
        bx,by=jnp.asarray(gp.positions[ix]),jnp.asarray(gp.measurements[ix]);rows=[]
        for repeat in range(4):
            start=time.perf_counter();updated,_=condition(gp.belief,observation[None,1:3],observation[None,3:5],gp.mask);jax.block_until_ready(updated)
            conditioning=time.perf_counter()-start
            # Include the actual four-step kernel update on alternate calls.
            training=0.
            if repeat%2==0:
                start=time.perf_counter();updated,*_=train(updated,bx,by,jnp.array(len(gp.positions)+1,dtype=jnp.float64),gp.adam_m,gp.adam_v,gp.adam_step,.01);jax.block_until_ready(updated);training=time.perf_counter()-start
            start=time.perf_counter();tables=fields(updated,jnp.array(len(gp.positions)+1),jax.random.PRNGKey(9101),jax.random.PRNGKey(279929));jax.block_until_ready(tables);sampling=time.perf_counter()-start
            start=time.perf_counter();seeds,previous,metadata=im.legacy.seed_bank(x,q,env,cfg,None);geometry=time.perf_counter()-start
            args=im.legacy.float32((x,q,tables,seeds,previous));start=time.perf_counter()
            with jax.experimental.disable_x64():result=planner(*args,jax.random.PRNGKey(10101));jax.block_until_ready(result)
            planning=time.perf_counter()-start
            row=dict(repeat=repeat,compilation_included=scenario==1 and repeat==0,conditioning_s=conditioning,kernel_training_s=training,
                field_construction_s=sampling,geometry_s=geometry,mbd_search_and_validation_s=planning,
                search_s=float(result[-1]['search_s']),validation_s=float(result[-1]['validation_s']),
                total_s=conditioning+training+sampling+geometry+planning,feasible=not bool(result[-1]['no_feasible']))
            rows.append(row);print(json.dumps(dict(scenario=scenario,**row)),flush=True)
        reports.append(dict(scenario=scenario,horizon_s=cfg.horizon,device=str(jax.devices()),rows=rows))
        (a.output/'report.json').write_text(json.dumps(reports,indent=2))
    signal.alarm(0)

if __name__=='__main__':main()
