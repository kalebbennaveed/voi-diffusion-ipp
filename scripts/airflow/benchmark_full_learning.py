"""Isolated timings of one recorded-sensor update and the full 15s MBD call."""
import argparse,gzip,json,pickle,signal,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import full_learning_monitoring as im
import efficient_learning_monitoring as efficient
from cached_poam_fields import CachedFields
import cluster_environment as ce
from waypoint_poam import condition,train

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--factorized-support',action='store_true')
    a=parser.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s isolated benchmark limit')));signal.alarm(240)
    root=im.m.physical.ROOT;cfg=im.settings()
    env,grid,_=ce.environment(im.legacy.physical_settings(cfg),root/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
    basis,points,shape=im.table_grid(env.bounds,.1);cache=CachedFields(points,shape,cfg,grid['all_cells'])
    with jax.experimental.disable_x64():
        planner=im.optimizer(im.legacy.float32(env),cfg,im.legacy.float32(basis),True)
        efficient_planner=efficient.optimizer(im.legacy.float32(env),cfg,im.legacy.float32(basis),True)
    reports=[]
    for scenario in (1,2):
        with (root/f'output/gp_safety_20261104/sampling/s{scenario}-dense-profile.pkl').open('rb') as f:gp=pickle.load(f)
        if a.factorized_support:gp.belief=gp.belief._replace(basis=gp.belief.basis._replace(support_inverse=jnp.zeros((0,0),dtype=gp.belief.mean.dtype)))
        record=json.load(gzip.open(root/f'output/full_learning_20261104/learned/s{scenario}-center-baseline1.json.gz'))
        observation=jnp.asarray(record['observations'][0]);x,q=ce.initial(ce.CASES[0],env,grid,im.legacy.physical_settings(cfg))
        tables=cache.build(gp.belief,jnp.array(len(gp.positions)),jax.random.PRNGKey(9101),jax.random.PRNGKey(279929));jax.block_until_ready(tables)
        rng=np.random.default_rng(442);ix=rng.integers(0,len(gp.positions),size=(4,256))
        bx,by=jnp.asarray(gp.positions[ix]),jnp.asarray(gp.measurements[ix]);rows=[]
        for repeat in range(4):
            start=time.perf_counter();updated,_=condition(gp.belief,observation[None,1:3],observation[None,3:5],gp.mask);jax.block_until_ready(updated)
            conditioning=time.perf_counter()-start
            # Include the actual four-step kernel update on alternate calls.
            training=0.
            if repeat%2==0:
                start=time.perf_counter();updated,gp.adam_m,gp.adam_v,gp.adam_step,_=train(updated,bx,by,jnp.array(len(gp.positions)+repeat+1,dtype=jnp.float64),gp.adam_m,gp.adam_v,gp.adam_step,.01);jax.block_until_ready(updated);training=time.perf_counter()-start
            start=time.perf_counter();tables=cache.safety(updated,tables,jax.random.PRNGKey(279929));jax.block_until_ready(tables);safety_refresh=time.perf_counter()-start
            start=time.perf_counter();tables=cache.planning(updated,tables,jax.random.PRNGKey(9101));jax.block_until_ready(tables);sampling=time.perf_counter()-start
            start=time.perf_counter();seeds,previous,metadata=im.legacy.seed_bank(x,q,env,cfg,None);geometry=time.perf_counter()-start
            args=im.legacy.float32((x,q,tables,seeds,previous));start=time.perf_counter()
            with jax.experimental.disable_x64():result=planner(*args,jax.random.PRNGKey(10101));jax.block_until_ready(result)
            planning=time.perf_counter()-start
            start=time.perf_counter()
            with jax.experimental.disable_x64():fast=efficient_planner(*args,jax.random.PRNGKey(10101));jax.block_until_ready(fast)
            efficient_s=time.perf_counter()-start
            np.testing.assert_allclose(fast[0],result[0],rtol=1e-6,atol=1e-6)
            # Cached costs are the original training evaluations. The previous
            # extra rollout batch has a different float32 reduction layout.
            # Record that numerical discrepancy, and require identical chosen
            # controls and feasibility decisions rather than hiding it.
            fc=np.asarray(fast[-1]['leader_costs']);rc=np.asarray(result[-1]['leader_costs'])
            np.testing.assert_array_equal(np.isfinite(fc),np.isfinite(rc))
            finite=np.isfinite(fc);leader_error=float(np.max(np.abs(fc[finite]-rc[finite]),initial=0))
            np.testing.assert_array_equal(fast[-1]['feasible'],result[-1]['feasible'])
            row=dict(repeat=repeat,compilation_included=scenario==1 and repeat==0,conditioning_s=conditioning,kernel_training_s=training,
                execution_belief_refresh_s=safety_refresh,planning_field_construction_s=sampling,geometry_s=geometry,mbd_search_and_validation_s=planning,
                search_s=float(result[-1]['search_s']),validation_s=float(result[-1]['validation_s']),
                planning_request_s=sampling+geometry+planning,
                efficient_search_and_validation_s=efficient_s,efficient_request_s=sampling+geometry+efficient_s,
                efficient_full_cycle_s=conditioning+training+safety_refresh+sampling+geometry+efficient_s,
                training_score_float32_difference=leader_error,selected_controls_match=True,validation_feasibility_match=True,
                total_with_measurement_and_safety_refresh_s=conditioning+training+safety_refresh+sampling+geometry+planning,
                feasible=not bool(result[-1]['no_feasible']))
            rows.append(row);print(json.dumps(dict(scenario=scenario,**row)),flush=True)
            gp.belief=updated
        reports.append(dict(scenario=scenario,horizon_s=cfg.horizon,device=str(jax.devices()),factorized_support=a.factorized_support,rows=rows))
        (a.output/'report.json').write_text(json.dumps(reports,indent=2))
    signal.alarm(0)

if __name__=='__main__':main()
