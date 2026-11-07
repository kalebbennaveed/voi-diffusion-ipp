"""Exercise an MBD-blocked information request from a real final recording.

Inference-only smoke test: no synthetic observations or resumed POAM state.
It is not a mission result or a benchmark of online belief assimilation.
"""
import argparse,gzip,json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import full_learning_monitoring as model
import efficient_learning_monitoring as efficient
import cluster_environment as ce
from benchmark_recorded_planner import restore_inference
from cached_poam_fields import CachedFields
from reachable_learning import ReachableLearning


def main():
    p=argparse.ArgumentParser();p.add_argument('recording',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lower-cost',action='store_true');p.add_argument('--downstream',action='store_true');a=p.parse_args()
    r=json.load(gzip.open(a.recording));cfg=model.settings();pcfg=model.legacy.physical_settings(cfg)
    env,grid,_=ce.environment(pcfg,model.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
    basis,points,shape=model.table_grid(env.bounds,.1);_,extended,_=model.table_grid(env.bounds,.4)
    outside=np.any((np.asarray(extended)<env.bounds[0])|(np.asarray(extended)>env.bounds[1]),axis=1)
    support=jnp.asarray(np.vstack((env.cells,np.asarray(extended)[outside])))
    b=restore_inference(a.recording.with_name(a.recording.name.replace('.json.gz','-posterior.npz')),support)
    cache=CachedFields(points,shape,cfg,grid['all_cells']);tables=cache.build(b,jnp.array(len(r['observations'])),jax.random.PRNGKey(661),jax.random.PRNGKey(662))
    x=jnp.asarray(r['samples'][-1][1:5]);q=jnp.asarray(r['clarity_frames'][-1]['q'])
    with jax.experimental.disable_x64():planner=efficient.optimizer(model.legacy.float32(env),cfg,model.legacy.float32(basis),True,wind_error_reserve=.25*cfg.umax,information_trigger='lower-cost' if a.lower_cost else 'stalled',prepare_validation=a.lower_cost)
    seeds,previous,_=model.legacy.seed_bank(x,q,env,cfg,None);args=model.legacy.float32((x,q,tables,seeds,previous))
    with jax.experimental.disable_x64():result=planner(*args,jax.random.PRNGKey(663));jax.block_until_ready(result)
    audit=result[-1]
    learner=ReachableLearning(env,grid,cfg,basis,'mask',global_goals=True,vector_precision=.2,
        prefix_screen='braking-feasible',learned_route_bounds=True,diversify_stalls=True,route_precision=True,downstream_information=a.downstream)
    # Exercise the reference acquisition regardless of whether this particular
    # random MBD bank triggers it; report the actual trigger separately.
    learner.task_reference=np.asarray(audit['learning_reference'])
    local_mean,local_cov=b.predict(x[None,:2]);ts=time.perf_counter()
    path,decision,weights,priority=learner.plan(np.asarray(x),np.asarray(q),b,cache,tables,300.,np.asarray(local_mean[0]),float(jnp.sqrt(local_cov[0,0,0])))
    first=time.perf_counter()-ts;ts=time.perf_counter()
    path,decision,weights,priority=learner.plan(np.asarray(x),np.asarray(q),b,cache,tables,300.2,np.asarray(local_mean[0]),float(jnp.sqrt(local_cov[0,0,0])))
    warm=time.perf_counter()-ts
    assert np.isfinite(path).all() and np.isfinite(weights).all() and np.isfinite(priority).all()
    assert weights.sum()>0 and np.max(priority)>0
    assert np.asarray(decision['information_reference']).shape==(cfg.steps+1,4)
    report=dict(source=str(a.recording),downstream=a.downstream,trigger=bool(audit['request_targeted_learning']),
        improvement=float(audit['uncertainty_blocked_improvement']),displacement=float(audit['selected_forecast_displacement_m']),
        acquisition_first_s=first,acquisition_warm_s=warm,path=path.tolist(),decision=decision)
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('path','decision')}),flush=True)


if __name__=='__main__':main()
