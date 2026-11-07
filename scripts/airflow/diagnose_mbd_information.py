"""Read-only reference/GP and acquisition diagnosis from a recorded posterior.

No observations are added. True wind is deliberately not loaded. Candidate
choices here are counterfactual diagnostics, not a resumed mission.
"""
import argparse,gzip,json,signal,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import full_learning_monitoring as model
import efficient_learning_monitoring as efficient
import cluster_environment as ce
from benchmark_recorded_planner import restore_inference
from cached_poam_fields import CachedFields
from reachable_learning import ReachableLearning,measured_route_mask
from wind_model_trust import acceleration_error_bound


def main():
    p=argparse.ArgumentParser();p.add_argument('recording',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180-second diagnostic limit')));signal.alarm(180)
    r=json.load(gzip.open(a.recording));cfg=model.settings();pcfg=model.legacy.physical_settings(cfg)
    env,grid,_=ce.environment(pcfg,model.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
    basis,points,shape=model.table_grid(env.bounds,.1);_,extended,_=model.table_grid(env.bounds,.4)
    outside=np.any((np.asarray(extended)<env.bounds[0])|(np.asarray(extended)>env.bounds[1]),axis=1)
    support=jnp.asarray(np.vstack((env.cells,np.asarray(extended)[outside])))
    b=restore_inference(a.recording.with_name(a.recording.name.replace('.json.gz','-posterior.npz')),support)
    cache=CachedFields(points,shape,cfg,grid['all_cells']);tables=cache.build(b,jnp.array(len(r['observations'])),jax.random.PRNGKey(661),jax.random.PRNGKey(662))
    x=jnp.asarray(r['samples'][-1][1:5]);q=jnp.asarray(r['clarity_frames'][-1]['q'])
    seeds,previous,metadata=model.legacy.seed_bank(x,q,env,cfg,None)
    with jax.experimental.disable_x64():planner=efficient.optimizer(model.legacy.float32(env),cfg,model.legacy.float32(basis),True,
        wind_error_reserve=.25*cfg.umax,information_trigger='lower-cost',prepare_validation=True)
    args=model.legacy.float32((x,q,tables,seeds,previous))
    with jax.experimental.disable_x64():result=planner(*args,jax.random.PRNGKey(663));jax.block_until_ready(result)
    audit=result[-1];rows=[];predict=jax.jit(b.predict)
    for i,seed in enumerate(seeds):
        reference,aa=model.legacy.old.nominal_reference(seed,x,cfg)
        mu,cov=predict(reference[:,:2]);sigma=jnp.sqrt(cov[:,0,0]);tm=basis.evaluate(reference[:,:2],tables.mean);ts=basis.evaluate(reference[:,:2],tables.std)[:,0]
        direct=acceleration_error_bound(mu,sigma,reference[:,2:]);interp=acceleration_error_bound(tm,ts,reference[:,2:])
        rows.append(dict(seed=i,endpoint=np.asarray(reference[-1,:2]).tolist(),maximum_y=float(reference[:,1].max()),
            min_clearance_m=float(jax.vmap(env.clearance)(reference).min()),
            direct_gp_max_std=float(sigma.max()),interpolated_max_std=float(ts.max()),
            maximum_std_discrepancy=float(jnp.max(jnp.abs(sigma-ts))),
            direct_error_bound_m_s2=float(direct.max()),interpolated_error_bound_m_s2=float(interp.max()),
            training_cost=float(result[2].initial_field_costs[i].mean())))
    choices=[]
    for downstream in (False,True):
        learner=ReachableLearning(env,grid,cfg,basis,'mask',global_goals=True,vector_precision=.2,
            prefix_screen='braking-feasible',learned_route_bounds=True,diversify_stalls=True,route_precision=True,downstream_information=downstream)
        actual=np.asarray(r['observations'])[:,1:3];hm=np.asarray(basis.evaluate(jnp.asarray(actual),tables.mean));hs=np.asarray(basis.evaluate(jnp.asarray(actual),tables.std))[:,0]
        _,bounds,_=measured_route_mask(actual,hm,hs,actual,cfg.umax)
        learner.policy.history=[(p.copy(),float(f)) for p,f in zip(actual,bounds)]
        learner.task_reference=np.asarray(audit['learning_reference'])
        lm,lc=predict(x[None,:2]);ts=time.perf_counter()
        path,decision,weights,priority=learner.plan(np.asarray(x),np.asarray(q),b,cache,tables,300.,np.asarray(lm[0]),float(jnp.sqrt(lc[0,0,0])))
        choices.append(dict(downstream=downstream,seconds=time.perf_counter()-ts,target=decision['target'],
            reason=decision['reason'],path=path.tolist(),task_weights=weights.tolist()))
    report=dict(source=str(a.recording),initializations=metadata,reference_diagnostics=rows,
        uncertainty_blocked_improvement=float(audit['uncertainty_blocked_improvement']),
        learning_requested=bool(audit['request_targeted_learning']),reference=np.asarray(audit['learning_reference']).tolist(),choices=choices)
    print(json.dumps(rows,indent=2));print(json.dumps([{k:v for k,v in c.items() if k not in ('path','task_weights')} for c in choices],indent=2))
    a.output.write_text(json.dumps(report,indent=2,default=lambda v:np.asarray(v).tolist()))
    signal.alarm(0)


if __name__=='__main__':main()
