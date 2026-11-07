"""Offline validation of learned-wind MBD; truth never enters optimization.

Frozen recorded POAM posteriors are evaluated at predeclared robot states.
The true plant is used only after selecting a trajectory, to assess forecast
accuracy. No hypothetical observation or posterior update is performed.
"""
import argparse, gzip, json, signal, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import full_learning_monitoring as model
import efficient_learning_monitoring as efficient
import cluster_environment as ce
from cached_poam_fields import CachedFields
from benchmark_recorded_planner import restore_inference


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('recordings',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--budgets',default='2:14,3:21')
    parser.add_argument('--scenarios',default='1,2')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded 360-second offline probe')))
    signal.alarm(360);reports=[]
    for budget in args.budgets.split(','):
        iterations,proposals=map(int,budget.split(':'))
        cfg=replace(model.settings(),iterations=iterations,proposals=proposals)
        pcfg=replace(model.legacy.physical_settings(cfg),safety_margin=cfg.safety_margin)
        env,grid,spec=ce.environment(pcfg,model.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
        basis,points,shape=model.table_grid(env.bounds,.1)
        _,extended,_=model.table_grid(env.bounds,.4)
        outside=np.any((np.asarray(extended)<np.asarray(env.bounds[0]))|(np.asarray(extended)>np.asarray(env.bounds[1])),axis=1)
        support=jnp.asarray(np.vstack((np.asarray(env.cells),np.asarray(extended)[outside])))
        cache=CachedFields(points,shape,cfg,grid['all_cells'])
        with jax.experimental.disable_x64():
            planner=efficient.optimizer(model.legacy.float32(env),cfg,model.legacy.float32(basis),True)
        execute=model.execution_block(env,cfg,pcfg,basis,seconds=cfg.horizon)
        for scenario in map(int,args.scenarios.split(',')):
            source=args.recordings/f's{scenario}-center-baseline1.json.gz'
            r=json.load(gzip.open(source));b=restore_inference(source.with_name(source.name.replace('.json.gz','-posterior.npz')),support)
            truth=model.m.physical.load_wind(scenario,model.m.physical.SITE/'revised')
            q=jnp.asarray(r['clarity_frames'][-1]['q'])
            states=[('recorded final state',np.array(r['samples'][-1][1:5]))]
            states += [('region '+reg['label']+' center',np.r_[np.mean(reg['bounds'],axis=0),[0.,0.]]) for reg in spec['regions']]
            for index,(label,state) in enumerate(states):
                x=jnp.asarray(state);key=jax.random.PRNGKey(55001+index+scenario*100)
                start=time.perf_counter();tables=cache.build(b,jnp.array(len(r['observations'])),key,jax.random.PRNGKey(88123));jax.block_until_ready(tables)
                field_s=time.perf_counter()-start;start=time.perf_counter()
                seeds,previous,_=model.legacy.seed_bank(x,q,env,cfg,None);geometry_s=time.perf_counter()-start
                start=time.perf_counter()
                with jax.experimental.disable_x64():
                    theta,cost,search,forecast,validation=planner(*model.legacy.float32((x,q,tables,seeds,previous)),key)
                    jax.block_until_ready(theta)
                plan_s=time.perf_counter()-start
                # Evaluate the selected command sequence against the unchanged
                # actual wind/plant, with the same learned controller and RPCBF.
                refs,aa=model.legacy.old.nominal_reference(theta.astype(jnp.float64),x,cfg)
                start=time.perf_counter();end,rows=execute(x,q,refs,aa,jnp.array(0.),tables,truth,jnp.array(True));jax.block_until_ready(end)
                xs,qs,us,nom,slack,correction,feasible,gaps,costs,alive,hits=map(np.asarray,rows)
                actual=np.vstack((state,xs[4::5]));predicted=np.asarray(forecast.x).mean(axis=0)
                error=np.linalg.norm(actual[:,:2]-predicted[:,:2],axis=1)
                actual_cost=float(costs.sum());predicted_cost=float(np.asarray(forecast.monitor).mean())
                relative_cost_error=abs(actual_cost-predicted_cost)/max(actual_cost,.05)
                displacement=float(np.max(np.linalg.norm(xs[:,:2]-state[:2],axis=1)))
                row=dict(scenario=scenario,budget=budget,state=label,posterior_rmse_m_s=r['summary']['final_rmse_m_s'],
                    startup_included=scenario==1 and index==0,field_s=field_s,geometry_s=geometry_s,
                    optimization_s=plan_s,trajectory_request_s=field_s+geometry_s+plan_s,
                    search_s=float(validation['search_s']),validation_s=float(validation['validation_s']),
                    offline_truth_check_s=time.perf_counter()-start,
                    search_evaluations=int(search.evaluations),accepted=not bool(validation['no_feasible']),
                    true_contact=not bool(end[-1]),true_infeasible_steps=int((~feasible).sum()),
                    true_minimum_clearance_m=float(gaps.min()),true_displacement_m=displacement,
                    position_rmse_m=float(np.sqrt(np.mean(error**2))),max_position_error_m=float(error.max()),
                    predicted_monitoring_cost=predicted_cost,true_monitoring_cost=actual_cost,
                    relative_cost_error=relative_cost_error,
                    pass_accuracy=bool(np.sqrt(np.mean(error**2))<=.25 and relative_cost_error<=.10),
                    pass_useful_execution=bool(not validation['no_feasible'] and end[-1] and feasible.all() and displacement>=.5))
                reports.append(row);print(json.dumps(row),flush=True)
                (args.output/'report.json').write_text(json.dumps(reports,indent=2,allow_nan=False))
    signal.alarm(0)


if __name__=='__main__':main()
