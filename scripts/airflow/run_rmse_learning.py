"""Executed uncertainty-waypoint learning and learn-then-MBD, in new files only."""
import argparse,gzip,hashlib,json,signal,time
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import jax
import jax.numpy as jnp
import rmse_monitoring as im
import cluster_environment as ce
from uncertainty_waypoints import WaypointPolicy
from waypoint_poam import condition,train
from rmse_gate import wind_vector_rmse,ready_for_mbd
from run_monitoring_mbd import write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=300)
    p.add_argument('--survey-seconds',type=int,default=300)
    p.add_argument('--rmse-threshold',type=float,default=.2)
    p.add_argument('--modes',default='baseline1')
    p.add_argument('--scenarios',default='1,2')
    p.add_argument('--cases',default='center')
    p.add_argument('--wall-limit',type=int,default=360)
    p.add_argument('--plan-limit',type=float,default=1.)
    p.add_argument('--train-kernel',action='store_true')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise RuntimeError('Existing outputs are preserved; choose a fresh directory')
    a.output.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Wall-time budget reached')))
    signal.alarm(a.wall_limit)
    cfg=im.settings();pcfg=replace(im.legacy.physical_settings(cfg),mission_seconds=a.seconds,
                                 safety_margin=cfg.safety_margin,inner_samples=cfg.construction_worlds)
    specpath=im.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json'
    site=im.m.physical.SITE/'revised';env,grid,spec=ce.environment(pcfg,specpath)
    basis,points,shape=im.table_grid(env.bounds);initial=im.initial_belief(env,points)
    sampler=im.field_builder(points,shape,cfg);safety_sampler=im.safety_builder(points,shape,cfg)
    execute=im.execution_block(env,cfg,pcfg,basis,seconds=.2)
    with jax.experimental.disable_x64():optimizer=im.optimizer(im.legacy.float32(env),cfg,im.legacy.float32(basis),True)
    @jax.jit
    def map_belief(b):return b.predict(jnp.asarray(grid['all_cells']))
    write(a.output/'environment.json',dict(grid=grid['grid'],dx_m=.125,free_indices=grid['free_indices'],
        cells=grid['cells'],membership=grid['membership'],targets=env.targets,process_noise=env.process_noise,
        initial_clarity=pcfg.initial_q,bounds=env.bounds,obstacles=env.obstacles,regions=spec['regions']))
    files=list((im.m.physical.ROOT/'scripts/airflow').glob('*.py'))+list((im.m.physical.ROOT/'src/voi_diffusion_ipp').rglob('*.py'))
    hashes={str(f.relative_to(im.m.physical.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    compiled=set();reports=[]
    cases=[c for c in ce.CASES if c['id'] in a.cases.split(',')]
    for scenario,case,mode in ((s,c,m) for s in map(int,a.scenarios.split(',')) for c in cases for m in a.modes.split(',')):
        ident=f's{scenario}-{case["id"]}-{mode}';began=time.perf_counter()
        truth=im.m.physical.load_wind(scenario,site) # only plant, actual sensor and offline diagnostics
        x,q=ce.initial(case,env,grid,pcfg);b=initial;mask=jnp.zeros(len(b.mean),bool)
        policy=WaypointPolicy(env);alive=jnp.array(True);clock=0.;cost=0.;contact=None;stop=None
        samples=[];qframes=[];bframes=[];plans=[];observations=[];updates=[];timings=[]
        noise=np.random.default_rng(case['seed']).normal(size=(round(a.seconds/.2)+1,2))
        previous=None;last_update=0.;path=np.asarray(x[:2])[None];handoff=None;gate_time=None;last_rmse=None
        adam_m=jax.tree.map(jnp.zeros_like,b.basis.parameters);adam_v=adam_m;adam_step=jnp.array(0)
        train_rng=np.random.default_rng(case['seed']+441)
        protocol=dict(id=ident,scenario=scenario,method=mode,initial_condition=case,settings=asdict(cfg),
            physical_settings=asdict(pcfg),source_sha256=hashes,initial_measurements=0,
            survey_seconds=a.seconds if mode=='survey' else a.survey_seconds,total_seconds=a.seconds,
            policy='Highest posterior variance/(1+geometric path length / 1m), persistent waypoints; actual-progress timeout with 30s cooldown; .4m/s waypoint reference; edges require exact mean-wind compensation <1.2m/s2 per component, leaving .4m/s2 authority for feedback. Full-uncertainty RPCBF on every command.',
            handoff='Oracle diagnostic RMSE threshold; state, velocity, clarity and actual POAM retained; no synthetic training data.',
            rmse_threshold_m_s=a.rmse_threshold,rmse_domain='All collision-free grid cells, vector error norm',learning_limit_s=a.survey_seconds,
            mbd='Original independent MBD chains, 15s, 43 search evaluations plus 7 mode-leader rescoring evaluations; unattenuated POAM posterior, 2 training and 2 independent validation fields; optimized alternate mode leader validated.',
            safety='Full-posterior RPCBF plus robust relative-degree-two wall constraints, rate 1.5/s; hard QP, logged minimum-slack fallback; no global guarantee.',
            sensor_period_s=.2,control_period_s=.05,integration_period_s=.01,waypoint_replan_s=.2,
            prior_std_m_s=1.8,noise_std_m_s=.1,inducing=128,inducing_allocation='Original POAM pivoted Cholesky over old inducing points plus actual new measurement positions; exact permutation transport for unchanged point sets.',
            kernel_training='4 existing variational EM Adam steps every 2s; actual observations only' if a.train_kernel else 'fixed',
            truth_access='Plant, actual local noisy sensor, displayed RMSE and phase-gate boolean; never planner or controller',device=str(jax.devices()))
        write(a.output/f'{ident}-protocol.json',protocol)
        def append(t,z,qq,u,nom,slack,correction,feasible,gap,living):
            membership=grid['membership']
            samples.append([t,*z,*u,*nom,float(slack),float(correction),int(feasible),float(gap),
                float(np.maximum(np.asarray(env.targets)-qq,0).mean()),cost,
                float(qq[membership[:,0]].mean()),float(qq[membership[:,1]].mean()),float(qq.mean()),int(living)])
            if abs(t-round(t))<1e-7:qframes.append(dict(t=t,q=np.round(qq,6).tolist()))
        def belief_frame(mu=None,cov=None):
            nonlocal last_rmse
            if mu is None:mu,cov=map_belief(b);jax.block_until_ready(mu)
            var=np.asarray(cov)[:,0,0];mean=np.asarray(mu);free=grid['free_indices']
            # Oracle error supplies only the phase switch, never ranking, inference or control.
            actual=np.asarray(truth.at(jnp.asarray(grid['all_cells'])));err=mean[free]-actual[free]
            last_rmse=wind_vector_rmse(mean,actual,free)
            bframes.append(dict(t=clock,mean=np.round(mean,5).tolist(),std=np.round(np.sqrt(var),5).tolist(),
                observations=len(observations),mean_variance=float(var[free].mean()),
                mean_std=float(np.sqrt(var[free]).mean()),rmse_m_s=last_rmse,
                protected_inducing=int(mask.sum())))
        def publish():
            arr=np.asarray(samples);visits={r['label']:next((float(s[0]) for s in arr if np.all(s[1:3]>=r['bounds'][0]) and np.all(s[1:3]<=r['bounds'][1])),None) for r in spec['regions']}
            warm=[r['wall_seconds'] for r in plans if not r['compilation_included']]
            summary=dict(rmse_threshold_m_s=a.rmse_threshold,gate_reached=gate_time is not None,gate_time_s=gate_time,completed=clock==a.seconds,simulated_seconds=clock,planned_seconds=a.seconds,
                contact=contact,stop_reason=stop,integrated_deficit=cost,min_clearance_m=float(arr[:,12].min()),
                infeasible_execution_steps=int((arr[1:,11]==0).sum()),max_execution_slack=float(arr[:,9].max()),
                max_control_component_m_s2=float(np.abs(arr[:,5:7]).max()),first_visit_s=visits,
                actual_measurements=len(observations),mean_variance_initial=bframes[0]['mean_variance'],
                mean_variance_final=bframes[-1]['mean_variance'],variance_reduction_fraction=1-bframes[-1]['mean_variance']/bframes[0]['mean_variance'],
                initial_rmse_m_s=bframes[0]['rmse_m_s'],final_rmse_m_s=bframes[-1]['rmse_m_s'],
                wall_seconds=time.perf_counter()-began,warm_planning_median_s=float(np.median(warm)) if warm else None,
                warm_planning_max_s=max(warm) if warm else None)
            rec=dict(id=ident,scenario=scenario,method=mode,samples=np.round(arr,7).tolist(),clarity_frames=qframes,
                belief_frames=bframes,plans=plans,observations=observations,updates=updates,execution_timing=timings,
                handoff=handoff,summary=summary)
            def clean(v):
                if isinstance(v,dict):return {k:clean(z) for k,z in v.items()}
                if isinstance(v,(list,tuple)):return [clean(z) for z in v]
                if isinstance(v,np.ndarray):return clean(v.tolist())
                if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
                return v
            payload=json.dumps(clean(rec),separators=(',',':'),allow_nan=False,default=lambda v:np.asarray(v).tolist()).encode()
            temporary=a.output/f'{ident}.json.gz.tmp'
            with gzip.GzipFile(temporary,'wb',mtime=0) as f:f.write(payload)
            temporary.replace(a.output/f'{ident}.json.gz')
            return summary
        def observe():
            nonlocal b,mask,last_update,adam_m,adam_v,adam_step
            y=np.asarray(truth.at(x[:2]))[0]+.1*noise[round(clock/.2)]
            observations.append([clock,*np.asarray(x[:2]),*y]);ts=time.perf_counter()
            b,mask=condition(b,x[None,:2],jnp.asarray(y)[None],mask);jax.block_until_ready(b)
            if a.train_kernel and clock>0 and abs(clock/2-round(clock/2))<1e-8:
                data=np.asarray(observations);idx=train_rng.integers(0,len(data),size=32)
                b,adam_m,adam_v,adam_step,loss=train(b,jnp.asarray(data[idx,1:3]),jnp.asarray(data[idx,3:5]),jnp.array(len(data),dtype=jnp.float64),adam_m,adam_v,adam_step);jax.block_until_ready(b)
            last_update=time.perf_counter()-ts;updates.append(dict(t=clock,seconds=last_update,compilation_included='update' not in compiled));compiled.add('update')
        append(0.,np.asarray(x),np.asarray(q),np.zeros(2),np.zeros(2),0.,0.,True,env.clearance(x),True)
        belief_frame();observe()
        tables=sampler(b,jnp.array(len(observations)),jax.random.PRNGKey(case['seed']+1000),jax.random.PRNGKey(case['seed']+271828));jax.block_until_ready(tables)
        for block in range(round(a.seconds/.2)):
            if block%10==0:
                belief_frame()
                if mode!='survey' and gate_time is None and ready_for_mbd(last_rmse,a.rmse_threshold):gate_time=clock
            learning=gate_time is None or mode=='survey'
            if learning and clock>=a.survey_seconds:
                stop='Learning accuracy threshold not reached; MBD has not started';break
            if learning:
                ts=time.perf_counter();mu,cov=map_belief(b);jax.block_until_ready(mu)
                path,decision=policy.plan(np.asarray(x),np.asarray(cov)[:,0,0],clock,np.asarray(mu))
                total=time.perf_counter()-ts+last_update
                plans.append(dict(t=clock,phase='learning',wall_seconds=total,compilation_included=block==0,
                    path=path.tolist(),**decision))
                if block%10==0:belief_frame(mu,cov)
                ref=policy.reference(np.asarray(x),path)
                refs=jnp.tile(jnp.asarray(ref),(cfg.steps+1,1));aa=jnp.zeros((cfg.steps,2));offset=0.
            else:
                if handoff is None:
                    handoff=dict(t=clock,state=np.asarray(x).tolist(),clarity=np.asarray(q).tolist(),observations=len(observations),
                                 inducing=np.asarray(b.basis.centers).tolist(),posterior_mean=np.asarray(b.mean).tolist(),
                                 posterior_covariance=np.asarray(b.covariance).tolist())
                    np.savez_compressed(a.output/f'{ident}-handoff.npz',state=np.asarray(x),clarity=np.asarray(q),
                        mean=np.asarray(b.mean),covariance=np.asarray(b.covariance),a=np.asarray(b.a),B=np.asarray(b.B),
                        inducing=np.asarray(b.basis.centers),mask=np.asarray(mask))
                if round((clock-gate_time)*5)%25==0:
                    ts=time.perf_counter();tables=sampler(b,jnp.array(len(observations)),jax.random.PRNGKey(case['seed']+1000+block),jax.random.PRNGKey(case['seed']+271828));jax.block_until_ready(tables)
                    field_s=time.perf_counter()-ts;ts=time.perf_counter();seeds,shifted,metadata=im.legacy.seed_bank(x,q,env,cfg,previous);geometry_s=time.perf_counter()-ts
                    args=im.legacy.float32((x,q,tables,seeds,shifted));ts=time.perf_counter()
                    with jax.experimental.disable_x64():result=optimizer(*args,jax.random.PRNGKey(case['seed']+2000+block));jax.block_until_ready(result)
                    mbd_s=time.perf_counter()-ts;theta,score,search,forecast,validation=result
                    first='mbd' not in compiled;compiled.add('mbd');total=field_s+geometry_s+mbd_s+last_update
                    plans.append(dict(t=clock,phase='monitoring',wall_seconds=total,compilation_included=first,
                        timing=dict(fields=field_s,geometry=geometry_s,optimization=mbd_s,posterior=last_update),
                        initialization=metadata,expected_cost=float(score),evaluations=int(search.evaluations)+int(validation['leader_reevaluations']),
                        initial_cost=float(search.initial_cost),optimized_training_cost=float(np.min(np.asarray(validation['leader_costs']))),
                        optimization_trace=np.asarray(search.trace).tolist(),
                        selected_parameters=np.asarray(theta).tolist(),
                        mode_cost_trace=np.asarray(search.mode_trace).tolist(),
                        predicted_states=np.asarray(forecast.x).tolist(),validation={k:np.asarray(v).tolist() for k,v in validation.items()}))
                    if bool(validation['no_feasible']) or not np.isfinite(float(score)) or (not first and total>a.plan_limit):
                        stop='No validated feasible MBD plan' if bool(validation['no_feasible']) else 'No finite MBD plan' if not np.isfinite(float(score)) else 'Warm planning budget exceeded';break
                    refs,aa=im.legacy.old.nominal_reference(theta.astype(jnp.float64),x,cfg);previous=theta;plan_t=clock
                    belief_frame()
                offset=clock-plan_t
            ts=time.perf_counter();end,rows=execute(x,q,refs,aa,jnp.array(float(offset)),tables,truth,alive);jax.block_until_ready(end)
            timings.append(dict(t=clock,seconds=time.perf_counter()-ts,compilation_included='execute' not in compiled));compiled.add('execute')
            xs,qs,us,noms,slacks,corrections,feasible,gaps,costs,living,hits=map(np.asarray,rows)
            for k in range(len(xs)):
                t=round(clock+(k+1)*pcfg.execution_dt,8)
                if contact is None and np.isfinite(hits[k]):contact=dict(time_s=clock+k*pcfg.execution_dt+float(hits[k]),position_m=xs[k,:2].tolist())
                cost+=float(costs[k]);append(t,xs[k],qs[k],us[k],noms[k],slacks[k],corrections[k],feasible[k],gaps[k],living[k])
            x,q,alive=end;clock=round(clock+.2,8)
            if bool(alive):observe()
            tables=safety_sampler(b,tables,jax.random.PRNGKey(case['seed']+271828));jax.block_until_ready(tables)
            if clock%30==0:
                belief_frame();summary=publish()
                print(json.dumps(dict(event='progress',id=ident,t=clock,phase='learning' if learning else 'monitoring',
                    position=np.asarray(x[:2]).tolist(),A=summary['first_visit_s']['A'],contact=contact,rmse=last_rmse,threshold=a.rmse_threshold,
                    infeasible=summary['infeasible_execution_steps'],variance_reduction=summary['variance_reduction_fraction'],wall_s=round(time.perf_counter()-began,2))),flush=True)
            if contact is not None:stop='Contact: pilot/mission stopped, no further sensor data';break
        belief_frame();summary=publish();reports.append(summary|dict(id=ident,scenario=scenario,method=mode))
        write(a.output/'summary.json',reports)
    signal.alarm(0)


if __name__=='__main__':main()
