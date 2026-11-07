"""Bounded unknown-wind missions; separate output files, actual observations only."""
import argparse,gzip,hashlib,json,signal,time
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import jax
import jax.numpy as jnp
import implicit_monitoring_v5 as im
import cluster_environment as ce
from run_monitoring_mbd import write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=300)
    p.add_argument('--horizon',type=float,default=15.)
    p.add_argument('--scenarios',default='1,2')
    p.add_argument('--methods',default='aware,baseline')
    p.add_argument('--cases',default='center')
    p.add_argument('--wall-limit',type=int,default=300)
    p.add_argument('--plan-limit',type=float,default=1.)
    a=p.parse_args()
    if a.output.exists() and any(a.output.glob('s*-*.json*')):raise RuntimeError('Existing experiment output is preserved; choose a new directory')
    a.output.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Experiment wall-time limit reached')))
    signal.alarm(a.wall_limit)
    cfg=replace(im.settings(),horizon=a.horizon);pcfg=replace(im.legacy.physical_settings(cfg),mission_seconds=a.seconds,
                                 safety_margin=cfg.safety_margin,inner_samples=cfg.construction_worlds)
    specpath=im.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json'
    site=im.m.physical.SITE/'revised'
    env,grid,spec=ce.environment(pcfg,specpath)
    basis,points,shape=im.table_grid(env.bounds)
    initial=im.initial_belief(env,points)
    sampler=im.field_builder(points,shape,cfg)
    safety_sampler=im.safety_builder(points,shape,cfg)
    @jax.jit
    def map_belief(b):return b.predict(jnp.asarray(grid['all_cells']))
    execute=im.execution_block(env,cfg,pcfg,basis,seconds=.2)
    with jax.experimental.disable_x64():
        planners={meth:im.optimizer(im.legacy.float32(env),cfg,im.legacy.float32(basis),meth=='aware') for meth in a.methods.split(',')}
    cases=[c for c in ce.CASES if c['id'] in a.cases.split(',')]
    assert len(cases)==len(a.cases.split(','))
    write(a.output/'environment.json',dict(grid=grid['grid'],dx_m=.125,free_indices=grid['free_indices'],
        cells=grid['cells'],membership=grid['membership'],targets=env.targets,process_noise=env.process_noise,
        initial_clarity=pcfg.initial_q,bounds=env.bounds,obstacles=env.obstacles,regions=spec['regions'],environment_spec=spec))
    hashfiles=[Path(__file__),Path(im.__file__),Path(ce.__file__),Path(im.legacy.__file__),
        Path(im.__file__).with_name('monitoring_routes.py'),Path(im.__file__).with_name('supported_poam.py'),
        Path(im.__file__).with_name('observed_poam.py'),Path(im.core.__file__),Path(im.previous.__file__),Path(im.predecessor.__file__),Path(im.__file__).with_name("supported_poam_identity.py")]
    hashfiles+=list((im.m.physical.ROOT/'src/voi_diffusion_ipp').rglob('*.py'))
    hashes={str(f.relative_to(im.m.physical.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in hashfiles}
    compiled=set();sample_compiled=False;execution_compiled=False;condition_compiled=False
    reports=[]
    for scenario,case,method in ((s,c,meth) for s in map(int,a.scenarios.split(',')) for c in cases for meth in planners):
        ident=f's{scenario}-{case["id"]}-{method}';began=time.perf_counter()
        truth=im.m.physical.load_wind(scenario,site)  # plant/sensor boundary only
        x,q=ce.initial(case,env,grid,pcfg);b=initial;mask=jnp.zeros(len(b.mean),bool)
        alive=jnp.array(True);previous=None;clock=0.;cost=0.;contact=None;stop=None
        samples=[];qframes=[];bframes=[];plans=[];observations=[];executions=[];updates=[]
        noise=np.random.default_rng(case['seed']).normal(size=(round(a.seconds/.2)+1,2))
        last_update=0.;last_update_compiled=False
        protocol=dict(id=ident,scenario=scenario,method=method,initial_condition=case,
            initial_wind_knowledge='none',initial_measurements=0,initial_survey=False,
            settings=asdict(cfg),physical_settings=asdict(pcfg),source_sha256=hashes,
            environment_spec=spec,wind_sha256=hashlib.sha256((site/f'data/scenario{scenario}/stationary-velocity.f32').read_bytes()).hexdigest(),
            performance_distribution='Zero before any observation. Afterwards r*mu + sqrt(r)*(posterior_draw-mu), r=max(0,1-var_post/var_prior). This planning distribution differs from full POAM; safety uses full posterior.',
            version='v5-supported-planning-mean-exact-identity-transport',safety='Full POAM mean and unattenuated coherent construction fields; original spline backup policy value; max(sampled force support, tighter certified force bound over 3*sqrt(posterior_var+sensor_noise_var) wind-radius ball). Hard execution QP retains actuator limits and logs minimum-slack fallback.',
            safety_limit='Pointwise model confidence bound and finite-sample backup value, not a deterministic global safety certificate or guaranteed coverage of the true wind.',
            prediction_filter='relaxed RPCBF search; independent hard execution-resolution validation, two finalists plus hold fallback if both fail, full safety uncertainty' if method=='aware' else 'none; still-air baseline',
            observation_period_s=.2,posterior_update_period_s=.2,replan_s=5.,local_measurement_before_first_command=True,
            poam=dict(inducing=128,allocation='Existing observation-supported allocation; full transported sufficient statistics; capacity saturation reported',kernel='existing fixed attentive kernel',prior_std=1.8,noise_std=.1,spectral_nodes=32),
            numerical_field_cache=dict(spacing_m=.2,bounds_padding_m=1.,interpolation='Existing C1 Catmull-Rom, full posterior sampling with conditional residual'),
            no_hypothetical_updates=True,no_exploration_reward=True,no_exploration_mode=True,
            common_random_fields_per_call=True,performance_fields=cfg.worlds,safety_fields=cfg.construction_worlds,
            device=str(jax.devices()),runtime_limit_s=a.plan_limit,
            timing_scope='Geometry + field sampling + MBD and selected forecasts + most recent actual POAM update. First JIT calls reported separately; execution and between-plan safety refresh timed separately.')
        write(a.output/f'{ident}-protocol.json',protocol)
        def append(t,z,qq,u,nom,slack,correction,feasible,gap,living):
            membership=grid['membership']
            samples.append([t,*z,*u,*nom,float(slack),float(correction),int(feasible),float(gap),
                float(np.maximum(np.asarray(env.targets)-qq,0).mean()),cost,
                float(qq[membership[:,0]].mean()),float(qq[membership[:,1]].mean()),float(qq.mean()),int(living)])
            if abs(t/.5-round(t/.5))<1e-7:qframes.append(dict(t=t,q=np.round(qq,6).tolist()))
        def belief_frame():
            mu,cov=map_belief(b);jax.block_until_ready(mu)
            var=np.asarray(cov)[:,0,0];r=np.asarray(im.information_support(var,float(jnp.exp(b.basis.parameters['log_amplitude'])),len(observations)))
            bframes.append(dict(t=clock,mean=np.round(mu,5).tolist(),std=np.round(np.sqrt(var),5).tolist(),
                planning_std=np.round(np.sqrt(r*var),5).tolist(),support=np.round(r,5).tolist(),
                observations=len(observations),protected_inducing=int(mask.sum()),planning_mean=np.round(np.asarray(mu)*r[:,None],5).tolist()))
        def publish(completed):
            arr=np.asarray(samples);visits={r['label']:next((float(s[0]) for s in arr if np.all(s[1:3]>=r['bounds'][0]) and np.all(s[1:3]<=r['bounds'][1])),None) for r in spec['regions']}
            warm=[r['wall_seconds'] for r in plans if not r['timing']['compilation_included']]
            summary=dict(completed=completed,simulated_seconds=clock,planned_seconds=a.seconds,contact=contact,stop_reason=stop,
                integrated_deficit=cost,min_clearance_m=float(arr[:,12].min()),infeasible_execution_steps=int((arr[:,11]==0).sum()),
                max_execution_slack=float(arr[:,9].max()),max_control_component_m_s2=float(np.abs(arr[:,5:7]).max()),
                first_visit_s=visits,first_transfer=ce.first_transfer(arr,spec),passage_crossings=ce.crossings(arr,spec),
                actual_measurements=len(observations),protected_inducing=int(mask.sum()),capacity_saturated=bool(mask.all()),
                wall_seconds=time.perf_counter()-began,warm_planning_median_s=float(np.median(warm)) if warm else None,
                warm_planning_max_s=max(warm) if warm else None)
            rec=dict(id=ident,scenario=scenario,method=method,initial_condition=case,samples=np.round(arr,7).tolist(),
                clarity_frames=qframes,belief_frames=bframes,plans=plans,observations=observations,
                execution_timing=executions,belief_update_timing=updates,summary=summary)
            def clean(v):
                if isinstance(v,dict):return {k:clean(z) for k,z in v.items()}
                if isinstance(v,(list,tuple)):return [clean(z) for z in v]
                if isinstance(v,np.ndarray):return clean(v.tolist())
                if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
                return v
            payload=json.dumps(clean(rec),separators=(',',':'),allow_nan=False,
                default=lambda value:np.asarray(value).tolist()).encode()
            temporary=a.output/f'{ident}.json.gz.tmp'
            with gzip.GzipFile(temporary,'wb',mtime=0) as f:f.write(payload)
            temporary.replace(a.output/f'{ident}.json.gz')
            return summary
        append(0.,np.asarray(x),np.asarray(q),np.zeros(2),np.zeros(2),0.,0.,True,env.clearance(x),True)
        belief_frame()
        for cycle in range(round(a.seconds/cfg.commit)):
            t0=time.perf_counter()
            tables=sampler(b,jnp.asarray(len(observations),jnp.int32),jax.random.PRNGKey(case['seed']+1000+cycle),jax.random.PRNGKey(case['seed']+271828));jax.block_until_ready(tables)
            sampling_s=time.perf_counter()-t0;first=method not in compiled or not sample_compiled
            sample_compiled=True
            ts=time.perf_counter();seeds,shifted,metadata=im.legacy.seed_bank(x,q,env,cfg,previous);geometry_s=time.perf_counter()-ts
            args=im.legacy.float32((x,q,tables,seeds,shifted));ts=time.perf_counter()
            with jax.experimental.disable_x64():result=planners[method](*args,jax.random.PRNGKey(case['seed']+2000+cycle));jax.block_until_ready(result)
            mbd_s=time.perf_counter()-ts;total=time.perf_counter()-t0+last_update
            compiled.add(method);theta,score,search,forecast,validation=result
            origin='initialization' if np.any(np.all(np.asarray(search.initial_parameters)==np.asarray(theta)[None],axis=1)) else 'diffusion'
            planrow=dict(t=clock,wall_seconds=total,timing=dict(initialization=geometry_s,field_sampling=sampling_s,
                optimization_validation=mbd_s,mbd_search=float(validation["search_s"]),execution_validation=float(validation["validation_s"]),actual_posterior_update=last_update,compilation_included=first or last_update_compiled),
                validation={k:np.asarray(v).tolist() for k,v in validation.items()},observations=len(observations),initialization=metadata,expected_cost=float(score),selected_origin=origin,
                evaluations=int(search.evaluations),seed_costs=np.asarray(search.initial_field_costs).mean(axis=1).tolist(),
                seed_costs_by_world=np.asarray(search.initial_field_costs).tolist(),trace=np.asarray(search.trace).tolist(),
                predicted_states=np.asarray(forecast.x).tolist(),predicted_contacts=np.asarray(forecast.terminated).tolist(),
                predicted_monitoring_cost=np.asarray(forecast.monitor).tolist(),predicted_safety_cost=np.asarray(forecast.penalty).tolist(),
                planning_field_max_abs=float(jnp.max(jnp.abs(tables.planning))),safety_prior_std_max=float(jnp.max(tables.std)),
                forecast_transfer=ce.first_transfer(np.c_[clock+cfg.dt*np.arange(cfg.steps+1),np.asarray(forecast.x)[0]],spec))
            plans.append(planrow)
            np.savez_compressed(a.output/f'{ident}-plan-{cycle:03d}.npz',theta=np.asarray(theta),initial_state=np.asarray(x),initial_clarity=np.asarray(q),
                validation_fields=np.asarray(tables.validation),planning_fields=np.asarray(tables.planning),safety_fields=np.asarray(tables.safety),mean=np.asarray(tables.mean),std=np.asarray(tables.std),support=np.asarray(tables.support),
                initial_parameters=np.asarray(search.initial_parameters),initial_field_costs=np.asarray(search.initial_field_costs),
                inducing=np.asarray(b.basis.centers),posterior_mean=np.asarray(b.mean),posterior_covariance=np.asarray(b.covariance),observations=len(observations))
            print(json.dumps(dict(event='plan',id=ident,t=clock,total_s=round(total,4),compile=first,cost=float(score),observations=len(observations))),flush=True)
            if not np.isfinite(float(score)) or (not first and not last_update_compiled and total>a.plan_limit):
                stop='No finite plan' if not np.isfinite(float(score)) else 'Planning time exceeded one-second budget'
                publish(False);break
            if cycle==0:
                # A normal local sensor reading before the first command, after
                # the required zero-wind initial planning call. No presurvey.
                y=np.asarray(truth.at(x[:2]))[0]+.1*noise[0]
                observations.append([0.,*np.asarray(x[:2]),*y])
                ts=time.perf_counter()
                b,mask=im.condition_supported(b,x[None,:2],jnp.asarray(y)[None],mask);jax.block_until_ready(b)
                updates.append(dict(t=0.,seconds=time.perf_counter()-ts,observations=1,compilation_included=not condition_compiled))
                condition_compiled=True
                tables=safety_sampler(b,tables,jax.random.PRNGKey(case['seed']+271828));jax.block_until_ready(tables)
            refs,aa=im.legacy.old.nominal_reference(theta.astype(jnp.float64),x,cfg);previous=theta
            for block in range(round(cfg.commit/.2)):
                second=round(block*.2,8)
                ts=time.perf_counter();end,rows=execute(x,q,refs,aa,jnp.array(float(second)),tables,truth,alive);jax.block_until_ready(end)
                executions.append(dict(t=clock,seconds=time.perf_counter()-ts,compilation_included=not execution_compiled));execution_compiled=True
                xs,qs,us,noms,slacks,corrections,feasible,gaps,costs,living,hits=map(np.asarray,rows)
                actual_p=[];actual_y=[]
                for k in range(len(xs)):
                    t=round(clock+(k+1)*pcfg.execution_dt,8)
                    if contact is None and np.isfinite(hits[k]):contact=dict(time_s=clock+k*pcfg.execution_dt+float(hits[k]),position_m=xs[k,:2].tolist())
                    cost+=float(costs[k]);append(t,xs[k],qs[k],us[k],noms[k],slacks[k],corrections[k],feasible[k],gaps[k],living[k])
                    if living[k] and abs(t/.2-round(t/.2))<1e-7:
                        y=np.asarray(truth.at(xs[k,:2]))[0]+.1*noise[round(t/.2)]
                        actual_p.append(xs[k,:2]);actual_y.append(y);observations.append([t,*xs[k,:2],*y])
                x,q,alive=end;clock=round(clock+.2,8)
                last_update=0.;last_update_compiled=False
                if actual_p:
                    ts=time.perf_counter();b,mask=im.condition_supported(b,jnp.asarray(actual_p),jnp.asarray(actual_y),mask);jax.block_until_ready(b)
                    last_update=time.perf_counter()-ts;last_update_compiled=not condition_compiled;condition_compiled=True
                    updates.append(dict(t=clock,seconds=last_update,observations=len(actual_p),compilation_included=last_update_compiled))
                if block<round(cfg.commit/.2)-1:
                    ts=time.perf_counter();tables=safety_sampler(b,tables,jax.random.PRNGKey(case['seed']+271828));jax.block_until_ready(tables)
                    updates.append(dict(t=clock,safety_refresh_seconds=time.perf_counter()-ts,observations=0,compilation_included=False))
            belief_frame()
            if clock%30==0 or clock==a.seconds:
                summary=publish(clock==a.seconds)
                print(json.dumps(dict(event='progress',id=ident,t=clock,x=np.asarray(x[:2]).tolist(),first_A=summary['first_visit_s']['A'],contact=contact,wall_s=round(time.perf_counter()-began,2))),flush=True)
        summary=publish(clock==a.seconds);reports.append(summary|dict(id=ident,scenario=scenario,method=method,case=case['id']))
        write(a.output/'summary.json',reports)
    signal.alarm(0)


if __name__=='__main__':main()
