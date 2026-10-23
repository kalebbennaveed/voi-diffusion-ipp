"""MBD trajectory optimization experiments with optional geometric initialization.

Planning and execution clocks are separate. Each run records its complete
configuration, code hashes, decisions, predictions and raw optimization audit.
"""
import argparse,json,time,hashlib,os,platform
from functools import cache
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation
from voi_diffusion_ipp.quarter.environment import environment,initial_belief,wind,metadata,table_sampler,planning_sampler
from voi_diffusion_ipp.quarter.continuous import ControlSettings,optimizer,rollout,continuation,nominal_reference
from voi_diffusion_ipp.quarter.execution import executor
from voi_diffusion_ipp.quarter.rrt import demonstrations
from voi_diffusion_ipp.quarter import sensing
from voi_diffusion_ipp.quarter.diffusion import cosine_schedule
from voi_diffusion_ipp.quadrotor.config import VehicleConfig
from voi_diffusion_ipp.quadrotor.types import State
from voi_diffusion_ipp.quadrotor.dynamics import Quadrotor
from voi_diffusion_ipp.quadrotor.aerodynamics import infer_wind

REVISION='quarter-full-control-mbd-v1'
METHODS=('proposed','blind','blind_filter','aware','optimistic','fading','no_learning')


def optimism_fraction(cfg,method,clock,mission_seconds):
    if method!='fading':return cfg.optimism
    duration=cfg.optimism_anneal_seconds if cfg.optimism_anneal_seconds is not None else mission_seconds*.5
    return cfg.optimism+(1-cfg.optimism)*min(max(clock/duration,0.),1.)


def write(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,separators=(',',':'),allow_nan=False))
    temporary.replace(path)


@cache
def source_hashes():
    root=Path(__file__).resolve().parents[1]
    files=[Path(__file__),root/'src/voi_diffusion_ipp/clarity.py',root/'src/voi_diffusion_ipp/experiments/urban_quad.py',*root.glob('src/voi_diffusion_ipp/quarter/*.py'),*root.glob('src/voi_diffusion_ipp/quadrotor/*.py'),*root.glob('src/voi_diffusion_ipp/safety/*.py'),*root.glob('src/voi_diffusion_ipp/wind/*.py')]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def run(layout,case,method,seconds,seed,cfg,out,plan_only=False,audit_all=False,resume=None,engine=None):
    started=time.perf_counter();env=environment(layout,cfg.environment_scale,cfg.inspection_scale,case,cfg.geometry,cfg.detour_depth);vehicle=VehicleConfig()
    plan=optimizer(env,cfg,'proposed' if method=='no_learning' else method)
    blind=method in ('blind','blind_filter')
    b=initial_belief(cfg);sample=table_sampler(cfg.construction_worlds+cfg.worlds+cfg.validation_worlds,cfg.environment_scale,cfg.geometry)
    shared_mode_inputs=cfg.initialization=='modes'
    fast_sample=planning_sampler(cfg) if cfg.multimode or shared_mode_inputs else None
    controller_sample=table_sampler(cfg.construction_worlds,cfg.environment_scale,cfg.geometry) if cfg.multimode or shared_mode_inputs else sample
    condition=jax.jit(lambda bb,p,y:bb.condition(p,y,relocate=cfg.wind_relocate));prediction=jax.jit(lambda bb,p:bb.predict(p))
    observation_period=min(cfg.observation_period,cfg.commit)
    execution_cfg=replace(cfg,commit=observation_period) if cfg.online_estimation else cfg
    execute=executor(env,execution_cfg,lambda p:wind(p,case,cfg.wind_strength,cfg.environment_scale,cfg.geometry,cfg.wind_at_start),filter_on=method!='blind')
    predict=jax.jit(lambda th,x,q,m,c,f:rollout(th,x,q,m,c,f,env,cfg,not blind))
    actual_wind=jax.jit(lambda p:wind(p,case,cfg.wind_strength,cfg.environment_scale,cfg.geometry,cfg.wind_at_start))
    @jax.jit
    def monitoring(q,old_position,states):
        def step(carry,packed):
            q,p=carry
            qn=sensing.step(q,p,packed[:2],env,cfg,.01,packed[2])
            deficit=lambda qq:jnp.maximum(env.targets-qq,0).mean()
            cost=.005*(deficit(q)+deficit(qn))
            return (qn,packed[:2]),(qn,cost,env.clearance(jnp.r_[packed[:2],packed[3:5]])-.275)
        return jax.lax.scan(step,(q,old_position),states)[1]
    if engine is not None:
        plan=engine.planner(env,method)
        execute=engine.execution(env,case)
        predict=engine.prediction(env,blind)
        actual_wind=engine.wind(case)
        monitoring=engine.monitoring(env)
    key=jax.random.PRNGKey(seed);construction_key=jax.random.fold_in(key,271828);rng=np.random.default_rng(seed+9001)
    start_xy=(np.asarray((env.regions[0,:2]+env.regions[0,2:])/2) if cfg.geometry in ('compact','compact_passages','compact_detour','suite') else
              np.array([-28.*cfg.environment_scale,8.*cfg.environment_scale]))
    state=State(np.r_[start_xy,3.2],np.zeros(3),np.eye(3),np.zeros(3),np.full(4,np.sqrt(vehicle.mass*vehicle.gravity/(4*vehicle.thrust_coefficient))))
    plant=Quadrotor(vehicle,np.zeros(3),np.zeros(3));q=jnp.full(len(env.cells),.35,dtype=jnp.float32).at[:42].set(.65)
    if cfg.geometry=='suite':
        inside=jnp.all((env.cells>=env.regions[0,:2])&(env.cells<=env.regions[0,2:]),axis=1)
        q=jnp.where(inside,.65,.35).astype(jnp.float32)
    previous=jnp.zeros(2*cfg.steps)
    xx,zz=np.meshgrid(np.linspace(float(env.bounds[0,0]),float(env.bounds[1,0]),49),np.linspace(float(env.bounds[0,1]),float(env.bounds[1,1]),41));grid=np.c_[xx.ravel(),zz.ravel()].astype(np.float32)
    grid_wind=np.asarray(actual_wind(jnp.array(grid)))
    frames=[];beliefs=[];plans=[];observations=[];cost=0.;clock=0.;minimum=100.;failure=None;errors=[]
    target=out/f'{layout}_{case}_{method}_{seed}.json';hashes=source_hashes()
    def frame(reference):
        frames.append(dict(t=round(clock,4),p=np.round(state.position,5).tolist(),quat=np.round(Rotation.from_matrix(state.rotation).as_quat(),6).tolist(),q=np.asarray(q).round(5).tolist(),cost=cost,ref=np.asarray(reference).round(5).tolist(),velocity=state.velocity.round(5).tolist(),wind=np.asarray(actual_wind(jnp.array(state.position[None,:2])))[0].round(4).tolist()))
    frame(state.position)
    start_iteration=0;prefix_wall_seconds=0.;continuation_record=None
    if resume is not None:
        resume=Path(resume)
        old=json.loads(resume.read_text())
        if plan_only or method=='fading':raise ValueError('Continuation requires a mission-length-independent flight objective')
        if not old.get('success') or old.get('plan_only') or old.get('status') not in ('completed','running'):
            raise ValueError('Continue only an unterminated physical mission at a saved replanning boundary')
        if target.resolve()==resume.resolve():raise ValueError('Continuation must preserve the original recording in a different directory')
        if (old['layout'],old['case'],old['method'],old['seed'])!=(layout,case,method,seed):raise ValueError('Continuation identity mismatch')
        # JSON stores tuple-valued settings as lists. Compare their serialized
        # values so a faithful round-trip is accepted without allowing a model
        # or optimizer setting to change.
        if json.dumps(asdict(ControlSettings(**old['config'])),sort_keys=True)!=json.dumps(asdict(cfg),sort_keys=True):
            raise ValueError('Continuation must preserve every model and optimizer setting')
        if seconds<=old['completed_seconds']:raise ValueError('Continuation must extend the saved physical time')
        runner=str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[1]))
        for name,digest in old['source_hashes'].items():
            if name!=runner and hashes.get(name)!=digest:raise ValueError(f'Model source changed since the prefix: {name}')
        start_iteration=len(old['plans'])
        if abs(old['completed_seconds']-start_iteration*cfg.commit)>.001:raise ValueError('Prefix does not end at a replanning boundary')
        observations_per_commit=round(cfg.commit/min(cfg.observation_period,cfg.commit))
        if len(old['observations'])!=start_iteration*observations_per_commit:raise ValueError('Prefix must include every completed wind observation')
        # Restore the unrounded physical state and clarity from the execution
        # record, never from the rounded display frames.
        with np.load(resume.parent/old['plans'][-1]['execution_audit']) as last:
            state=State.unpack(last['states'][-1].astype(float));q=jnp.asarray(last['clarity'][-1])
        frames=old['frames'];beliefs=old['beliefs'];plans=old['plans'];observations=old['observations']
        clock=old['completed_seconds'];cost=old['cost'];minimum=old['min_clearance']
        previous=continuation(jnp.asarray(plans[-1]['policy'],jnp.float32),cfg)
        for start in range(0,len(observations),observations_per_commit):
            batch=observations[start:start+observations_per_commit]
            if method!='no_learning':
                groups=[[observation] for observation in batch] if cfg.online_estimation else [batch]
                for group in groups:b=condition(b,jnp.array([observation['position'] for observation in group],jnp.float32),jnp.array([observation['measurement'] for observation in group],jnp.float32))
            key,_,_=jax.random.split(key,3)
            rng.normal(size=(observations_per_commit,2))
        for prior_plan in plans:
            names=[prior_plan['execution_audit'],*prior_plan.get('chain_audits',[])]
            if prior_plan.get('audit'):names.append(prior_plan['audit'])
            for name in set(names):
                # Raw archives are immutable. Keep the exact bytes without
                # duplicating a potentially large checkpoint prefix.
                source=resume.parent/name;destination=out/name
                if not destination.exists():os.link(source,destination)
            with np.load(resume.parent/prior_plan['execution_audit']) as audit:
                errors.extend(float(np.linalg.norm(s[:3].astype(float)-r)) for s,r in zip(audit['states'],audit['references']))
        prefix_wall_seconds=old['wall_seconds']
        continuation_record=dict(source=str(resume.resolve()),source_sha256=hashlib.sha256(resume.read_bytes()).hexdigest(),
            original_mission_seconds=old['mission_seconds'],continued_at_seconds=clock,prefix_source_hashes=old['source_hashes'],
            restoration='Unrounded 100 Hz state and clarity; chronological observed posterior; original random-number streams; shifted previous trajectory')
        print(f'CONTINUING {resume} at {clock:g} s to {seconds:g} s',flush=True)
    if blind:score_description='zero-disturbance monitoring plus soft constraint penalties'
    elif method in ('proposed','optimistic','fading','no_learning'):
        score_description=('favorable-tail monitoring plus mean constraint penalty' if cfg.penalty_aggregation=='mean' else 'favorable-tail total cost')
    else:score_description='expected monitoring plus expected constraint penalty'
    def result(completed):
        return dict(revision=REVISION,completed=completed,plan_only=plan_only,status='planning_only' if plan_only else ('failed' if failure else ('completed' if completed else 'running')),layout=layout,case=case,method=method,seed=seed,mission_seconds=seconds,completed_seconds=clock,cost=None if plan_only else cost,success=None if plan_only else failure is None,failure=failure,min_clearance=None if plan_only else minimum,config=asdict(cfg),vehicle=asdict(vehicle),frames=frames,beliefs=beliefs,plans=plans,observations=observations,wall_seconds=prefix_wall_seconds+time.perf_counter()-started,source_hashes=hashes,continuation=continuation_record,
            runtime=dict(jax=jax.__version__,devices=list(map(str,jax.devices())),machine=platform.machine(),xla_flags=os.environ.get('XLA_FLAGS',''),construction_key=np.asarray(construction_key).tolist(),construction_key_scheme='fold_in(seed,271828); disjoint from performance/search streams'),
            environment={**metadata(layout,cfg.environment_scale,cfg.inspection_scale,case,cfg.geometry,cfg.detour_depth),'wind_at_start':cfg.wind_at_start,'grid':grid.tolist(),'grid_shape':[41,49],'wind':grid_wind.tolist(),'wind_units':'m/s'},
            model=dict(planner=f'full nominal acceleration sequence; 2 independent coordinates per {cfg.dt:g} s interval; MBD Eqs. 6, 8, 9',decision_dimension=2*cfg.steps,decision_map='a = umax * tanh(theta), elementwise',parameter_domain=[-cfg.latent_bound,cfg.latent_bound],execution='nonlinear rigid-body quadrotor and motor dynamics; flatness mapping and geometric attitude tracking',planning_step=cfg.dt,execution_step=.01,schedule='cosine finite interval',cosine_end=cfg.cosine_end,cosine_cumulative=np.asarray(cosine_schedule(cfg.iterations,cfg.cosine_end)[1]).tolist(),belief='POAM, actual observations only; frozen posterior within each planning call',filter='state-dependent policy-value projection with explicit infeasibility fallback',planning_filter=('off' if blind else cfg.prediction_filter),penalty_aggregation=cfg.penalty_aggregation,prediction_control_step=cfg.dt/cfg.prediction_substeps,observation_period=min(cfg.observation_period,cfg.commit),estimator_update_period=(min(cfg.observation_period,cfg.commit) if cfg.online_estimation else cfg.commit),inducing_relocation=cfg.wind_relocate,sampler_support_grid=cfg.wind_support_nodes,prediction_termination=cfg.terminate_predictions,kernel_fixed=True,execution_filter=cfg.execution_filter if method!='blind' else 'off',execution_margin=cfg.execution_margin,filter_rate_hz=1/cfg.execution_dt,planning_filter_rate_hz=1/cfg.dt,scoring=score_description,selection='independent common validation field bank; evaluated parameters only',collision_penalty='remaining mission without sensing',sensing='finite 45-degree nadir footprint; Gaussian quality and building visibility',realtime=False),rms_tracking_error=float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.,planning_seconds=[p['seconds'] for p in plans])
    for iteration in range(start_iteration,1 if plan_only else round(seconds/cfg.commit)):
        complete_planning_started=time.perf_counter()
        key,ks,kp=jax.random.split(key,3)
        mean,fields=(fast_sample(b,ks,construction_key if cfg.common_construction else jax.random.fold_in(ks,37))
                     if fast_sample is not None else sample(b,ks));fields.block_until_ready()
        construction=fields[:cfg.construction_worlds];worlds=fields[cfg.construction_worlds:cfg.construction_worlds+cfg.worlds];validation=fields[-cfg.validation_worlds:]
        if cfg.common_construction and fast_sample is None:
            # Common base draws remove resampling-induced changes to the filter.
            # The field distribution still changes with the learned posterior.
            # Performance/validation fields above retain their independent keys.
            _,stable_fields=sample(b,construction_key)
            construction=stable_fields[:cfg.construction_worlds]
        field_seconds=time.perf_counter()-complete_planning_started
        diagnostic_started=time.perf_counter()
        mu,cov=prediction(b,jnp.array(grid));std=np.sqrt(np.maximum(np.asarray(cov)[:,0,0],0))
        beliefs.append(dict(t=round(clock,4),mean=np.asarray(mu).round(4).tolist(),std=std.round(4).tolist(),rmse=float(np.sqrt(np.mean((np.asarray(mu)-grid_wind)**2)))))
        diagnostic_seconds=time.perf_counter()-diagnostic_started
        x=jnp.array(np.r_[state.position[:2],state.velocity[:2]],jnp.float32)
        eta=optimism_fraction(cfg,method,clock,seconds)
        start=time.perf_counter()
        demonstration_controls=None;rrt_record=None;rrt_seconds=0.
        if cfg.initialization=='rrt':
            rrt_start=time.perf_counter()
            demonstration_controls,rrt_record=demonstrations(x,env,cfg,seed+100003*iteration)
            demonstration_controls=jnp.asarray(demonstration_controls)
            rrt_seconds=time.perf_counter()-rrt_start
            print(f'  RRT: {rrt_record["found"]} initial trajectories, {rrt_seconds:.2f}s',flush=True)
        elif cfg.initialization=='modes':
            from voi_diffusion_ipp.quarter.modes import demonstrations as geometric_demonstrations,select_initial_modes
            rrt_start=time.perf_counter()
            bank,record=geometric_demonstrations(x,env,cfg,seed+100003*iteration)
            demonstration_controls,selection=select_initial_modes(bank,record,x,env,cfg,q)
            demonstration_controls=jnp.asarray(demonstration_controls)
            rrt_record=dict(**record,selection=selection)
            rrt_seconds=time.perf_counter()-rrt_start
            print(f'  Initial trajectories: {cfg.mode_count} modes, {rrt_seconds:.3f}s',flush=True)
        chain_results=[];chain_audits=[];planning_elapsed=0.;audit_elapsed=0.
        for chain in range(cfg.chains):
            chain_key=kp if chain==0 else jax.random.fold_in(kp,chain)
            chain_start=time.perf_counter()
            found=plan(x,q,mean,construction,worlds,validation,previous,chain_key,jnp.array(eta),jnp.array(iteration==0),demonstration_controls)
            found[0].block_until_ready();planning_elapsed+=time.perf_counter()-chain_start;chain_results.append(found)
            if iteration==0 or audit_all:
                audit_start=time.perf_counter();path=target.stem+f'_search_{iteration:04d}_chain_{chain:02d}.npz'
                np.savez_compressed(out/path,**{name:np.asarray(value) for name,value in found[3]._asdict().items()},initial_state=np.asarray(x),clarity=np.asarray(q),mean=np.asarray(mean),construction=np.asarray(construction),performance_fields=np.asarray(worlds),validation_fields=np.asarray(validation),chosen_index=np.asarray(found[4]),chosen_theta=np.asarray(found[0]),rng_key=np.asarray(chain_key),eta=eta)
                chain_audits.append(path);audit_elapsed+=time.perf_counter()-audit_start
            print(f'  chain {chain+1}/{cfg.chains}: validation={float(found[1]):.4f}, training={float(found[3].trace[-1,0]):.4f}',flush=True)
        winning_chain=int(np.argmin([float(r[1]) for r in chain_results]))
        theta,score,expected,search,chosen=chain_results[winning_chain]
        elapsed=planning_elapsed
        if not np.isfinite(float(score)):failure='no finite evaluated control sequence';break
        forecast_mean=jnp.zeros_like(mean) if blind else mean
        forecast_construction=jnp.zeros_like(construction) if blind else construction
        pred=predict(theta,x,q,forecast_mean,forecast_construction,forecast_mean)
        nominal,aa=nominal_reference(theta,x,cfg)
        index=int(search.finalist_indices[int(chosen)])
        n_initial=len(search.initial_parameters)
        if cfg.multimode:
            origin=('previous trajectory' if index==n_initial-1 else 'geometry initialization') if index<n_initial else ('final denoised vector' if index>=int(search.evaluations)-cfg.mode_count else 'diffusion proposal')
        else:origin=('previous trajectory' if index==n_initial-1 else ('RRT initialization' if index>=2 else 'initialization')) if index<n_initial else ('final denoised vector' if index==int(search.evaluations)-1 else 'diffusion proposal')
        audit=None
        if iteration==0 or audit_all:
            audit=target.stem+f'_search_{iteration:04d}.npz'
            np.savez_compressed(out/audit,**{name:np.asarray(value) for name,value in search._asdict().items()},initial_state=np.asarray(x),clarity=np.asarray(q),mean=np.asarray(mean),construction=np.asarray(construction),performance_fields=np.asarray(worlds),validation_fields=np.asarray(validation),chosen_index=np.asarray(chosen),chosen_theta=np.asarray(theta),nominal=np.asarray(nominal),accelerations=np.asarray(aa),rng_key=np.asarray(kp),eta=eta)
        plans.append(dict(t=clock,seconds=elapsed,compilation_included=iteration==start_iteration,predicted_optimistic=float(score),predicted_mean=float(expected),mean_field_monitor=float(pred.monitor),mean_field_constraint_penalty=float(pred.penalty),trace=np.asarray(search.trace).tolist(),evaluations=sum(int(r[3].evaluations) for r in chain_results),policy=theta.tolist(),decision_dimension=int(theta.size),decision_origin=origin,archive_index=index,audit=audit,chain_audits=chain_audits,winning_chain=winning_chain,chain_validation_costs=[float(r[1]) for r in chain_results],max_slack=float(pred.max_slack),points=np.asarray(pred.x[:,:2]).round(4).tolist(),nominal_points=np.asarray(nominal[:,:2]).round(4).tolist(),accelerations=np.asarray(aa).round(6).tolist(),initial_cost=float(search.initial_cost),best_training_cost=float(search.trace[-1,0])))
        plans[-1].update(optimism_fraction=eta if method in ('proposed','optimistic','fading','no_learning') else 1.,
            archive_mixture=cfg.mixture if method in ('proposed','fading','no_learning') else 0.,
            rrt=rrt_record,rrt_seconds=rrt_seconds,total_planning_seconds=elapsed+rrt_seconds,
            guidance=(None if cfg.guidance_mode=='off' else f'MBD Eq. 13; Gaussian on {cfg.guidance_mode} positions; objective selects final trajectory') if rrt_record else None,
            guidance_used_fraction=float(np.mean(np.asarray(search.guidance_log_weights)>-np.asarray(search.sample_costs)/cfg.temperature)))
        plans[-1].update(field_seconds=field_seconds,diagnostic_seconds=diagnostic_seconds,
            complete_planning_seconds=field_seconds+rrt_seconds+elapsed,
            planning_time_definition='Posterior field generation + geometry initialization + synchronized MBD and final selection; excludes reporting and execution')
        if cfg.multimode:
            mode_index=(index if index<cfg.mode_count else
                (index-(int(search.evaluations)-cfg.mode_count) if index>=int(search.evaluations)-cfg.mode_count else
                 (None if index==n_initial-1 else int(np.asarray(search.proposal_modes)[(index-n_initial)%cfg.proposals]))))
            plans[-1].update(mode_trace=np.asarray(search.mode_trace).tolist(),
                selected_mode=mode_index,
                unique_validation_entries=int(np.sum(np.asarray(search.finalist_unique))))
        print(f'{layout}/{case}/{method} plan {iteration} {elapsed:.2f}s initial={float(search.initial_cost):.4f} best={float(search.trace[-1,0]):.4f} independent={float(score):.4f} origin={origin} dim={theta.size}',flush=True)
        if plan_only:break
        execution_initial_state=state.pack().copy()
        execution_started=time.perf_counter()
        online_observations={};controller_means=[];controller_fields=[]
        if cfg.online_estimation:
            current=jnp.array(execution_initial_state,jnp.float32);chunks=[]
            for chunk in range(round(cfg.commit/observation_period)):
                cm,cc=mean,construction
                if chunk:
                    cm,cf=controller_sample(b,construction_key if cfg.common_construction else ks)
                    cc=cf[:cfg.construction_worlds]
                controller_means.append(np.asarray(cm));controller_fields.append(np.asarray(cc))
                current,part=execute(current,theta,cm,cc,x,jnp.array(round(chunk*observation_period/cfg.dt)))
                chunks.append(part)
                physical_state=State.unpack(np.asarray(current).astype(float));command=np.asarray(part[2][-1])
                plant.wind=np.r_[np.asarray(actual_wind(jnp.array(physical_state.position[None,:2])))[0],0.]
                residual=plant.derivative(physical_state.pack(),command)[3:6]-(physical_state.rotation[:,2]*(plant.allocation@(physical_state.motor_speed**2))[0]/vehicle.mass-np.array([0,0,vehicle.gravity]))
                measured=infer_wind(physical_state.velocity,residual)[:2]+cfg.observation_std*rng.normal(size=2)
                endpoint=round((chunk+1)*observation_period/.01)-1
                online_observations[endpoint]=measured
                if method!='no_learning':b=condition(b,jnp.array(physical_state.position[None,:2],jnp.float32),jnp.array(measured[None],jnp.float32))
            logs=jax.tree.map(lambda *parts:jnp.concatenate(parts),*chunks)
        else:_,logs=execute(jnp.array(execution_initial_state,jnp.float32),theta,mean,construction)
        states,refs,commands,_,applied,slacks=map(np.asarray,logs)
        plans[-1]['execution_seconds']=time.perf_counter()-execution_started
        qs,costs,gaps=map(np.asarray,monitoring(q,jnp.array(state.position[:2]),jnp.array(states)))
        executed_count=0;new_positions=[];new_measurements=[]
        for step,(packed,ref,command) in enumerate(zip(states,refs,commands)):
            if not np.isfinite(packed).all() or not np.isfinite(qs[step]).all() or not np.isfinite(costs[step]):
                failure='numerical instability';break
            state=State.unpack(packed.astype(float));clock+=.01;q=jnp.asarray(qs[step]);cost+=float(costs[step]);gap=float(gaps[step]);minimum=min(minimum,gap)
            executed_count+=1
            errors.append(float(np.linalg.norm(state.position-ref)))
            if step%10==9:frame(ref)
            if (step+1)%round(min(cfg.observation_period,cfg.commit)/.01)==0:
                if cfg.online_estimation:measured=online_observations[step]
                else:
                    plant.wind=np.r_[np.asarray(actual_wind(jnp.array(state.position[None,:2])))[0],0.]
                    residual=plant.derivative(state.pack(),command)[3:6]-(state.rotation[:,2]*(plant.allocation@(state.motor_speed**2))[0]/vehicle.mass-np.array([0,0,vehicle.gravity]))
                    measured=infer_wind(state.velocity,residual)[:2]+cfg.observation_std*rng.normal(size=2)
                observations.append(dict(t=clock,position=state.position[:2].tolist(),measurement=measured.tolist()))
                new_positions.append(state.position[:2].copy());new_measurements.append(measured)
            if gap<0:failure='collision or environment boundary contact';break
            if state.position[2]<.5 or state.position[2]>7:failure='altitude tracking failure';break
        plans[-1]['executed_max_slack']=float(np.max(slacks[:executed_count])) if executed_count else None
        plans[-1]['applied_controls']=applied[:executed_count:10].round(5).tolist()
        execution_audit=target.stem+f'_execution_{iteration:04d}.npz'
        np.savez_compressed(out/execution_audit,initial_state=execution_initial_state,
            states=states[:executed_count],references=refs[:executed_count],motor_commands=commands[:executed_count],
            applied_controls=applied[:executed_count],slacks=slacks[:executed_count],clarity=qs[:executed_count],
            cost_increments=costs[:executed_count],clearances=gaps[:executed_count],mean=np.asarray(mean),
            construction=np.asarray(construction),theta=np.asarray(theta),
            controller_mean_tables=np.asarray(controller_means[:int(np.ceil(executed_count/(observation_period/.01)))]),
            controller_construction_tables=np.asarray(controller_fields[:int(np.ceil(executed_count/(observation_period/.01)))]))
        plans[-1]['execution_audit']=execution_audit
        if failure and abs(frames[-1]['t']-clock)>1e-5:frame(ref)
        if failure:break
        if method!='no_learning' and new_positions and not cfg.online_estimation:b=condition(b,jnp.array(np.asarray(new_positions),jnp.float32),jnp.array(np.asarray(new_measurements),jnp.float32))
        previous=continuation(theta,cfg)
        print(f'  executed {clock:.1f}/{seconds:g}s cost={cost:.4f} p={state.position[:2].round(2)} clearance={minimum:.3f}',flush=True)
        write(target,result(False))
    if failure:
        # Monitoring keeps deteriorating after a stopped/collided mission.
        for _ in range(round(max(0,seconds-clock)/.1)):
            q=q/(1+env.process_noise*q*.1);cost+=.1*float(jnp.maximum(env.targets-q,0).mean())
        print('FAILED:',failure,'at',clock,flush=True)
    write(target,result(True))
    print('SAVED',target,'planning-only diagnostic; no flight outcome' if plan_only else f'cost {cost} success {failure is None}',flush=True)
    return result(True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--layout',choices=['two','multi'],default='two');p.add_argument('--case',choices=['forward','reverse','crosswind','commitment','calm_a'],default='forward');p.add_argument('--methods',nargs='+',choices=METHODS,default=['proposed','blind','blind_filter']);p.add_argument('--seconds',type=float,default=120);p.add_argument('--seed',type=int,default=8101)
    for name,default,kind in [('dt',.2,float),('horizon',45.,float),('iterations',6,int),('proposals',16,int),('worlds',4,int),('validation-worlds',8,int),('temperature',.3,float),('wind',3.1,float),('optimism',.5,float),('commit',5.,float),('cosine-end',.05,float),('batch-size',16,int),('chains',1,int),('scale',.35,float),('inspection-scale',.55,float)]:p.add_argument('--'+name,type=kind,default=default)
    p.add_argument('--initialization',choices=['gaussian','zero','rrt'],default='rrt');p.add_argument('--rrt-paths',type=int,default=3);p.add_argument('--rrt-visits',type=int,default=3);p.add_argument('--common-construction',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--guidance-sigma',type=float,default=1.);p.add_argument('--planning-clearance',type=float,default=.6);p.add_argument('--slack-weight',type=float,default=15.);p.add_argument('--output',default='output/european_quarter/full_control_v1');p.add_argument('--plan-only',action='store_true');p.add_argument('--audit-all',action='store_true');p.add_argument('--resume-directory',type=Path)
    p.add_argument('--prediction-filter',choices=['soft','hard'],default='hard');p.add_argument('--guidance-mode',choices=['off','nominal','executed'],default='off')
    p.add_argument('--rrt-diversity',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--online-estimation',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--observation-period',type=float,default=1.);p.add_argument('--wind-support-nodes',type=int,default=25);p.add_argument('--wind-relocate',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--terminate-predictions',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--penalty-aggregation',choices=['mean','tail'],default='mean')
    p.add_argument('--prior-std',type=float,default=1.2);p.add_argument('--violation-weight',type=float,default=1.);p.add_argument('--execution-margin',type=float,default=.55)
    a=p.parse_args()
    cfg=ControlSettings(dt=a.dt,horizon=a.horizon,iterations=a.iterations,proposals=a.proposals,worlds=a.worlds,validation_worlds=a.validation_worlds,temperature=a.temperature,wind_strength=a.wind,optimism=a.optimism,commit=a.commit,cosine_end=a.cosine_end,batch_size=a.batch_size,chains=a.chains,environment_scale=a.scale,inspection_scale=a.inspection_scale,initialization=a.initialization,rrt_paths=a.rrt_paths,rrt_visits=a.rrt_visits,common_construction=a.common_construction,guidance_sigma=a.guidance_sigma,planning_clearance=a.planning_clearance,slack_weight=a.slack_weight,seed=a.seed,prediction_filter=a.prediction_filter,guidance_mode=a.guidance_mode,rrt_diversity=a.rrt_diversity,online_estimation=a.online_estimation,observation_period=a.observation_period,wind_support_nodes=a.wind_support_nodes,wind_relocate=a.wind_relocate,terminate_predictions=a.terminate_predictions,penalty_aggregation=a.penalty_aggregation,prior_std=a.prior_std,violation_weight=a.violation_weight,execution_margin=a.execution_margin)
    if abs(a.seconds/cfg.commit-round(a.seconds/cfg.commit))>1e-6:raise ValueError('Mission duration must be divisible by commit interval')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for method in a.methods:
        run(a.layout,a.case,method,a.seconds,a.seed,cfg,out,a.plan_only,a.audit_all,
            a.resume_directory/f'{a.layout}_{a.case}_{method}_{a.seed}.json' if a.resume_directory else None)
        jax.clear_caches()


if __name__=='__main__':main()
