"""Runtime-guarded active-sensing comparison; original physics and POAM execution."""
import argparse,hashlib,json,time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import legacy_monitoring as legacy
import survey_monitoring as active
import survey_routes
from supported_poam import condition_supported
import monitoring_mbd as m
from run_monitoring_mbd import write,sources
from voi_diffusion_ipp.wind.cache import TableMean
from voi_diffusion_ipp.wind.gp import InnerFields

def run(args,cfg):
    if args.objective=='ledger':
        import ledger_monitoring as active
    else:
        import survey_monitoring as active
    started=time.perf_counter();pcfg=replace(legacy.physical_settings(cfg),mission_seconds=args.seconds)
    if args.execution_margin is not None:pcfg=replace(pcfg,safety_margin=args.execution_margin)
    execution_fields=args.execution_fields or cfg.construction_worlds
    update_seconds=args.posterior_update_seconds
    assert update_seconds>=.2 and abs(update_seconds/.2-round(update_seconds/.2))<1e-7
    assert update_seconds<=cfg.commit and abs(cfg.commit/update_seconds-round(cfg.commit/update_seconds))<1e-7
    env,grid=m.environment(pcfg);truth=m.physical.load_wind(args.scenario,args.site)
    aware=True;ident=f's{args.scenario}-{args.method}';out=args.output;out.mkdir(parents=True,exist_ok=True)
    resumed=json.loads((out/(ident+'.json')).read_text()) if args.resume else None
    previous_protocol=json.loads((out/(ident+'-protocol.json')).read_text()) if args.resume else None
    if resumed:
        assert not resumed['summary']['completed'] and resumed['summary']['stop_reason']=='Planning runtime budget exceeded'
        assert previous_protocol['legacy_settings']==json.loads(json.dumps(asdict(cfg)))
        assert previous_protocol['execution_belief']==dict(posterior_update_seconds=update_seconds,construction_fields=execution_fields,safety_margin_m=pcfg.safety_margin)
    hashes=sources()
    for path in [Path(__file__),Path(active.__file__),Path(__file__).with_name('survey_monitoring.py'),Path(__file__).with_name('survey_information.py'),Path(__file__).with_name('survey_routes.py'),Path(__file__).with_name('supported_poam.py'),Path(__file__).with_name('observed_poam.py'),Path(__file__).with_name('active_sensing_monitoring.py'),Path(__file__).with_name('exploration_monitoring.py'),Path(__file__).with_name('survey_forecast.py'),Path(legacy.__file__),*[m.physical.ROOT/f'src/voi_diffusion_ipp/quarter/{n}.py' for n in ('continuous','multimode','mbd_core','diffusion','environment')],m.physical.ROOT/'src/voi_diffusion_ipp/wind/cache.py']:
        hashes[str(path.relative_to(m.physical.ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    settings=asdict(pcfg)|dict(diffusion_steps=cfg.iterations,proposals=cfg.proposals,candidate_chunk=cfg.batch_size,multimode=True,legacy_safety_costs=True,planning_clearance=cfg.planning_clearance,mode_count=cfg.mode_count,validation_worlds=cfg.validation_worlds)
    protocol=dict(id=ident,scenario=args.scenario,method=args.method,seed=args.seed,settings=settings,legacy_settings=asdict(cfg),
        implementation='Original MBD score and reverse equations; route-targeted survey or monitoring objective; original relaxed RPCBF in search and full hard RPCBF in validation/execution',
        exploration_mode=args.mode,cost_allowance=args.cost_allowance,planning_safety=dict(construction_fields=2,margin_m=.06,filter='relaxed'),validation_safety=dict(construction_fields=8,margin_m=.12,filter='hard',finalists=5,selection='global MBD winner, original hold, three passage mode leaders; no validation field used to generate candidates',full_horizon_seconds=cfg.horizon,control_dt=.05,physics_clarity_dt=.01,performance_fields=cfg.validation_worlds),
        information_objective='Monitoring cost minus bounded stationary-use route-value times full-predictive-GP variance reduction; finite cost-unit surrogate',
        no_anticipated_updates=True,no_optimistic_objective=True,source_sha256=hashes,
        wind_sha256=hashlib.sha256((args.site/f'data/scenario{args.scenario}/stationary-velocity.f32').read_bytes()).hexdigest(),
        belief_evaluation='Original C1 table evaluation at 0.2 m spacing; full POAM covariance and coherent field draws retained; old fixed-kernel online conditioning with inducing relocation.',
        initialization=('Shared pre-mission noisy spatial survey; full POAM posterior, followed by actual local observations. The planner receives no true-field queries.' if args.survey else 'Zero-mean POAM prior and one actual noisy local wind observation. No survey or oracle in planning.'),
        numerics=dict(planning_dtype='float32',search_and_route_value_surrogate_dt=cfg.dt,validation_full_horizon_control_dt=.05,validation_full_horizon_physics_clarity_dt=.01,physics_dtype='float64',actual_inference_dtype='float64',posterior_field_generation_dtype='float64',physics_dt=.01,execution_filter_dt=.05),
        execution_belief=dict(posterior_update_seconds=update_seconds,construction_fields=execution_fields,safety_margin_m=pcfg.safety_margin),
        prior_configuration='survey-matched 128 inducing points, fixed kernel, geometry-only initialization' if args.matched_prior else 'original prior or supplied survey',
        runtime_limit_s=args.plan_limit,runtime_scope='All geometry, posterior-field sampling, MBD search and independent validation until executable control selection; diagnostic display rollout and serialization recorded separately',method_definition='Route-targeted online survey using full-predictive-GP covariance reduction' if args.mode=='survey' else 'Matched monitoring objective with the same observed-location inducing allocation and geometric starts', inducing_allocation='Protect inducing centers at actual observation locations spaced at least0.15m; replace nearest unprotected center, then nearest at capacity128; full POAM transported statistics and residual retained; float64 inference', survey_objective='J minus N max_r(V_r g_r), with sampled route improvement V, full-predictive-GP fractional reduction g at 25 prospective sensor locations, N=clip(remaining_seconds/15,1,20); linear repeated-use surrogate, actual relocated POAM reduction audited separately', survey_gate=dict(wind_std_m_s=.30, force_variation_m_s2=.10,task_cost_improvement=.01,tracking_improvement_m=.10,perfect_information_probe='Three coarse 0.25 s paired route-value surrogates in one posterior physical field, same sample/reference, no true field; final candidate trajectories independently validated for all15s at .05s control and .01s physics/clarity',monitoring_cost_allowance=args.cost_allowance))
    if args.objective=='ledger':
        protocol.update(objective_revision='constrained-route-survey-ledger',
            information_objective='Maximize full-horizon task-route covariance reduction subject to predicted committed monitoring-loss credit; otherwise minimize monitoring cost',
            survey_objective='Full predictive-GP fractional covariance reduction at all75 forecast5Hz positions over15s; exactfirst5s cost debit relative to minimum15s monitoring candidate in same validation bank',
            ledger=dict(initial_credit=args.ledger_initial,capacity=args.ledger_capacity,refill_per_second=args.ledger_rate,scope='Cumulative predicted positive5s monitoring-plus-penalty loss relative to finite same-bank monitoring comparator; not actual-regret guarantee'),
            initialization_alignment='Identical paired task and aerodynamic-relevance target weights in geometricfrontier construction and MBD; persistentfrontier follows a collision-free route suffix',
            method_definition='Constrained task-directed online wind survey with persistent geometricfrontiers' if args.mode=='survey' else 'Matched monitoring objective with same posterior, geometricstarts and execution-rate validation')
        protocol['survey_gate']['monitoring_cost_allowance']=None
        protocol['survey_gate']['task_definition']='Positive cost OR tracking improvement OR rescued contact in paired sampled route probes; no benefit requirement relative to globally best alternative'
    if resumed:
        protocol['source_history']=previous_protocol.get('source_history',[])+[previous_protocol['source_sha256']]
        protocol['rejected_planning_calls']=previous_protocol.get('rejected_planning_calls',[])+[dict(t=resumed['plans'][-1]['t'],wall_seconds=resumed['plans'][-1]['wall_seconds'],timing=resumed['plans'][-1]['timing'],reason='One-second guard stopped execution before this plan was applied')]
        protocol['resumed_at_s']=resumed['summary']['simulated_seconds']
    write(out/(ident+'-protocol.json'),protocol)
    write(out/'environment.json',dict(grid=grid['grid'],dx_m=.125,free_indices=grid['free_indices'],cells=grid['cells'],membership=grid['membership'],targets=env.targets,process_noise=env.process_noise,initial_clarity=pcfg.initial_q,bounds=env.bounds,obstacles=env.obstacles,regions=[dict(label='A',bounds=[[4.98,2.295],[8,4.625]]),dict(label='B',bounds=[[0,0],[2.9,1.755]])]))
    x=jnp.array([1.45,.875,0.,0.]);q=jnp.full(len(env.cells),pcfg.initial_q)
    # Restore the old estimator workflow: condition the full posterior on real
    # measurements; kernel parameters stay fixed during the mission.
    b=m.estimator(pcfg,jax.random.PRNGKey(args.seed+101)).planning_belief()
    if args.matched_prior and not args.survey:b=legacy.matched_prior(env)
    survey=None
    if args.survey:
        survey=np.load(args.survey)
        b=jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(b),[jnp.asarray(survey[k]) for k in sorted(survey.files) if k.startswith('belief_leaf_')])
        protocol['initial_survey']=dict(file=args.survey.name,sha256=hashlib.sha256(args.survey.read_bytes()).hexdigest(),measurements=len(survey['positions']),inducing_points=len(b.mean),noise_std_m_s=.1)
        write(out/(ident+'-protocol.json'),protocol)
    support_mask=jnp.zeros(len(b.mean),dtype=bool)
    def condition(bb,p,y):
        nonlocal support_mask
        bb,support_mask=condition_supported(bb,p,y,support_mask)
        return bb
    rng=np.random.default_rng(args.seed);noise=rng.normal(size=(round(args.seconds/.2)+1,2))
    y=truth.at(x[:2])+pcfg.observation_std*noise[0]
    b32,x32,y32=legacy.float32((b,x,y))
    with jax.experimental.disable_x64():
        b=condition(b32,x32[None,:2],y32);jax.block_until_ready(b)
    samples=[];states=[];clarities=[];qframes=[];bframes=[];plans=[];observations=[[0.,*np.asarray(x[:2]),*np.asarray(y)[0]]]
    clock=0.;cost=0.;alive=True;contact=None;minimum=float(env.clearance(x));infeasible=0;max_slack=0.;previous=None
    start_cycle=0;previous_wall=0.;ledger_credit=args.ledger_initial;frontier_state=None
    if resumed:
        raw=np.load(out/(ident+'-execution.npz'));summary=resumed['summary'];clock=summary['simulated_seconds'];start_cycle=round(clock/cfg.commit)
        assert resumed['plans'][-1]['t']==clock
        pending=out/f'{ident}-plan-{start_cycle:03d}.npz';saved=np.load(pending)
        b=jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(b),[jnp.asarray(saved[k]) for k in sorted(saved.files) if k.startswith('belief_leaf_')])
        support_mask=jnp.asarray(saved['support_mask'])
        samples=raw['samples'].tolist();states=list(raw['states']);clarities=list(raw['clarity']);observations=raw['observation_rows'].tolist()
        x=jnp.asarray(states[-1]);q=jnp.asarray(clarities[-1]);cost=summary['integrated_deficit'];alive=bool(samples[-1][18]);contact=summary['contact']
        minimum=summary['min_clearance_m'];infeasible=summary['infeasible_execution_steps'];max_slack=summary['max_execution_slack'];previous_wall=summary.get('total_wall_seconds',summary['wall_seconds'])
        qframes=resumed['clarity_frames'];bframes=[f for f in resumed['belief_frames'] if f['t']<clock];plans=resumed['plans'][:-1]
        previous=jnp.asarray(np.load(out/f'{ident}-plan-{start_cycle-1:03d}.npz')['theta']) if start_cycle else None
        if args.objective=='ledger' and plans:
            ledger_credit=plans[-1]['ledger_credit_after']
            frontier_state=plans[-1].get('frontier_state_after')
        pending.rename(out/f'{ident}-rejected-{len(protocol["rejected_planning_calls"]):03d}.npz')
    execute=m.execution_kernel(env,pcfg)
    sample=active.field_sampler(cfg)
    @jax.jit
    def execution_tables(bb):
        coefficients=bb.sample_field(jax.random.PRNGKey(args.seed+271828),execution_fields)
        fields=jax.vmap(lambda c:bb.basis.evaluate(legacy.POINTS,c))(coefficients).reshape((-1,*legacy.SHAPE))
        return bb.mean_at(legacy.POINTS).reshape(legacy.SHAPE),fields
    env32=legacy.float32(env)
    with jax.experimental.disable_x64():
        plan=active.optimizer(env32,cfg,args.mode,args.cost_allowance)
        relevance_probe=active.prepare_relevance(env32,cfg) if args.objective=='ledger' else None
    def record(t,u,nom,info):
        nonlocal minimum
        gap=float(env.clearance(x));minimum=min(minimum,gap)
        means=[float(np.mean(np.asarray(q)[grid['membership'][:,j]])) for j in range(2)]
        samples.append([t,*np.asarray(x),*np.asarray(u),*np.asarray(nom),float(info.slack) if info else 0.,float(info.correction) if info else 0.,int(bool(info.feasible)) if info else 1,gap,float(m.clarity.deficit(q,env.targets)),cost,*means,float(np.mean(q)),int(alive)])
        states.append(np.asarray(x));clarities.append(np.asarray(q))
        if abs(t/.5-round(t/.5))<1e-6:qframes.append(dict(t=t,q=np.round(np.asarray(q),6)))
    def snapshot():
        mu,cov=b.predict(jnp.asarray(grid['all_cells']));mu=np.asarray(mu);std=np.sqrt(np.asarray(cov)[:,0,0])
        bframes.append(dict(t=clock,mean=np.round(mu,5),std=np.round(std,5),observations=len(observations)+(len(survey['positions']) if survey is not None else 0)))
    def publish(complete=False,reason=None):
        crossings=m.physical.passage_crossings(np.asarray(samples)[:,:5])
        summary=dict(supported_inducing_points=int(np.asarray(support_mask).sum()),support_capacity_saturated=bool(np.all(np.asarray(support_mask))),completed=complete,simulated_seconds=clock,planned_seconds=args.seconds,contact=contact,integrated_deficit=cost,min_clearance_m=minimum,infeasible_execution_steps=infeasible,max_execution_slack=max_slack,max_control_component_m_s2=float(np.max(np.abs(np.asarray(samples)[:,5:7]))),fallback_periods=0,wall_seconds=time.perf_counter()-started,passage_crossings=crossings,final_region_clarity=samples[-1][15:17],final_whole_clarity=samples[-1][17],stop_reason=reason,
            warm_plan_max_seconds=max([p['wall_seconds'] for p in plans if not p['timing']['compilation_included']] or [0.]),rejected_planning_calls=protocol.get('rejected_planning_calls',[]),total_wall_seconds=previous_wall+time.perf_counter()-started)
        write(out/(ident+'.json'),dict(id=ident,scenario=args.scenario,method=args.method,protocol=ident+'-protocol.json',sample_columns=['t','x','y','vx','vy','ux','uy','nominal_x','nominal_y','slack','correction','feasible','clearance','deficit','cost','clarity_A','clarity_B','clarity_whole','alive'],samples=np.round(samples,7),clarity_frames=qframes,belief_frames=bframes,plans=plans,summary=summary))
        if complete or reason or abs(clock/args.checkpoint_seconds-round(clock/args.checkpoint_seconds))<1e-7:
            np.savez_compressed(out/(ident+'-execution.npz'),samples=samples,states=states,clarity=clarities,observation_rows=observations,support_mask=np.asarray(support_mask))
    if not resumed:record(0.,np.zeros(2),np.zeros(2),None)
    for cycle in range(start_cycle,round(args.seconds/cfg.commit)):
        snapshot();t0=time.perf_counter()
        bp,xp,qp=legacy.float32((b,x,q))
        ts=time.perf_counter();tables=sample(b,jax.random.PRNGKey(args.seed+1000+cycle),jax.random.PRNGKey(args.seed+271828));mean,construction,worlds,validation=legacy.float32(tables);jax.block_until_ready(validation);sampling_s=time.perf_counter()-ts
        relevance_result=None;relevance_seconds=0.
        def route_weights(positions,weights,task_seeds,route_indices):
            nonlocal relevance_result,relevance_seconds
            begin=time.perf_counter()
            with jax.experimental.disable_x64():
                relevance_result=relevance_probe(bp,xp,qp,mean,construction,worlds[0],jnp.asarray(task_seeds,jnp.float32),jnp.asarray(positions,jnp.float32),jnp.asarray(weights,jnp.float32),jnp.asarray(route_indices,jnp.int32))
                jax.block_until_ready(relevance_result)
            relevance_seconds=time.perf_counter()-begin
            return np.asarray(relevance_result.weights)
        ts=time.perf_counter()
        route_options=dict(relevance_weights=route_weights,frontier_state=frontier_state,survey_enabled=True) if args.objective=='ledger' else {}
        seeds,shifted,metadata,target_positions,target_weights=survey_routes.seed_bank(x,q,env,cfg,b,previous,**route_options)
        initialization_s=time.perf_counter()-ts
        seeds,shifted=legacy.float32((seeds,shifted))
        if args.objective=='ledger':
            if relevance_result is None:
                publish(False,'No connected geometric task route');raise RuntimeError('No connected geometric task route')
            extra=(relevance_result,jnp.asarray(ledger_credit,jnp.float32))
        else:extra=()
        with jax.experimental.disable_x64():
            ts=time.perf_counter();result=plan(bp,xp,qp,mean,construction,worlds,validation,shifted,jax.random.PRNGKey(args.seed+2000+cycle),seeds,jnp.asarray(target_positions,jnp.float32),jnp.asarray(target_weights,jnp.float32),jnp.asarray(metadata['task_route_indices'],jnp.int32),jnp.asarray(args.seconds-clock,jnp.float32),*extra);jax.block_until_ready(result);optimization_s=time.perf_counter()-ts
        elapsed=time.perf_counter()-t0
        theta,score,expected,search,index=result[:5]
        if not bool(jnp.isfinite(score)):
            publish(False,'No finite validated candidate');raise RuntimeError('No finite validated candidate')
        # Reuse the selected independent validation forecast for the display.
        # It was already included in planning time; no additional rollout.
        prediction_seconds=0.
        validation_result=result.validation
        candidate_search_index=int(result.candidate_indices[index])
        selected_origin=('hold' if candidate_search_index<0 else 'initialization' if int(search.finalist_indices[candidate_search_index])<len(search.initial_parameters) else 'diffusion')
        seed_costs=np.mean(np.asarray(search.initial_field_costs),axis=1)
        plans.append(dict(t=clock,valid=True,wall_seconds=elapsed,timing=dict(initialization=initialization_s,field_sampling=sampling_s,optimization_validation=optimization_s,diagnostic_prediction_after_selection=prediction_seconds,compilation_included=cycle==start_cycle),initialization=metadata,expected_cost=float(expected),selection_score=float(score),survey_fraction=float(result.information[index]),exploring=bool(result.exploring),route_wind_std=float(result.route_uncertainty),route_force_std=float(result.route_relevance),task_relevance_benefits=np.asarray(result.relevance_benefits),score_normalization=float(result.score_normalization),route_values=np.asarray(result.route_values),validation_bonuses=np.asarray(result.validation_bonuses),amortization_factor=float(result.amortization_factor),information_enabled=bool(result.information_enabled),task_index=int(result.task_index),cost_budget=float(result.budget),initial_cost_budget=float(result.initial_budget),validation_task_costs=np.asarray(result.task_costs),validation_information=np.asarray(result.information),validation_contacts=np.asarray(result.contacts),evaluations=int(search.evaluations),budget_initialization_evaluations=0,reused_initial_evaluations=len(seeds)+1,validation_evaluations=len(result.candidates),total_candidate_evaluations=int(search.evaluations)+len(result.candidates),relevance_forecasts=6,selected_origin=selected_origin,candidate_indices=np.asarray(result.candidate_indices),selected_candidate_index=int(index),validation_prefix_max_slack=np.asarray(validation_result.prefix_max_slack),validation_prefix_infeasible_steps=np.asarray(validation_result.prefix_infeasible_steps),seed_costs=seed_costs,trace=np.asarray(search.trace),predicted_states=np.asarray(validation_result.states[index]),predicted_contacts=np.asarray(result.contacts[index]),predicted_min_clearance=np.asarray(validation_result.min_clearance[index]),predicted_max_slack=np.asarray(validation_result.max_slack[index]),predicted_monitoring_cost=np.asarray(validation_result.values[index,:,4]),predicted_safety_cost=np.asarray(validation_result.values[index,:,5])))
        extra_archive={}
        if args.objective=='ledger':
            next_frontier=metadata.get('next_frontier_state')
            selected_mode=(candidate_search_index%cfg.mode_count if 0<=candidate_search_index<2*cfg.mode_count else None)
            if bool(result.exploring) and selected_mode is not None:
                next_frontier=metadata['seed_frontier_states'][selected_mode]
            next_credit=float(active.replenish(ledger_credit,float(result.ledger_debit),cfg.commit,args.ledger_rate,args.ledger_capacity))
            plans[-1].update(ledger_credit_before=ledger_credit,ledger_credit_after=next_credit,ledger_debit=float(result.ledger_debit),validation_commit_costs=np.asarray(result.commit_costs),ledger_eligible=np.asarray(result.ledger_eligible),frontier_state_before=frontier_state,frontier_state_after=next_frontier)
            plans[-1]['timing'].update(relevance_inside_initialization=relevance_seconds,geometry_without_relevance=initialization_s-relevance_seconds)
            extra_archive=dict(ledger_credit=np.asarray(ledger_credit),ledger_debit=np.asarray(result.ledger_debit),ledger_eligible=np.asarray(result.ledger_eligible),validation_commit_costs=np.asarray(result.commit_costs),relevance_benefits=np.asarray(relevance_result.benefits),route_uncertainty=np.asarray(relevance_result.uncertainty),route_force=np.asarray(relevance_result.force))
        np.savez_compressed(out/f'{ident}-plan-{cycle:03d}.npz',**extra_archive,support_mask=np.asarray(support_mask),candidates=np.asarray(result.candidates),candidate_indices=np.asarray(result.candidate_indices),validation_prefix_max_slack=np.asarray(validation_result.prefix_max_slack),validation_prefix_infeasible_steps=np.asarray(validation_result.prefix_infeasible_steps),target_positions=np.asarray(target_positions),target_weights=np.asarray(target_weights),route_indices=np.asarray(metadata['task_route_indices']),theta=np.asarray(theta),initial_state=np.asarray(x),initial_clarity=np.asarray(q),mean=np.asarray(mean),construction=np.asarray(construction),worlds=np.asarray(worlds),validation=np.asarray(validation),validation_task_costs=np.asarray(result.task_costs),validation_information=np.asarray(result.information),validation_contacts=np.asarray(result.contacts),cost_budget=np.asarray(result.budget),route_values=np.asarray(result.route_values),validation_bonuses=np.asarray(result.validation_bonuses),amortization_factor=np.asarray(result.amortization_factor),information_enabled=np.asarray(result.information_enabled),task_index=np.asarray(result.task_index),selected_index=np.asarray(index),**{'search_'+name:np.asarray(value) for name,value in zip(search._fields,search)},**{f'belief_leaf_{i:03d}':np.asarray(value) for i,value in enumerate(jax.tree_util.tree_leaves(b))})
        print(json.dumps(dict(event='plan',id=ident,t=clock,seconds=elapsed,compilation=cycle==start_cycle,cost=float(expected),survey_fraction=float(result.information[index]),exploring=bool(result.exploring),support_points=int(np.asarray(support_mask).sum()))),flush=True)
        if cycle>start_cycle and elapsed>args.plan_limit:
            publish(False,'Planning runtime budget exceeded');raise RuntimeError(f'Planning budget exceeded: {elapsed:.3f} s > {args.plan_limit:g} s')
        previous=theta;refs,aa=legacy.old.nominal_reference(theta.astype(jnp.float64),x,cfg)
        # The actual plant uses the unchanged true wind and original hard filter.
        if execution_fields==cfg.construction_worlds:exec_mean,exec_draws=mean,construction
        else:
            exec_mean,exec_draws=execution_tables(b)
        mean_live=TableMean(legacy.BASIS,exec_mean.astype(jnp.float64));inner=InnerFields(legacy.BASIS,exec_draws.astype(jnp.float64))
        positions=[];ys=[]
        for k in range(round(cfg.commit/pcfg.execution_dt)):
            local=k*pcfg.execution_dt;i=min(int(round(local/cfg.dt,8)),cfg.steps-1);off=local-i*cfg.dt
            ref=refs[i];a=aa[i];reference=jnp.r_[ref[:2]+off*ref[2:]+.5*off*off*a,ref[2:]+off*a]
            nom=m.tracking_control(x,reference,a,mean_live,pcfg,True)
            was_alive=alive
            end,u,info=execute(x,q,nom,mean_live,inner,truth,jnp.asarray(alive))
            xn,qn,survived,incurred,gap,hit=end
            if alive and not bool(survived):contact=dict(time_s=clock+float(hit),position_m=np.asarray(xn[:2]).tolist())
            if alive:infeasible+=int(not bool(info.feasible));max_slack=max(max_slack,float(info.slack))
            x,q=xn,qn;cost+=float(incurred);alive=bool(survived);clock=round(clock+pcfg.execution_dt,8)
            record(clock,u if was_alive else jnp.zeros(2),nom,info if was_alive else None)
            if alive and abs(clock/.2-round(clock/.2))<1e-6:
                y=np.asarray(truth.at(x[:2]))[0]+pcfg.observation_std*noise[round(clock/.2)]
                positions.append(np.asarray(x[:2]));ys.append(y);observations.append([clock,*np.asarray(x[:2]),*y])
            if positions and abs(clock/update_seconds-round(clock/update_seconds))<1e-7:
                with jax.experimental.disable_x64():
                    b=condition(b,jnp.asarray(positions,jnp.float32),jnp.asarray(ys,jnp.float32));jax.block_until_ready(b)
                if update_seconds<cfg.commit:exec_mean,exec_draws=execution_tables(b);jax.block_until_ready(exec_draws)
                positions=[];ys=[]
                if update_seconds<cfg.commit:
                    mean_live=TableMean(legacy.BASIS,exec_mean.astype(jnp.float64));inner=InnerFields(legacy.BASIS,exec_draws.astype(jnp.float64))
        if positions:
            with jax.experimental.disable_x64():
                b=condition(b,jnp.asarray(positions,jnp.float32),jnp.asarray(ys,jnp.float32));jax.block_until_ready(b)
        if args.objective=='ledger':
            ledger_credit=next_credit;frontier_state=next_frontier
        # Retain every sample, but serialize the growing viewer/checkpoint only
        # at the checkpoint interval. Progress logs and plan audits stay at 5 s.
        if abs(clock/args.checkpoint_seconds-round(clock/args.checkpoint_seconds))<1e-7:publish()
        print(json.dumps(dict(event='executed',id=ident,t=clock,position=np.asarray(x[:2]).tolist(),contact=contact,wall_seconds=time.perf_counter()-started)),flush=True)
    snapshot();publish(True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scenario',type=int,choices=(1,2),required=True);p.add_argument('--method',choices=('aware','baseline'),required=True)
    p.add_argument('--mode',choices=('survey','passive'),required=True);p.add_argument('--cost-allowance',type=float,default=1.)
    p.add_argument('--objective',choices=('task_value','ledger'),default='task_value')
    p.add_argument('--ledger-initial',type=float,default=.4);p.add_argument('--ledger-capacity',type=float,default=.6);p.add_argument('--ledger-rate',type=float,default=.01)
    p.add_argument('--seconds',type=float,default=300);p.add_argument('--horizon',type=float,default=15);p.add_argument('--proposals',type=int,default=14);p.add_argument('--levels',type=int,default=2)
    p.add_argument('--worlds',type=int,default=1);p.add_argument('--validation',type=int,default=1);p.add_argument('--validation-batch',type=int,default=5);p.add_argument('--seed',type=int,default=8101);p.add_argument('--plan-limit',type=float,default=1.)
    p.add_argument('--proposal-batch',type=int,help='Parallel evaluation batch size only; does not change the proposal count')
    p.add_argument('--site',type=Path,default=m.physical.SITE);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--survey',type=Path,help='Explicit shared noisy pre-mission survey fixture; never an oracle field')
    p.add_argument('--matched-prior',action='store_true',help='Use survey-matched GP capacity/kernel/inducing geometry with zero survey observations')
    p.add_argument('--posterior-update-seconds',type=float,default=.2,help='Assimilation and execution-field refresh interval; measurements remain at 5 Hz')
    p.add_argument('--execution-fields',type=int,default=8,help='Execution construction sample count; planning count is unchanged')
    p.add_argument('--execution-margin',type=float,default=.12,help='Execution-only barrier margin in meters; physical geometry is unchanged')
    p.add_argument('--checkpoint-seconds',type=float,default=30.,help='Full execution-array checkpoint interval; all steps are retained in the final archive')
    p.add_argument('--resume',action='store_true',help='Continue an explicit planning-budget stop; preserve its rejected call and exact recorded prefix')
    args=p.parse_args();
    
    cfg=replace(legacy.Settings(),safety_margin=.12,construction_worlds=8,prediction_filter='hard',horizon=args.horizon,proposals=args.proposals,batch_size=args.proposal_batch or 7,iterations=args.levels,worlds=args.worlds,validation_worlds=args.validation,validation_batch_size=args.validation_batch);cfg.__post_init__();run(args,cfg)
