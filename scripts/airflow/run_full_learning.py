"""Full-duration spatial POAM, uncertainty-waypoint RPCBF and MBD experiments."""
import argparse,hashlib,json,signal,time,pickle,zipfile,shutil
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import jax
import jax.numpy as jnp
import full_learning_monitoring as im
import cluster_environment as ce
from guarded_waypoints import GuardedWaypointPolicy
from spatial_poam import SpatialPOAM,SpatialPOAMSettings
from cached_poam_fields import CachedFields
from rmse_gate import wind_vector_rmse,ready_for_mbd
from run_monitoring_mbd import write
from task_wind_acquisition import TaskWindAcquisition,deficit_weights
import efficient_learning_monitoring as efficient
from information_paths import InformationPathPlanner
from reachable_learning import ReachableLearning,measured_route_mask
from recording_io import write_recording
from mission_schedule import MissionSchedule,block_capacity
from exact_poam_variance import ResolvedVarianceBasis
from handoff_validation import HandoffValidator
from geometric_task_mask import CandidatePathAcquisition
from profile_io import load_profile,save_profile
from replay_prefix import ReplayPrefix
from uncertainty_handoff import ready as uncertainty_ready
from comparison_execution import unfiltered_execution
from monitoring_resume import MonitoringResume


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--require-completion',action='store_true',help='Stop the batch after saving any incomplete mission; do not spend compute on subsequent cases')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--environment-spec',type=Path,help='Explicit new scenario geometry; old default is preserved')
    p.add_argument('--wind-site',type=Path,help='Directory containing data/scenarioN stationary fields')
    p.add_argument('--initial-state',type=float,nargs=4,help='Explicit initial condition shared by all methods in the scenario')
    p.add_argument('--initial-region-clarity',type=float,help='Initial clarity in monitoring regions, applied equally to methods')
    p.add_argument('--uncertainty-only',action='store_true',help='No truth-based handoff or forecast veto; full posterior RMS only')
    p.add_argument('--global-std-threshold',type=float,default=1.1,help='Whole-map vector posterior RMS threshold in m/s')
    p.add_argument('--comparison-method',choices=['learning','mbd-unfiltered','mbd-filtered','aware'],default='learning',help='Explicit ablation; never infer method from desired outcome')
    p.add_argument('--replay-prefix',type=Path,help='Restore an interrupted prefix by verifying saved decisions through unchanged physics and actual-measurement inference, then continue MBD')
    p.add_argument('--resume-monitoring-from','--resume-from',type=Path,help='Continue full-precision robot, policy and POAM checkpoints without restarting the mission')
    p.add_argument('--verify-execution-prefix',type=Path,help='Rerun normally and verify every state and actual measurement against an interrupted recording; never replace a decision')
    p.add_argument('--seconds',type=int,default=300)
    p.add_argument('--match-duration-from',type=Path,help='Use each scenario baseline’s completed survey + monitoring duration as the total duration; fails if the baseline is incomplete')
    p.add_argument('--monitoring-seconds',type=int,default=None,help='Run this many seconds AFTER a successful survey handoff; --survey-seconds remains the maximum learning duration')
    p.add_argument('--validate-handoff',action='store_true',help='Simulation-only held-out forecast checks after RMSE is met; truth remains outside optimization/control')
    p.add_argument('--allow-validation-slack',action='store_true',help='Safety-aware comparison: allow nonzero RPCBF slack in contact-free finite forecasts; retain all costs, bounds and hard execution filtering, and report infeasibility')
    p.add_argument('--handoff-metric',choices=['position-and-cost','monitoring-cost'],default='position-and-cost',help='Explicit finite forecast validation criterion; position errors and infeasible commands are always reported')
    p.add_argument('--waypoint-precision',type=float,default=.2,help='Local posterior vector precision used to retire sensing goals; independent of global diagnostic RMSE')
    p.add_argument('--mbd-iterations',type=int,default=2)
    p.add_argument('--mbd-proposals',type=int,default=14)
    p.add_argument('--survey-seconds',type=int,default=None,help='Maximum survey duration; defaults to the requested total duration')
    p.add_argument('--rmse-threshold',type=float,default=.2)
    p.add_argument('--rmse-thresholds',default='',help='Optional prevalidated per-scenario thresholds, e.g. 1:1.25,2:2.01')
    p.add_argument('--modes',default='baseline1')
    p.add_argument('--scenarios',default='1,2')
    p.add_argument('--cases',default='center')
    p.add_argument('--wall-limit',type=int,default=3600,help='Per-mission emergency wall limit, never shared across scenarios')
    p.add_argument('--train-seconds',type=float,default=10.,help='Match validated 50-observation GP training cadence')
    p.add_argument('--plan-limit',type=float,default=1.)
    p.add_argument('--safety-confidence',type=float,default=3.,help='Shared confidence-disk multiplier for RPCBF, robust braking, and sensing-route screening; does not alter the POAM posterior')
    p.add_argument('--observation-first-inducing',action='store_true',help='Resolve equal first-pivot prior variances in favor of actual current measurements, retaining original greedy Cholesky and POAM transport')
    p.add_argument('--exact-safety-variance',action='store_true',help='Evaluate full POAM marginal variance directly at each control/forecast query; keep the coarse initialization grid')
    p.add_argument('--gp-lengths',default='.2,.5,1,2',help='Four spatial attentive-kernel scales in meters; record and validate changes explicitly')
    p.add_argument('--train-kernel',action='store_true')
    p.add_argument('--prelearned',type=Path)
    p.add_argument('--cache-spacing',type=float,default=.1)
    p.add_argument('--save-profile',action='store_true',help='Optional large inference checkpoint; recordings always saved')
    p.add_argument('--learning-policy',choices=['all','mask','correlated','routes'],default='all')
    p.add_argument('--task-std-threshold',type=float,default=.2,help='Targeted variants: task-weighted vector posterior RMS uncertainty, m/s')
    p.add_argument('--path-mask-radius',type=float,default=.25,help='Candidate-path acquisition tube radius only; no change to any physical, GP or safety model')
    p.add_argument('--efficient-planning',action='store_true',help='Reuse exact evaluated MBD training scores; batch the same validation candidates')
    p.add_argument('--learning-controller',choices=['waypoints','information-paths','reachable-waypoints'],default='waypoints')
    p.add_argument('--frontier-prefixes',action='store_true',help='Screen sensing references for current holding authority, then validate their full filtered motion')
    p.add_argument('--learning-replan-seconds',type=float,default=5.,help='Sensing only: replan from actual new measurements; MBD commitment stays 5s')
    p.add_argument('--global-information-goals',action='store_true',help='Choose distant information goals, advancing along currently supported path prefixes')
    p.add_argument('--factorized-support',action='store_true',help='Apply the unchanged support covariance using its Cholesky factor, avoiding dense inverse construction')
    p.add_argument('--prefix-screen',choices=['holding','braking-feasible'],default='holding',help='Global-goal geometric initialization only; actual execution RPCBF remains unchanged')
    p.add_argument('--learned-route-bounds',action='store_true',help='Avoid locally measured route nodes violating the same actuator reserve as recovery; no uncertainty suppression')
    p.add_argument('--diversify-stalled-routes',action='store_true',help='After measured lack of progress, try another geometric passage to the same information goal')
    p.add_argument('--model-trust',action='store_true',help='Experimental: reject MBD finalists whose 3-sigma wind acceleration-error bound exceeds the existing 25 percent actuator reserve anywhere in their predicted path')
    p.add_argument('--adaptive-learning',action='store_true',help='When model uncertainty blocks a better MBD trajectory and the accepted forecast stays within one sensor radius, resume targeted actual wind measurements along that blocked reference')
    p.add_argument('--information-trigger',choices=['stalled','lower-cost'],default='stalled',help='Lower-cost: target any lower-cost uncertainty-blocked free-space reference; retain the sensing goal across MBD probes and precompile validation branches')
    p.add_argument('--adaptive-domain',choices=['reference','clarity'],default='reference',help='Clarity: retain the global clarity-deficit weighted GP information objective during relearning; reference: weight the selected blocked MBD reference')
    a=p.parse_args()
    if a.survey_seconds is None:a.survey_seconds=a.seconds
    if a.uncertainty_only and a.validate_handoff:raise ValueError('Deployment stopping cannot use oracle forecast validation')
    if a.comparison_method!='learning' and (a.monitoring_seconds is not None or a.validate_handoff):
        raise ValueError('Immediate MBD comparison uses a fixed total mission duration without a survey')
    replay=ReplayPrefix(a.replay_prefix) if a.replay_prefix else None
    resume=MonitoringResume(a.resume_monitoring_from) if a.resume_monitoring_from else None
    verification=ReplayPrefix(a.verify_execution_prefix,allow_learning=True) if a.verify_execution_prefix else None
    if verification is not None and (resume is not None or replay is not None or ',' in a.scenarios or ',' in a.modes or ',' in a.cases):
        raise ValueError('Prefix verification requires one cold-start trial')
    if resume is not None and (replay is not None or a.prelearned or ',' in a.scenarios or ',' in a.modes or ',' in a.cases):
        raise ValueError('Exact checkpoint continuation requires one trial and no other restoration mode')
    if replay is not None and (',' in a.scenarios or ',' in a.modes or ',' in a.cases):
        raise ValueError('Replay recovery is one explicitly identified mission')
    if replay is not None and (a.adaptive_learning or a.learning_controller!='reachable-waypoints'):
        raise ValueError('Recovery supports the two-stage waypoint-then-MBD experiment only')
    matched={}
    if a.match_duration_from:
        if a.monitoring_seconds is not None:raise ValueError('Choose matched total or post-handoff duration, not both')
        for row in json.loads((a.match_duration_from/'summary.json').read_text()):
            if row['scenario'] not in set(map(int,a.scenarios.split(','))):continue
            if not row['completed'] or row.get('monitoring_duration_s')!=300:raise ValueError('Matched baseline must complete 300 s after handoff')
            matched[row['scenario']]=int(row['simulated_seconds'])
    default_rmse=a.rmse_threshold
    scenario_rmse={int(k):float(v) for k,v in (item.split(':') for item in a.rmse_thresholds.split(',') if item)}
    if a.learning_replan_seconds<=0 or abs(a.learning_replan_seconds/.2-round(a.learning_replan_seconds/.2))>1e-8:
        raise ValueError('Learning replanning must be a positive exact multiple of the .2s sensor interval')
    if a.output.exists() and any(a.output.iterdir()):raise RuntimeError('Existing outputs are preserved; choose a fresh directory')
    a.output.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Wall-time budget reached')))
    max_seconds=a.survey_seconds+a.monitoring_seconds if a.monitoring_seconds is not None else a.seconds
    if a.safety_confidence<3:raise ValueError('The shared safety confidence bound may not be narrowed below the existing value')
    cfg=replace(im.settings(),iterations=a.mbd_iterations,proposals=a.mbd_proposals,safety_confidence=a.safety_confidence)
    pcfg=replace(im.legacy.physical_settings(cfg),mission_seconds=block_capacity(max_seconds,cfg.commit),
                                 safety_margin=cfg.safety_margin,inner_samples=cfg.construction_worlds)
    specpath=a.environment_spec or im.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json'
    site=a.wind_site or im.m.physical.SITE/'revised';env,grid,spec=ce.environment(pcfg,specpath)
    basis,points,shape=im.table_grid(env.bounds,a.cache_spacing)
    if a.exact_safety_variance:basis=ResolvedVarianceBasis(basis.lower,basis.spacing)
    _,extended,_=im.table_grid(env.bounds,.4)
    outside=np.any((np.asarray(extended)<np.asarray(env.bounds[0]))|(np.asarray(extended)>np.asarray(env.bounds[1])),axis=1)
    support_points=np.vstack((np.asarray(env.cells),np.asarray(extended)[outside]))
    cached=CachedFields(points,shape,cfg,grid['all_cells'],exact_variance=a.exact_safety_variance)
    sampler=cached.build;safety_sampler=cached.safety
    execute=(unfiltered_execution if a.comparison_method=='mbd-unfiltered' else im.execution_block)(env,cfg,pcfg,basis,seconds=.2)
    if a.model_trust and not a.efficient_planning:raise ValueError('Model-support experiment uses the unchanged cached MBD evaluation budget')
    if a.allow_validation_slack and (not a.efficient_planning or a.model_trust):
        raise ValueError('Safety-aware selection requires efficient planning without the separate model-trust veto')
    if a.adaptive_learning and (not a.model_trust or a.learning_controller!='reachable-waypoints'):
        raise ValueError('Adaptive information fallback requires model-support validation and the existing reachable waypoint policy')
    with jax.experimental.disable_x64():
        optimizer=(efficient.optimizer(im.legacy.float32(env),cfg,im.legacy.float32(basis),True,wind_error_reserve=.25*cfg.umax,
            information_trigger=a.information_trigger,prepare_validation=a.information_trigger=='lower-cost')
            if a.model_trust else efficient.optimizer(im.legacy.float32(env),cfg,im.legacy.float32(basis),a.comparison_method not in ('mbd-unfiltered','mbd-filtered'),prepare_validation=True,allow_validation_slack=a.allow_validation_slack)
            if a.efficient_planning else im.optimizer(im.legacy.float32(env),cfg,im.legacy.float32(basis),True))
    validator=HandoffValidator(env,cfg,pcfg,basis,spec,allow_validation_slack=a.allow_validation_slack,metric=a.handoff_metric) if a.validate_handoff else None
    info_planner=InformationPathPlanner(env,grid,cfg,basis,frontier_prefixes=a.frontier_prefixes) if a.learning_controller=='information-paths' else None
    map_belief=cached.map
    @jax.jit
    def local_belief(b,position):
        mean,cov=b.predict(position[None])
        return mean[0],jnp.sqrt(cov[0,0,0])
    write(a.output/'environment.json',dict(grid=grid['grid'],dx_m=.125,free_indices=grid['free_indices'],
        cells=grid['cells'],membership=grid['membership'],targets=env.targets,process_noise=env.process_noise,
        initial_clarity=pcfg.initial_q,bounds=env.bounds,obstacles=env.obstacles,regions=spec['regions']))
    files=list((im.m.physical.ROOT/'scripts/airflow').glob('*.py'))+list((im.m.physical.ROOT/'src/voi_diffusion_ipp').rglob('*.py'))
    hashes={str(f.relative_to(im.m.physical.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    with zipfile.ZipFile(a.output/'executed-source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for f in files:archive.write(f,str(f.relative_to(im.m.physical.ROOT)))
    compiled=set();reports=[]
    cases=[c for c in ce.CASES if c['id'] in a.cases.split(',')]
    if a.initial_state:cases=[dict(c,state=a.initial_state) for c in cases]
    for scenario,case,mode in ((s,c,m) for s in map(int,a.scenarios.split(',')) for c in cases for m in a.modes.split(',')):
        a.rmse_threshold=scenario_rmse.get(scenario,default_rmse)
        if a.match_duration_from:
            if scenario not in matched:raise ValueError(f'Missing completed baseline for scenario {scenario}')
            a.seconds=matched[scenario];max_seconds=a.seconds
            pcfg=replace(pcfg,mission_seconds=block_capacity(max_seconds,cfg.commit))
        ident=f's{scenario}-{case["id"]}-{mode}';began=time.perf_counter()
        if replay is not None and replay.id!=ident:raise ValueError('Replay identity differs from requested trial')
        if verification is not None and verification.id!=ident:raise ValueError('Verification identity differs')
        deadline=began+a.wall_limit;signal.alarm(a.wall_limit+120)
        truth=im.m.physical.load_wind(scenario,site) # only plant, actual sensor and offline diagnostics
        x,q=ce.initial(case,env,grid,pcfg)
        if a.initial_region_clarity is not None:
            if not 0<=a.initial_region_clarity<=1:raise ValueError('Clarity must lie in [0,1]')
            q=jnp.where(jnp.asarray(grid['membership'].any(axis=1)),a.initial_region_clarity,q)
        if resume is not None:gp=resume.gp
        elif a.prelearned:
            gp=load_profile(a.prelearned/f's{scenario}-dense-profile.pkl')
            if a.factorized_support:gp.belief=gp.belief._replace(basis=gp.belief.basis._replace(support_inverse=jnp.zeros((0,0),dtype=jnp.float64)))
        else:gp=SpatialPOAM(env.cells,env.bounds,settings=SpatialPOAMSettings(lengths_m=tuple(map(float,a.gp_lengths.split(','))),observation_first_inducing=a.observation_first_inducing),prior_support=support_points,training_seed=case['seed']+441,factorized_support=a.factorized_support)
        b=gp.belief;mask=gp.mask;prior_measurements=resume.protocol['initial_measurements'] if resume is not None else len(gp.positions)
        learning_kind='mask' if mode in ('routeprecision','pathmask') else mode if mode in ('mask','correlated','routes') else a.learning_policy
        route_handoff=mode=='routeprecision'
        if route_handoff and (a.learning_controller!='reachable-waypoints' or not a.global_information_goals):
            raise ValueError('Route-precision handoff requires the declared global information-goal controller')
        if info_planner is not None and learning_kind not in ('all','mask'):
            raise ValueError('Information-path comparison supports uniform or clarity-deficit task weights')
        targeted=learning_kind!='all'
        policy=GuardedWaypointPolicy(env,umax=cfg.umax,confidence=cfg.safety_confidence);alive=jnp.array(True);clock=0.;cost=0.;contact=None;stop=None
        reacher=ReachableLearning(env,grid,cfg,basis,learning_kind,global_goals=a.global_information_goals,
            vector_precision=a.waypoint_precision,prefix_screen=a.prefix_screen,
            learned_route_bounds=a.learned_route_bounds,diversify_stalls=a.diversify_stalled_routes,
            route_precision=route_handoff) if a.learning_controller=='reachable-waypoints' else None
        if reacher is not None:policy=reacher.policy
        acquisition=TaskWindAcquisition(env,grid,cfg,learning_kind) if targeted else None
        if mode=='pathmask':
            if reacher is None:raise ValueError('Candidate-path comparison requires reachable-waypoints')
            acquisition=CandidatePathAcquisition(env,grid,cfg,a.path_mask_radius)
        priority=None;task_weights=None;task_frames=[];acquisition_time=0.;task_info=None
        samples=[];qframes=[];bframes=[];plans=[];observations=[];observation_predictions=[];updates=[];timings=[];field_timings=[]
        noise=np.random.default_rng(case['seed']).normal(size=(round(max_seconds/.2)+1,2))
        schedule=MissionSchedule(a.seconds,a.survey_seconds,a.monitoring_seconds)
        gate_checks=[];last_gate_check=-1e9
        previous=None;last_update=0.;last_update_compiled=False;path=np.asarray(x[:2])[None];handoff=None;gate_time=None;last_rmse=None
        refs=jnp.tile(x,(cfg.steps+1,1));aa=jnp.zeros((cfg.steps,2))
        learning_plan_t=None;tables=None;information_fallback=False
        protocol=dict(id=ident,scenario=scenario,method=mode,initial_condition=case,settings=asdict(cfg),
            physical_settings=asdict(pcfg),source_sha256=hashes,initial_measurements=prior_measurements,
            prelearned_source=str(a.prelearned) if a.prelearned else None,
            prelearned_scope="Unrestricted full-coverage GP diagnostic measurements; not a flown robot survey" if a.prelearned else None,
            gp_profile=asdict(gp.settings),cache_spacing_m=a.cache_spacing,prior_support_count=len(b.basis.support),
            learning_policy=learning_kind,task_std_threshold_m_s=a.task_std_threshold if targeted else None,
            matched_duration_source=str(a.match_duration_from) if a.match_duration_from else None,
            exact_safety_variance=a.exact_safety_variance,
            inducing_tie_break='Actual current observations first among equal prior-diagonal pivots' if a.observation_first_inducing else 'Previous center order first',
            efficient_planning=a.efficient_planning,monitoring_seconds_after_handoff=a.monitoring_seconds,
            waypoint_precision_m_s=a.waypoint_precision,validated_handoff=a.validate_handoff,
            handoff_validation_definition=validator.definition if validator else None,
            allow_validation_slack=a.allow_validation_slack,
            handoff_metric=a.handoff_metric,
            finalist_eligibility='Finite expected cost and no predicted contact; RPCBF relaxation is allowed and logged' if a.allow_validation_slack else 'Finite expected cost, no predicted contact and no RPCBF relaxation',
            model_trust=a.model_trust,
            adaptive_learning=a.adaptive_learning,
            information_trigger=a.information_trigger,
            adaptive_information_domain=a.adaptive_domain,
            learning_controller=a.learning_controller,
            frontier_prefixes=a.frontier_prefixes,
            learning_replan_seconds=a.learning_replan_seconds,
            global_information_goals=a.global_information_goals,
            factorized_support=a.factorized_support,
            prefix_screen=a.prefix_screen,
            learned_route_bounds=a.learned_route_bounds,
            diversify_stalled_routes=a.diversify_stalled_routes,
            route_precision_handoff=route_handoff,
            information_path_definition='Seven hold/geometric references, exact five-second execution-resolution motion forecast under 2 training plus 2 validation wind fields. Full conditional GP integrated-variance reduction at every predicted .2s sensor position, frozen kernel for acquisition only. Actual POAM updates and 15-second monitoring MBD unchanged.' if info_planner else None,
            targeted_handoff='Task-weighted vector posterior RMS uncertainty <= threshold; no true wind in decision. Mask uses exact no-sensing clarity deficit; correlated uses one-observation integrated posterior variance reduction; routes uses geometric route occupancy and expected squared aerodynamic Jacobian. These are bounded acquisition heuristics, not exact monitoring value of information.' if targeted else None,
            survey_seconds=a.seconds if mode=='survey' else a.survey_seconds,total_seconds=max_seconds,
            legacy_block_capacity_seconds=pcfg.mission_seconds,
            policy='Highest posterior variance/(1+geometric path length / 1m), persistent waypoints; actual-progress timeout with 30s cooldown; .4m/s waypoint reference; edges require exact mean-wind compensation <1.2m/s2 per component, leaving .4m/s2 authority for feedback. Full-uncertainty RPCBF on every command. Recovery: when the local learned 3-sigma holding-force bound exceeds 75% of authority, retreat along the executed path to a measured point below 50% of authority; exit once low force, low speed and clearance restore control authority. Force thresholds provide hysteresis, with no distance-from-trigger condition. Reject blocked historical return segments. No truth access.',
            handoff='Oracle diagnostic RMSE threshold; state, velocity, clarity and actual POAM retained; no synthetic training data.',
            rmse_threshold_m_s=a.rmse_threshold,rmse_domain='All collision-free grid cells, vector error norm',learning_limit_s=a.survey_seconds,
            mbd=(f'Original independent MBD chains, 15s, {cfg.proposals*(cfg.iterations+1)+1} search evaluations; cached existing mode-leader scores, same two finalists batched for validation. ' if a.efficient_planning else f'Original independent MBD chains, 15s, {cfg.proposals*(cfg.iterations+1)+1} search evaluations plus 7 mode-leader rescoring evaluations. ')+ 'Unattenuated POAM posterior, 2 training and 2 independent validation fields; optimized alternate mode leader validated.',
            safety='Full-posterior RPCBF plus robust relative-degree-two wall AND circular-obstacle braking constraints, rate 1.5/s; exact distance Hessian and nonlinear force support; hard QP, logged minimum-slack fallback; no global guarantee.',
            sensor_period_s=.2,control_period_s=.05,integration_period_s=.01,waypoint_replan_s=.2,
            prior_std_m_s=1.8,noise_std_m_s=.1,inducing=768,inducing_allocation='Original POAM pivoted Cholesky over old inducing points plus actual new measurement positions; exact permutation transport for unchanged point sets.',
            kernel_training=f'Four independent 256-row history minibatches every {round(a.train_seconds/.2)} actual readings; first at t={round(a.train_seconds/.2)*.2-.2:g}s. Per-measurement conditioning; no temporal wind model.' if a.train_kernel else 'fixed',
            truth_access='Plant, actual local noisy sensor, displayed RMSE and phase-gate boolean; never planner or controller',device=str(jax.devices()))
        if info_planner is not None:protocol['policy']=protocol['information_path_definition']
        if reacher is not None:
            protocol['waypoint_completion_variance']='Direct full POAM posterior at the 32 already evaluated candidate waypoints, including the current reachable goal. Mean and coherent-field interpolation and coarse route screening retain their existing spatial representation.'
            protocol['policy']='At every actual .2s observation, rank 32 candidate waypoints by full-GP expected integrated variance reduction / (1 + geometric distance). Restrict the .05m geometric initialization graph to the connected nodes satisfying a sufficient stationary robust braking/holding screen with the same 3-sigma confidence disk, .75 actuator reserve and rate 1.5. This is a candidate screen, not a dynamic safety certificate. Keep shared measured-path recovery and actual full execution RPCBF. Local waypoint lookahead .3m and speed .4m/s; plant, sensor, clarity and 15s MBD are unchanged.'
            if a.global_information_goals:protocol['policy']='Choose persistent global information goals over the existing mean-wind compensation graph, using full GP integrated-variance reduction and geometric distance. Unknown prior-mean wind is zero for this goal-routing screen; covariance is not suppressed. Advance only along a prefix supported by the current 3-sigma holding and braking bounds, re-evaluated after each actual .2s measurement. Learned mean wind can exclude a passage and change the geometric route. Retain shared recovery, physical dynamics, full execution RPCBF and downstream 15s MBD. No assumed future observations or altered posterior.'
            if a.global_information_goals:protocol['policy']+=' Retire a learned target at per-component variance <= half the squared declared vector-precision threshold; proximity alone does not establish learning. Check progress over successive 15s windows.'
            if a.prefix_screen=='braking-feasible':protocol['policy']+=' Prefix screen: explicitly test existence of a bounded at-rest control satisfying all robust circle/wall braking inequalities with 75% actuator authority. This replaces the stronger sufficient zero-residual-control holding test for geometry only. Actual moving-state RPCBF is unchanged.'
            if a.learned_route_bounds:protocol['policy']+=' Recompute geometric connectivity using the same nonlinear 3-sigma actuator-reserve bound that triggers recovery, applied within 0.15 m of actual sensor positions. Unmeasured nodes retain the existing mean-wind routing assumption. This explicit geometric support radius is not a sensor footprint, observation, uncertainty reset or physical obstacle. Safety retains full uncertainty everywhere.'
            if a.diversify_stalled_routes:protocol['policy']+=' Keep the information goal and progress window across temporary measured-path recovery. On less than 0.30 m net geometric progress over a 15 s window, search another passage to the SAME information goal using the existing geometry-derived temporary bottleneck cuts. Retain the cut until goal completion or abandonment. These cuts diversify references only; they never change the physical environment, belief or execution safety constraints.'
        if route_handoff:protocol['handoff']='Separate route-precision diagnostic: RMS vector posterior uncertainty <= 0.20 m/s on unique 0.05 m initialization-grid cells intersected by the actually traversed polyline AND the full remaining geometric path to the selected information goal. Repeated visits do not duplicate cell weights. Use the same full-posterior C1 uncertainty field as execution. Freeze a qualifying route domain and recheck it using the latest posterior at handoff. This finite route domain is a declared stopping proxy, not a whole-map accuracy certificate or a replacement monitoring objective. MBD and RPCBF retain the unmodified full spatial posterior. No true error enters this gate.'
        if a.model_trust:protocol['model_support']='Additional finalist-admissibility test, not altered physics or an extra monitoring objective. Keep the original 43 MBD training evaluations and independent chains. Among their already-evaluated archive members, choose the two lowest-cost distinct references with nonlinear wind acceleration-error bound k(2||mu-v||R+R^2) <= 0.4 m/s^2 throughout the nominal reference, R=3sqrt(sigma^2+0.1^2). Recheck this bound at every 0.05 s predicted state in both full hard-RPCBF validation rollouts. Additionally require ||u||_infinity + error_bound <= 1.6 m/s^2 at every predicted control step, and the same condition for nominal wind-compensated references. The 0.4 cap is the existing 25 percent actuator-reserve scale; the explicit headroom inequality checks that this authority remains available. Hold fallback remains explicit. This finite-grid model-support condition is conditional on posterior calibration, not a global safety certificate. No posterior attenuation, true wind or synthetic readings.'
        if a.adaptive_learning:protocol['adaptive_information']='After the initial route gate, keep the same five-second MBD request cadence. Resume the existing filtered sensing policy when an uncertainty-blocked reference has lower training cost by more than 1e-4 on the SAME wind fields and the accepted forecast moves less than one sensor sigma, or no finalist is valid. Exclude references already infeasible under mean-wind compensation from this information trigger. Rank actual waypoint measurements by full-GP expected variance reduction along the blocked reference, weighted by no-sensing clarity deficit through the physical sensing footprint and exact expected squared aerodynamic Jacobian. This is a bounded acquisition proxy, not exact long-term value of information. A 0.25 m tube is used only to shortlist 32 geometric measurement candidates; GP information gain is evaluated directly at the reference positions. Persist in a measurement goal across replanning/recovery. Return to MBD execution when the next request no longer triggers this condition. Actual measurements only; no simulated observation enters POAM.'
        if a.adaptive_learning and a.information_trigger=='lower-cost':protocol['adaptive_information']+=' Revised opportunity criterion: movement within the current region does not suppress learning. Any lower-cost uncertainty-blocked reference can request measurements, provided all its nominal reference positions clear physical obstacles/walls by the existing safety margin and mean wind compensation is feasible. This finite reference screen only selects information queries; the executed measurement path still comes from the unchanged geometric graph and full RPCBF. Retain persistent information goals, progress windows and temporary passage cuts across MBD probes. Reconstruct recovery history from actual sensor positions. Compile all validation-count branches during the first explicitly timed startup request; the actual evaluated candidates and wind draws are unchanged.'
        if a.adaptive_learning and a.adaptive_domain=='clarity':
            protocol['adaptive_information']='Keep the same MBD opportunity trigger, five-second request cadence, trajectory support checks and 43 training evaluations. During relearning, retain the original clarity-deficit acquisition over all free cells instead of restricting acquisition to one blocked reference. Predict no-sensing clarity with the existing dynamics over 15 s. Rank 32 measurement waypoints by full-GP expected reduction of that deficit-weighted posterior variance, divided by 1 + geometric travel distance in meters. Retain the same persistent goals, direct waypoint variance, geometric route alternatives, actual-position measurements and execution RPCBF. This is a task-weighted Bayesian experimental-design proxy, not an exact value of information. No anticipated measurement is assimilated and no true wind enters acquisition or MBD. Recheck the unchanged MBD trigger every five seconds.'
        protocol['safety_confidence_multiplier']=cfg.safety_confidence
        protocol['safety_confidence_definition']='Radius = multiplier * sqrt(full POAM marginal variance + sensor variance). The posterior and sampled wind fields are unchanged. A wider empirical bound is not a simultaneous probabilistic certificate for a misspecified GP.'
        if mode=='pathmask':
            protocol['learning_policy']='candidate-path clarity mask'
            protocol['candidate_path_mask']=dict(radius_m=a.path_mask_radius,refresh_s=2.,
                definition='Original online clarity-guided geometric alternatives; compact C2 tubes weighted by the endpoint sensing deficit. Transit inherits destination value. Exact one-reading full-GP integrated variance reduction on this domain, divided by geometric travel length. No wind truth, hypothetical observations, extra MBD objective or posterior attenuation.')
        protocol['policy']=protocol['policy'].replace('3-sigma',f'{cfg.safety_confidence:g}-sigma')
        if a.uncertainty_only:
            protocol.update(handoff='Posterior vector RMS uncertainty only; truth is diagnostic only',
                truth_access='Plant, actual noisy local measurements and offline metrics only; no truth in handoff, planner, acquisition or controller',
                rmse_threshold_m_s=None,global_std_threshold_m_s=a.global_std_threshold,
                uncertainty_only=True,validated_handoff=False)
        protocol.update(comparison_method=a.comparison_method,environment_spec=str(specpath),
            environment_spec_sha256=hashlib.sha256(specpath.read_bytes()).hexdigest())
        protocol['estimator_active']=a.comparison_method!='mbd-unfiltered'
        if a.comparison_method!='learning':
            gate_time=0.;schedule.handoff(0.)
            protocol['handoff']='No dedicated learning phase; MBD begins at t=0'
            protocol['planning_wind']='zero spatial field' if a.comparison_method.startswith('mbd-') else 'full learned POAM posterior'
            protocol['execution_safety_filter']=a.comparison_method!='mbd-unfiltered'
            if a.comparison_method.startswith('mbd-'):
                protocol['mbd']='Original independent 15-second MBD chains and clarity objective; still-air rollouts without RPCBF. Same optimization budget and geometric initializations as disturbance-aware methods.'
            if a.comparison_method=='mbd-unfiltered':
                protocol['safety']='No safety filter. Original bounded tracking input and physical contact detection only.'
                protocol['policy']='Immediate still-air MBD and still-air tracking. No dedicated sensing stage; the unused wind estimator is not updated.'
            elif a.comparison_method=='mbd-filtered':
                protocol['policy']='Immediate still-air MBD. Actual measurements update POAM for learned-wind tracking compensation and the full execution RPCBF only.'
            else:
                protocol['policy']='Immediate disturbance-aware MBD with full spatial POAM updates from actual measurements; no dedicated sensing stage.'
        write(a.output/f'{ident}-protocol.json',protocol)
        if resume is not None:
            resume.validate(ident,protocol)
            protocol['checkpoint_continuation']=resume.audit
            write(a.output/f'{ident}-protocol.json',protocol)
        if replay is not None:
            protocol['replay_recovery']=dict(replay.audit)
            write(a.output/f'{ident}-protocol.json',protocol)
        def append(t,z,qq,u,nom,slack,correction,feasible,gap,living):
            membership=grid['membership']
            samples.append([t,*z,*u,*nom,float(slack),float(correction),int(feasible),float(gap),
                float(np.maximum(np.asarray(env.targets)-qq,0).mean()),cost,
                float(qq[membership[:,0]].mean()),float(qq[membership[:,1]].mean()),float(qq.mean()),int(living)])
            # Keep rounded numeric buffers compact until the streaming writer
            # encodes each frame. It writes identical JSON values without
            # retaining millions of boxed Python floats throughout a mission.
            if abs(t-round(t))<1e-7:qframes.append(dict(t=t,q=np.round(qq,6)))
        def belief_frame(mu=None,cov=None):
            nonlocal last_rmse
            if mu is None:mu,cov=map_belief(b);jax.block_until_ready(mu)
            var=np.asarray(cov)[:,0,0];mean=np.asarray(mu);free=grid['free_indices']
            # Offline diagnostic for uncertainty-only trials. Only legacy
            # RMSE-gated experiments may use this number for the phase switch.
            actual=np.asarray(truth.at(jnp.asarray(grid['all_cells'])));err=mean[free]-actual[free]
            last_rmse=wind_vector_rmse(mean,actual,free)
            bframes.append(dict(t=clock,mean=np.round(mean,5),std=np.round(np.sqrt(var),5),
                observations=prior_measurements+len(observations),mean_variance=float(var[free].mean()),
                mean_std=float(np.sqrt(var[free]).mean()),rmse_m_s=last_rmse,
                latent_three_sigma_coverage=float(np.mean(np.linalg.norm(err,axis=1)<=3*np.sqrt(var[free]))),
                safety_disk_coverage=float(np.mean(np.linalg.norm(err,axis=1)<=cfg.safety_confidence*np.sqrt(var[free]+.01))),
                protected_inducing=int(mask.sum()),
                task_vector_std_m_s=float(np.sqrt(2*np.dot(task_weights,var))) if task_weights is not None else None,
                task_rmse_m_s=float(np.sqrt(np.sum(task_weights[:,None]*(mean-actual)**2))) if task_weights is not None else None))
            if route_handoff and tables is not None:
                metric=reacher.route_metric(tables)
                if metric is not None:
                    error=metric['mean']-np.asarray(truth.at(jnp.asarray(metric['positions'])))
                    bframes[-1].update(route_vector_std_m_s=metric['vector_std_m_s'],
                        route_rmse_m_s=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),route_cells=len(metric['positions']))
        def publish():
            arr=np.asarray(samples);visits={r['label']:next((float(s[0]) for s in arr if np.all(s[1:3]>=r['bounds'][0]) and np.all(s[1:3]<=r['bounds'][1])),None) for r in spec['regions']}
            warm=[r['wall_seconds'] for r in plans if not r['compilation_included']]
            summary=dict(rmse_threshold_m_s=None if a.uncertainty_only else a.rmse_threshold,gate_reached=gate_time is not None,gate_time_s=gate_time,completed=schedule.completed(clock),simulated_seconds=clock,planned_seconds=schedule.end,
                survey_duration_s=gate_time if gate_time is not None else clock,
                monitoring_duration_s=max(0.,clock-gate_time) if gate_time is not None else 0.,
                requested_monitoring_duration_s=a.monitoring_seconds,
                validated_handoff=a.validate_handoff,
                contact=contact,stop_reason=stop,integrated_deficit=cost,min_clearance_m=float(arr[:,12].min()),
                infeasible_execution_steps=int((arr[1:,11]==0).sum()),max_execution_slack=float(arr[:,9].max()),
                max_control_component_m_s2=float(np.abs(arr[:,5:7]).max()),first_visit_s=visits,
                actual_measurements=len(observations),mean_variance_initial=bframes[0]['mean_variance'],
                mean_variance_final=bframes[-1]['mean_variance'],variance_reduction_fraction=1-bframes[-1]['mean_variance']/bframes[0]['mean_variance'],
                initial_rmse_m_s=bframes[0]['rmse_m_s'],final_rmse_m_s=bframes[-1]['rmse_m_s'],
                wall_seconds=time.perf_counter()-began+(resume.record['summary']['wall_seconds'] if resume is not None else 0.),warm_planning_median_s=float(np.median(warm)) if warm else None,
                warm_planning_max_s=max(warm) if warm else None,
                initial_measurements=prior_measurements,
                learning_policy=learning_kind,gate_kind='no dedicated learning' if a.comparison_method!='learning' else 'route uncertainty' if route_handoff else 'task uncertainty' if targeted else 'whole-map uncertainty' if a.uncertainty_only else 'whole-map RMSE',
                task_std_threshold_m_s=a.task_std_threshold if targeted else None,
                final_task_vector_std_m_s=bframes[-1]['task_vector_std_m_s'],final_task_rmse_m_s=bframes[-1]['task_rmse_m_s'],
                final_route_vector_std_m_s=bframes[-1].get('route_vector_std_m_s'),final_route_rmse_m_s=bframes[-1].get('route_rmse_m_s'),
                warm_update_median_s=float(np.median([r['seconds'] for r in updates if not r['compilation_included']])) if len(updates)>2 else None,
                warm_update_max_s=max([r['seconds'] for r in updates if not r['compilation_included']],default=None),
                warm_safety_fields_median_s=float(np.median([r['seconds'] for r in field_timings[2:]])) if len(field_timings)>2 else None,
                planning_budget_pass=bool(warm) and max(warm)<=a.plan_limit)
            summary.update(final_clarity_deficit=float(arr[-1,13]),comparison_method=a.comparison_method,
                uncertainty_only=a.uncertainty_only,global_std_threshold_m_s=a.global_std_threshold if a.comparison_method=='learning' and a.uncertainty_only and not targeted else None)
            rec=dict(id=ident,scenario=scenario,method=mode,samples=np.round(arr,7).tolist(),clarity_frames=qframes,
                belief_frames=bframes,plans=plans,observations=observations,updates=updates,execution_timing=timings,safety_field_timing=field_timings,
                handoff=handoff,handoff_validation=gate_checks,summary=summary,task_frames=task_frames,observation_predictions=observation_predictions)
            if replay is not None:rec['replay_verification']=dict(replay.audit)
            if resume is not None:rec['checkpoint_continuation']=dict(resume.audit)
            if verification is not None:rec['prefix_verification']=dict(verification.audit,definition='Normal execution with fresh optimization and sensing decisions; check the entire saved prefix before continuing. No decisions, robot states, or observations are substituted.')
            write_recording(a.output/f'{ident}.json.gz',rec)
            # Full precision robot/clarity/reference checkpoint for future
            # recovery; playback recordings intentionally round these arrays.
            state_path=a.output/f'{ident}-execution-checkpoint.npz'
            state_tmp=state_path.with_suffix('.tmp')
            with state_tmp.open('wb') as stream:
                np.savez_compressed(stream,t=clock,x=np.asarray(x),q=np.asarray(q),
                    previous=np.asarray(previous) if previous is not None else np.empty(0),
                    refs=np.asarray(refs),accelerations=np.asarray(aa),gate_time=gate_time if gate_time is not None else -1.)
            state_tmp.replace(state_path)
            return summary
        def observe():
            nonlocal b,mask,last_update,last_update_compiled
            y=np.asarray(truth.at(x[:2]))[0]+.1*noise[round(clock/.2)]
            observations.append([clock,*np.asarray(x[:2]),*y]);ts=time.perf_counter()
            predictive_mean,predictive_std=local_belief(b,x[:2]);jax.block_until_ready(predictive_mean)
            if replay is not None:replay.verify_observation(clock,x[:2],y,predictive_mean,predictive_std)
            if verification is not None:verification.verify_observation(clock,x[:2],y,predictive_mean,predictive_std)
            observation_predictions.append(dict(t=clock,mean=np.asarray(predictive_mean).tolist(),std=float(predictive_std),
                standardized_innovation=float(np.linalg.norm(y-np.asarray(predictive_mean))/np.sqrt(float(predictive_std)**2+.01))))
            training=a.train_kernel and len(observations)%max(1,round(a.train_seconds/.2))==0
            if a.comparison_method!='mbd-unfiltered':
                gp.update(x[None,:2],jnp.asarray(y)[None],train_kernel=training)
                b,mask=gp.belief,gp.mask;jax.block_until_ready(b)
            kind='train_update' if training else 'update';last_update_compiled=kind not in compiled
            last_update=time.perf_counter()-ts;updates.append(dict(t=clock,seconds=last_update,training=training,compilation_included=last_update_compiled));compiled.add(kind)
            if verification is not None:verification.verify_posterior(clock,b)
        if resume is None:
            append(0.,np.asarray(x),np.asarray(q),np.zeros(2),np.zeros(2),0.,0.,True,env.clearance(x),True)
            belief_frame();observe()
        else:
            r=resume.record;state=resume.state
            x=jnp.asarray(state['x']);q=jnp.asarray(state['q']);clock=float(state['t'])
            refs=jnp.asarray(state['refs']);aa=jnp.asarray(state['accelerations']);previous=jnp.asarray(state['previous']) if state['previous'].size else None
            gate_time=float(state['gate_time']) if float(state['gate_time'])>=0 else None
            if gate_time is not None:
                if schedule.gate_time is None:schedule.handoff(gate_time)
                elif schedule.gate_time!=gate_time:raise ValueError('Continuation handoff differs')
            elif resume.policy_state is not None:
                policy=resume.policy_state['policy']
                if reacher is not None:
                    reacher.policy=policy
                    for key,value in resume.policy_state['reacher'].items():setattr(reacher,key,value)
            cost=r['summary']['integrated_deficit'];handoff=r['handoff'];gate_checks=r['handoff_validation']
            samples=r['samples'];qframes=r['clarity_frames'];bframes=r['belief_frames'];plans=r['plans']
            observations=r['observations'];observation_predictions=r['observation_predictions'];updates=r['updates']
            timings=r['execution_timing'];field_timings=r['safety_field_timing'];task_frames=r.get('task_frames',[])
            if task_frames:
                task_weights=np.asarray(task_frames[-1]['weights']);priority=np.asarray(task_frames[-1]['priority']);task_info=task_frames[-1]
            last_rmse=bframes[-1]['rmse_m_s'];plan_t=gate_time+cfg.commit*np.floor((clock-gate_time)/cfg.commit) if gate_time is not None else None
            print(json.dumps(dict(event='exact_checkpoint_restored',**resume.audit)),flush=True)
        tables=sampler(b,jnp.array(len(observations)),jax.random.PRNGKey(case['seed']+1000),jax.random.PRNGKey(case['seed']+271828));jax.block_until_ready(tables)
        for block in range(round(clock/.2),round(max_seconds/.2)):
            if schedule.completed(clock):break
            replaying=replay is not None and clock<replay.end
            if (a.output/'STOP').exists():
                stop='Stopped at a requested experiment checkpoint; recordings and posterior retained';break
            if time.perf_counter()>deadline:
                stop='Explicit experiment wall-time limit reached';break
            if shutil.disk_usage(a.output).free<256*1024**2:
                stop='Storage reserve reached; full inference and execution checkpoints retained';break
            if block%10==0:
                if replaying and clock in replay.tasks:
                    frame=replay.tasks[clock]
                    task_weights=np.asarray(frame['weights']);priority=np.asarray(frame['priority']);task_info=frame
                    reacher.task_weights=task_weights;task_frames.append(frame)
                elif targeted and gate_time is None:
                    mu,cov=map_belief(b);jax.block_until_ready(mu)
                    priority,task_weights,task_info=acquisition.compute(np.asarray(x),np.asarray(q),b,cached,np.asarray(mu),np.asarray(cov)[:,0,0],policy.target,policy.reachable)
                    if mode=='pathmask':reacher.task_weights=task_weights
                    acquisition_time=task_info['seconds']
                    task_frames.append(dict(t=clock,weights=np.round(task_weights,8),priority=np.round(priority,8),**task_info))
                belief_frame()
                if a.uncertainty_only:
                    mu,cov=map_belief(b);jax.block_until_ready(cov)
                    domain=task_weights if targeted else np.isin(np.arange(len(grid['all_cells'])),grid['free_indices']).astype(float)
                    ready=uncertainty_ready(np.asarray(cov)[:,0,0],domain,a.task_std_threshold if targeted else a.global_std_threshold)
                else:
                    ready=(task_weights.sum()>0 and task_info['weighted_vector_std_m_s']<=a.task_std_threshold) if targeted and task_info is not None else ready_for_mbd(last_rmse,a.rmse_threshold)
                if route_handoff:
                    metric=reacher.route_metric(tables)
                    ready=metric is not None and metric['handoff_route_selected'] and metric['vector_std_m_s']<=a.task_std_threshold
                if replaying and clock==replay.gate:
                    gate_time=clock;schedule.handoff(clock);gate_checks=list(replay.checks)
                if not replaying and mode!='survey' and gate_time is None and ready:
                    approved=True
                    if validator is not None:
                        if clock-last_gate_check>=20:
                            last_gate_check=clock
                            check=validator.check(x,q,b,cached,tables,truth,case['seed'])
                            check.update(t=clock,rmse_m_s=last_rmse,
                                threshold_m_s=a.task_std_threshold if targeted else a.rmse_threshold,
                                gate_kind='task posterior uncertainty' if targeted else 'whole-map RMSE')
                            gate_checks.append(check);approved=check['passed']
                            print(json.dumps(dict(event='handoff_validation',id=ident,**check)),flush=True)
                        else:approved=False
                    if approved:gate_time=clock;schedule.handoff(clock)
            learning=gate_time is None or mode=='survey'
            if learning and clock>=a.survey_seconds:
                stop='Learning accuracy threshold not reached; MBD has not started';break
            if learning and replaying:
                recorded=replay.plans[clock];path=np.asarray(recorded['path']);plans.append(recorded)
                ref=policy.reference(np.asarray(x),path)
                refs=jnp.tile(jnp.asarray(ref),(cfg.steps+1,1));aa=jnp.zeros((cfg.steps,2));offset=0.
            elif learning and info_planner is not None:
                if learning_plan_t is None or block%round(a.learning_replan_seconds/.2)==0:
                    weights=np.zeros(len(grid['all_cells']))
                    weights[grid['free_indices']]=(deficit_weights(np.asarray(q),env.targets,env.process_noise,cfg.horizon) if targeted else 1.)
                    weights/=max(weights.sum(),1e-12)
                    refs,aa,tables,decision=info_planner.plan(x,q,b,cached,tables,weights,jax.random.PRNGKey(case['seed']+4000+block))
                    total=sum(decision['timing'].values())+last_update+(field_timings[-1]['seconds'] if field_timings else 0.)
                    decision['timing']['posterior']=last_update
                    decision['timing']['safety_belief_refresh']=field_timings[-1]['seconds'] if field_timings else 0.
                    plans.append(dict(t=clock,phase='learning',wall_seconds=total,**decision))
                    learning_plan_t=clock;path=np.asarray(decision['path']);belief_frame()
                    if not decision['compilation_included'] and total>a.plan_limit:
                        print(json.dumps(dict(event='information_planning_deadline_missed',id=ident,t=clock,seconds=total,budget=a.plan_limit)),flush=True)
                offset=clock-learning_plan_t
            elif learning and reacher is not None:
                ts=time.perf_counter();local_mean,local_std=local_belief(b,x[:2]);jax.block_until_ready(local_mean)
                path,decision,weights,priorities=reacher.plan(np.asarray(x),np.asarray(q),b,cached,tables,clock,np.asarray(local_mean),float(local_std))
                acquisition_s=time.perf_counter()-ts;safety_refresh=field_timings[-1]['seconds'] if field_timings else 0.
                total=acquisition_s+last_update+safety_refresh
                first='reachable_plan' not in compiled;compiled.add('reachable_plan')
                plans.append(dict(t=clock,phase='learning',wall_seconds=total,compilation_included=first or last_update_compiled,
                    timing=dict(acquisition=acquisition_s,posterior=last_update,safety_belief_refresh=safety_refresh),path=path.tolist(),**decision))
                ref=policy.reference(np.asarray(x),path)
                refs=jnp.tile(jnp.asarray(ref),(cfg.steps+1,1));aa=jnp.zeros((cfg.steps,2));offset=0.
            elif learning:
                ts=time.perf_counter();mu,cov=map_belief(b);jax.block_until_ready(mu)
                local_mean,local_std=local_belief(b,x[:2])
                local_mean=np.asarray(local_mean);local_std=float(local_std)
                path,decision=policy.plan(np.asarray(x),np.asarray(cov)[:,0,0],clock,np.asarray(mu),local_mean,local_std,priority=priority)
                total=time.perf_counter()-ts+last_update+(field_timings[-1]['seconds'] if field_timings else 0.)+(acquisition_time if block%10==0 else 0.)
                plans.append(dict(t=clock,phase='learning',wall_seconds=total,compilation_included=block==0 or last_update_compiled,
                    path=path.tolist(),**decision))
                if block%10==0:belief_frame(mu,cov)
                ref=policy.reference(np.asarray(x),path)
                refs=jnp.tile(jnp.asarray(ref),(cfg.steps+1,1));aa=jnp.zeros((cfg.steps,2));offset=0.
            else:
                if handoff is None:
                    handoff=dict(t=clock,validation=gate_checks[-1] if gate_checks else None,state=np.asarray(x).tolist(),clarity=np.asarray(q).tolist(),observations=len(observations),
                                 inducing=np.asarray(b.basis.centers).tolist(),posterior_mean=np.asarray(b.mean).tolist(),
                                 posterior_file=f'{ident}-handoff.npz')
                    if route_handoff:
                        metric=reacher.route_metric(tables)
                        handoff['route_precision']={k:np.asarray(v).tolist() if isinstance(v,np.ndarray) else v for k,v in metric.items()}
                    np.savez_compressed(a.output/f'{ident}-handoff.npz',state=np.asarray(x),clarity=np.asarray(q),
                        mean=np.asarray(b.mean),covariance=np.asarray(b.covariance),a=np.asarray(b.a),B=np.asarray(b.B),
                        inducing=np.asarray(b.basis.centers),mask=np.asarray(mask))
                if round((clock-gate_time)*5)%25==0:
                    ts=time.perf_counter();tables=cached.planning(b,tables,jax.random.PRNGKey(case['seed']+1000+block));jax.block_until_ready(tables)
                    field_s=time.perf_counter()-ts
                    if replaying:
                        recorded=replay.plans[clock];theta=jnp.asarray(recorded['selected_parameters'],dtype=jnp.float32);plans.append(recorded)
                        if recorded.get('execution_fallback'):
                            seeds,_,_=im.legacy.seed_bank(x,q,env,cfg,previous);theta=seeds[0]
                        refs,aa=im.legacy.old.nominal_reference(theta.astype(jnp.float64),x,cfg);previous=theta;plan_t=clock
                    else:
                        ts=time.perf_counter();seeds,shifted,metadata=im.legacy.seed_bank(x,q,env,cfg,previous);geometry_s=time.perf_counter()-ts
                        args=im.legacy.float32((x,q,tables,seeds,shifted));ts=time.perf_counter()
                        with jax.experimental.disable_x64():result=optimizer(*args,jax.random.PRNGKey(case['seed']+2000+block));jax.block_until_ready(result)
                        mbd_s=time.perf_counter()-ts;theta,score,search,forecast,validation=result
                        safety_refresh_s=field_timings[-1]['seconds'] if field_timings else 0.
                        first='mbd' not in compiled;compiled.add('mbd');total=field_s+geometry_s+mbd_s+last_update+safety_refresh_s
                        plans.append(dict(t=clock,phase='monitoring',wall_seconds=total,compilation_included=first,
                            timing=dict(fields=field_s,geometry=geometry_s,optimization=mbd_s,posterior=last_update,safety_belief_refresh=safety_refresh_s),
                            initialization=metadata,expected_cost=float(score),evaluations=int(search.evaluations)+int(validation['leader_reevaluations']),
                            initial_cost=float(search.initial_cost),optimized_training_cost=float(np.min(np.asarray(validation['leader_costs']))),
                            optimization_trace=np.asarray(search.trace).tolist(),
                            selected_parameters=np.asarray(theta).tolist(),
                            mode_cost_trace=np.asarray(search.mode_trace).tolist(),
                            predicted_states=np.asarray(forecast.x).tolist(),validation={k:np.asarray(v).tolist() for k,v in validation.items()}))
                        if not first and total>a.plan_limit:
                            print(json.dumps(dict(event='planning_deadline_missed',id=ident,t=clock,seconds=total,budget=a.plan_limit)),flush=True)
                        if bool(validation['no_feasible']) or not np.isfinite(float(score)):
                            # Log the failure; keep execution filtering the existing hold
                            # fallback until the next real measurement/replanning call.
                            theta=seeds[0]
                            plans[-1]['execution_fallback']='No validated candidate: filtered hold; retry at next planning call'
                            print(json.dumps(dict(event='no_validated_plan',id=ident,t=clock)),flush=True)
                        refs,aa=im.legacy.old.nominal_reference(theta.astype(jnp.float64),x,cfg);previous=theta;plan_t=clock
                        if a.adaptive_learning:
                            entering=not information_fallback and bool(validation['request_targeted_learning'])
                            information_fallback=bool(validation['request_targeted_learning'])
                            plans[-1]['execution_mode']='targeted wind learning' if information_fallback else 'MBD monitoring'
                            reacher.task_reference=(np.asarray(validation['learning_reference'])
                                if information_fallback and a.adaptive_domain=='reference' else None)
                            if entering:
                                if a.information_trigger=='stalled':
                                    policy.target=None;policy.path=None;policy.retreat=None;policy.blocked_cuts[:]=False
                                actual_positions=np.asarray(observations)[:,1:3]
                                history_mean=np.asarray(basis.evaluate(jnp.asarray(actual_positions),tables.mean))
                                history_std=np.asarray(basis.evaluate(jnp.asarray(actual_positions),tables.std))[:,0]
                                _,history_bounds,_=measured_route_mask(actual_positions,history_mean,history_std,actual_positions,cfg.umax)
                                policy.history=[(p.copy(),float(f)) for p,f in zip(actual_positions,history_bounds)]
                            if information_fallback:
                                previous=None
                                if 'execution_fallback' in plans[-1]:plans[-1]['execution_fallback']='Targeted wind learning under the same execution RPCBF'
                    belief_frame()
                offset=clock-plan_t
            if a.adaptive_learning and gate_time is not None and information_fallback:
                ts=time.perf_counter();local_mean,local_std=local_belief(b,x[:2]);jax.block_until_ready(local_mean)
                path,decision,task_weights,priorities=reacher.plan(np.asarray(x),np.asarray(q),b,cached,tables,clock,np.asarray(local_mean),float(local_std))
                acquisition_s=time.perf_counter()-ts;safety_refresh=field_timings[-1]['seconds'] if field_timings else 0.
                first='referenced_learning_plan' not in compiled;compiled.add('referenced_learning_plan')
                preceding=plans[-1] if plans and plans[-1]['phase']=='monitoring' and plans[-1]['t']==clock else None
                probe_s=sum(preceding['timing'][key] for key in ('fields','geometry','optimization')) if preceding else 0.
                plans.append(dict(t=clock,phase='learning',wall_seconds=acquisition_s+last_update+safety_refresh+probe_s,
                    compilation_included=first or last_update_compiled,
                    timing=dict(acquisition=acquisition_s,posterior=last_update,safety_belief_refresh=safety_refresh,mbd_request=probe_s),path=path.tolist(),**decision))
                if block%10==0:task_frames.append(dict(t=clock,weights=np.round(task_weights,8).tolist(),priority=np.round(priorities,8).tolist(),seconds=acquisition_s,source='MBD opportunity blocked by wind uncertainty; acquisition domain: '+a.adaptive_domain))
                ref=policy.reference(np.asarray(x),path);refs=jnp.tile(jnp.asarray(ref),(cfg.steps+1,1));aa=jnp.zeros((cfg.steps,2));offset=0.;learning=True
            ts=time.perf_counter();end,rows=execute(x,q,refs,aa,jnp.array(float(offset)),tables,truth,alive);jax.block_until_ready(end)
            timings.append(dict(t=clock,seconds=time.perf_counter()-ts,compilation_included='execute' not in compiled));compiled.add('execute')
            xs,qs,us,noms,slacks,corrections,feasible,gaps,costs,living,hits=map(np.asarray,rows)
            for k in range(len(xs)):
                t=round(clock+(k+1)*pcfg.execution_dt,8)
                if contact is None and np.isfinite(hits[k]):contact=dict(time_s=clock+k*pcfg.execution_dt+float(hits[k]),position_m=xs[k,:2].tolist())
                if replay is not None:replay.verify_state(t,xs[k],qs[k])
                if verification is not None:verification.verify_state(t,xs[k],qs[k])
                cost+=float(costs[k]);append(t,xs[k],qs[k],us[k],noms[k],slacks[k],corrections[k],feasible[k],gaps[k],living[k])
            x,q,alive=end;clock=round(clock+.2,8)
            if bool(alive):observe()
            ts=time.perf_counter()
            if a.comparison_method!='mbd-unfiltered':
                tables=safety_sampler(b,tables,jax.random.PRNGKey(case['seed']+271828));jax.block_until_ready(tables)
            field_timings.append(dict(t=clock,seconds=time.perf_counter()-ts,compilation_included='safety_fields' not in compiled));compiled.add('safety_fields')
            if clock%10==0:
                belief_frame();summary=publish()
                if replay is not None and clock<=replay.end:
                    print(json.dumps(dict(event='replay_verified',t=clock,**replay.audit)),flush=True)
                print(json.dumps(dict(event='progress',id=ident,t=clock,phase='learning' if learning else 'monitoring',
                    position=np.asarray(x[:2]).tolist(),A=summary['first_visit_s']['A'],contact=contact,rmse=last_rmse,
                    threshold=a.task_std_threshold if targeted else a.global_std_threshold if a.uncertainty_only else a.rmse_threshold,gate_kind=summary['gate_kind'],
                    infeasible=summary['infeasible_execution_steps'],variance_reduction=summary['variance_reduction_fraction'],wall_s=round(time.perf_counter()-began,2))),flush=True)
            if contact is not None:stop='Contact: pilot/mission stopped, no further sensor data';break
        belief_frame();summary=publish();reports.append(summary|dict(id=ident,scenario=scenario,method=mode))
        write(a.output/'summary.json',reports)
        np.savez_compressed(a.output/f'{ident}-posterior.npz',mean=np.asarray(b.mean),covariance=np.asarray(b.covariance),
            inverse=np.asarray(b.inverse),centers=np.asarray(b.basis.centers),lengths=np.asarray(b.basis.lengths),
            **{('param_'+k):np.asarray(v) for k,v in b.basis.parameters.items()})
        if a.save_profile or (stop is not None and contact is None):
            save_profile(a.output/f'{ident}-final-profile.pkl',gp)
        if gate_time is None and contact is None:
            if info_planner is not None:raise RuntimeError('Information-path policy checkpoint is not implemented')
            keys=('route_visited','last_route_position','route_handoff_ids','route_current_ids','route_goal','task_weights','task_reference')
            policy_state=dict(t=clock,policy=policy,reacher={key:getattr(reacher,key) for key in keys} if reacher is not None else {})
            policy_path=a.output/f'{ident}-policy-checkpoint.pkl';temporary=policy_path.with_suffix('.tmp')
            with temporary.open('wb') as f:pickle.dump(policy_state,f)
            temporary.replace(policy_path)
        print(json.dumps(dict(event='complete',**summary,id=ident)),flush=True)
        if a.require_completion and not summary['completed']:
            raise RuntimeError(f'{ident}: incomplete mission saved; remaining batch stopped')
    signal.alarm(0)
    (a.output/'RUN_COMPLETE').write_text('All declared trials and checkpoints written.\n')


if __name__=='__main__':main()
