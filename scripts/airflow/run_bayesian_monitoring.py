"""Task-directed Bayesian look-ahead over original MBD proposals; fixed physical models."""
import argparse,hashlib,json,time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import legacy_monitoring as legacy
import monitoring_mbd as m
import exploration_monitoring as dual
from run_monitoring_mbd import write,sources
from voi_diffusion_ipp.wind.cache import TableMean
from voi_diffusion_ipp.wind.gp import InnerFields

def run(args,cfg):
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
    for path in [Path(__file__),Path(dual.__file__),Path(legacy.__file__),*[m.physical.ROOT/f'src/voi_diffusion_ipp/quarter/{n}.py' for n in ('continuous','multimode','mbd_core','diffusion','environment')],m.physical.ROOT/'src/voi_diffusion_ipp/wind/cache.py']:
        hashes[str(path.relative_to(m.physical.ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    settings=asdict(pcfg)|dict(diffusion_steps=cfg.iterations,proposals=cfg.proposals,candidate_chunk=cfg.batch_size,multimode=True,legacy_safety_costs=True,planning_clearance=cfg.planning_clearance,mode_count=cfg.mode_count,validation_worlds=cfg.validation_worlds)
    protocol=dict(id=ident,scenario=args.scenario,method=args.method,seed=args.seed,settings=settings,legacy_settings=asdict(cfg),
        implementation='Original MBD proposal optimization with finite task-directed Bayesian rollout selection',
        learning_lookahead=args.method=='aware',lookahead=dict(roots=3,branches=1,conditional_worlds=1,continuations=4,update_s=cfg.commit,construction_fields=8,margin_m=.12,objective='Expected integrated monitoring deficit plus original physical penalties; no information bonus'),
        no_anticipated_updates=args.method!='aware',no_optimistic_objective=True,source_sha256=hashes,
        wind_sha256=hashlib.sha256((args.site/f'data/scenario{args.scenario}/stationary-velocity.f32').read_bytes()).hexdigest(),
        belief_evaluation='Original C1 table evaluation at 0.2 m spacing; full POAM covariance and coherent field draws retained; old fixed-kernel online conditioning with inducing relocation.',
        initialization=('Shared pre-mission noisy spatial survey; full POAM posterior, followed by actual local observations. The planner receives no true-field queries.' if args.survey else 'Zero-mean POAM prior and one actual noisy local wind observation. No survey or oracle in planning.'),
        numerics=dict(planning_dtype='float32',planning_dt=cfg.dt,physics_dtype='float64',physics_dt=.01,execution_filter_dt=.05),
        execution_belief=dict(posterior_update_seconds=update_seconds,construction_fields=execution_fields,safety_margin_m=pcfg.safety_margin),
        prior_configuration='survey-matched 128 inducing points, fixed kernel, geometry-only initialization' if args.matched_prior else 'original prior or supplied survey',
        runtime_limit_s=args.plan_limit,method_definition='Task-directed Bayesian learning look-ahead' if args.method=='aware' else 'Matched look-ahead with frozen forecast wind belief; actual execution still learns')
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
    condition=jax.jit(lambda bb,p,y:bb.condition(p,y,relocate=True))
    rng=np.random.default_rng(args.seed);noise=rng.normal(size=(round(args.seconds/.2)+1,2))
    y=truth.at(x[:2])+pcfg.observation_std*noise[0]
    b32,x32,y32=legacy.float32((b,x,y))
    with jax.experimental.disable_x64():
        b=condition(b32,x32[None,:2],y32);jax.block_until_ready(b)
    samples=[];states=[];clarities=[];qframes=[];bframes=[];plans=[];observations=[[0.,*np.asarray(x[:2]),*np.asarray(y)[0]]]
    clock=0.;cost=0.;alive=True;contact=None;minimum=float(env.clearance(x));infeasible=0;max_slack=0.;previous=None
    start_cycle=0;previous_wall=0.
    if resumed:
        raw=np.load(out/(ident+'-execution.npz'));summary=resumed['summary'];clock=summary['simulated_seconds'];start_cycle=round(clock/cfg.commit)
        assert resumed['plans'][-1]['t']==clock
        pending=out/f'{ident}-plan-{start_cycle:03d}.npz';saved=np.load(pending)
        b=jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(b),[jnp.asarray(saved[k]) for k in sorted(saved.files) if k.startswith('belief_leaf_')])
        samples=raw['samples'].tolist();states=list(raw['states']);clarities=list(raw['clarity']);observations=raw['observation_rows'].tolist()
        x=jnp.asarray(states[-1]);q=jnp.asarray(clarities[-1]);cost=summary['integrated_deficit'];alive=bool(samples[-1][18]);contact=summary['contact']
        minimum=summary['min_clearance_m'];infeasible=summary['infeasible_execution_steps'];max_slack=summary['max_execution_slack'];previous_wall=summary.get('total_wall_seconds',summary['wall_seconds'])
        qframes=resumed['clarity_frames'];bframes=[f for f in resumed['belief_frames'] if f['t']<clock];plans=resumed['plans'][:-1]
        previous=jnp.asarray(np.load(out/f'{ident}-plan-{start_cycle-1:03d}.npz')['theta']) if start_cycle else None
        pending.rename(out/f'{ident}-rejected-{len(protocol["rejected_planning_calls"]):03d}.npz')
    execute=m.execution_kernel(env,pcfg)
    sample=legacy.table_sampler(cfg)
    @jax.jit
    def execution_tables(bb):
        coefficients=bb.sample_field(jax.random.PRNGKey(args.seed+271828),execution_fields)
        fields=jax.vmap(lambda c:bb.basis.evaluate(legacy.POINTS,c))(coefficients).reshape((-1,*legacy.SHAPE))
        return bb.mean_at(legacy.POINTS).reshape(legacy.SHAPE),fields
    env32=legacy.float32(env)
    with jax.experimental.disable_x64():
        plan=legacy.optimizer(env32,cfg,True)
        selector=dual.BayesianSelector(env32,replace(cfg,safety_margin=.12,construction_worlds=8,prediction_filter='hard'),learning=args.method=='aware',roots=3,branches=1,conditional_worlds=1)
        predictor=jax.jit(lambda th,x,q,mean,construction,worlds:jax.vmap(lambda field:legacy.rollout(th,x,q,mean,construction,field,env32,cfg,aware))(worlds))
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
        summary=dict(completed=complete,simulated_seconds=clock,planned_seconds=args.seconds,contact=contact,integrated_deficit=cost,min_clearance_m=minimum,infeasible_execution_steps=infeasible,max_execution_slack=max_slack,max_control_component_m_s2=float(np.max(np.abs(np.asarray(samples)[:,5:7]))),fallback_periods=0,wall_seconds=time.perf_counter()-started,passage_crossings=crossings,final_region_clarity=samples[-1][15:17],final_whole_clarity=samples[-1][17],stop_reason=reason,
            warm_plan_max_seconds=max([p['wall_seconds'] for p in plans if not p['timing']['compilation_included']] or [0.]),rejected_planning_calls=protocol.get('rejected_planning_calls',[]),total_wall_seconds=previous_wall+time.perf_counter()-started)
        write(out/(ident+'.json'),dict(id=ident,scenario=args.scenario,method=args.method,protocol=ident+'-protocol.json',sample_columns=['t','x','y','vx','vy','ux','uy','nominal_x','nominal_y','slack','correction','feasible','clearance','deficit','cost','clarity_A','clarity_B','clarity_whole','alive'],samples=np.round(samples,7),clarity_frames=qframes,belief_frames=bframes,plans=plans,summary=summary))
        if complete or reason or abs(clock/args.checkpoint_seconds-round(clock/args.checkpoint_seconds))<1e-7:
            np.savez_compressed(out/(ident+'-execution.npz'),samples=samples,states=states,clarity=clarities,observation_rows=observations)
    if not resumed:record(0.,np.zeros(2),np.zeros(2),None)
    for cycle in range(start_cycle,round(args.seconds/cfg.commit)):
        snapshot();t0=time.perf_counter()
        seeds,shifted,metadata=legacy.seed_bank(x,q,env,cfg,previous);initialization_s=time.perf_counter()-t0
        bp,xp,qp=legacy.float32((b,x,q));seeds,shifted=legacy.float32((seeds,shifted))
        with jax.experimental.disable_x64():
            ts=time.perf_counter();mean,fields=sample(bp,jax.random.PRNGKey(args.seed+1000+cycle),jax.random.PRNGKey(args.seed+271828));jax.block_until_ready(fields);sampling_s=time.perf_counter()-ts
            construction=fields[:cfg.construction_worlds];worlds=fields[cfg.construction_worlds:cfg.construction_worlds+cfg.worlds];validation=fields[-cfg.validation_worlds:]
            ts=time.perf_counter();result=plan(xp,qp,mean,construction,worlds,validation,shifted,jax.random.PRNGKey(args.seed+2000+cycle),jnp.array(1.),jnp.array(False),seeds);jax.block_until_ready(result);optimization_s=time.perf_counter()-ts
        theta,score,expected,search,index=result
        with jax.experimental.disable_x64():
            candidates,root_indices=dual.diverse_prefixes(theta,seeds,xp,cfg,count=3)
            decision=selector.select(bp,xp,qp,candidates,jax.random.PRNGKey(args.seed+3000+cycle),jax.random.PRNGKey(args.seed+271828))
        elapsed=time.perf_counter()-t0
        root_index=decision['index'];theta=candidates[root_index];score=decision['scores'][root_index]
        if not bool(jnp.isfinite(score)):
            publish(False,'No finite validated candidate');raise RuntimeError('No finite validated candidate')
        with jax.experimental.disable_x64():
            prediction=predictor(theta,xp,qp,mean if aware else jnp.zeros_like(mean),construction if aware else jnp.zeros_like(construction),validation if aware else jnp.zeros_like(validation));jax.block_until_ready(prediction)
        seed_costs=np.mean(np.asarray(search.initial_field_costs),axis=1)
        plans.append(dict(t=clock,valid=True,wall_seconds=elapsed,timing=dict(initialization=initialization_s,field_sampling=sampling_s,optimization_validation=optimization_s,lookahead=decision['timing'],compilation_included=cycle==start_cycle),initialization=metadata,expected_cost=float(score),root_scores=decision['scores'],selected_root=root_index,mbd_expected_cost=float(expected),evaluations=int(search.evaluations),validation_evaluations=int(np.asarray(search.finalist_unique).sum()),selected_origin=('mbd' if root_index==0 else 'hold' if root_index==1 else 'alternative'),seed_costs=seed_costs,trace=np.asarray(search.trace),predicted_states=decision['predicted_states'][root_index],predicted_contacts=decision['predicted_contacts'][root_index],predicted_min_clearance=decision['predicted_min_clearance'][root_index],predicted_max_slack=np.asarray(prediction.max_slack),predicted_monitoring_cost=np.asarray(prediction.monitor),predicted_safety_cost=np.asarray(prediction.penalty)))
        np.savez_compressed(out/f'{ident}-plan-{cycle:03d}.npz',theta=np.asarray(theta),mbd_theta=np.asarray(result[0]),selected_root=root_index,root_candidates=np.asarray(candidates),root_indices=root_indices,**{'dual_'+key:np.asarray(value) for key,value in decision.items() if key not in ('timing',)},initial_state=np.asarray(x),initial_clarity=np.asarray(q),mean=np.asarray(mean),construction=np.asarray(construction),worlds=np.asarray(worlds),validation=np.asarray(validation),**{'search_'+name:np.asarray(value) for name,value in zip(search._fields,search)},**{f'belief_leaf_{i:03d}':np.asarray(value) for i,value in enumerate(jax.tree_util.tree_leaves(b))})
        print(json.dumps(dict(event='plan',id=ident,t=clock,seconds=elapsed,compilation=cycle==start_cycle,cost=float(score))),flush=True)
        if cycle>start_cycle and elapsed>args.plan_limit:
            publish(False,'Planning runtime budget exceeded');raise RuntimeError(f'Planning budget exceeded: {elapsed:.3f} s > {args.plan_limit:g} s')
        previous=theta;refs,aa=legacy.old.nominal_reference(theta.astype(jnp.float64),x,cfg)
        # The actual plant uses the unchanged true wind and original hard filter.
        if execution_fields==cfg.construction_worlds:exec_mean,exec_draws=mean,construction
        else:
            with jax.experimental.disable_x64():exec_mean,exec_draws=execution_tables(b)
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
        # Retain every sample, but serialize the growing viewer/checkpoint only
        # at the checkpoint interval. Progress logs and plan audits stay at 5 s.
        if abs(clock/args.checkpoint_seconds-round(clock/args.checkpoint_seconds))<1e-7:publish()
        print(json.dumps(dict(event='executed',id=ident,t=clock,position=np.asarray(x[:2]).tolist(),contact=contact,wall_seconds=time.perf_counter()-started)),flush=True)
    snapshot();publish(True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scenario',type=int,choices=(1,2),required=True);p.add_argument('--method',choices=('aware','baseline'),required=True)
    p.add_argument('--seconds',type=float,default=300);p.add_argument('--horizon',type=float,default=15);p.add_argument('--proposals',type=int,default=14);p.add_argument('--levels',type=int,default=2)
    p.add_argument('--worlds',type=int,default=1);p.add_argument('--validation',type=int,default=2);p.add_argument('--validation-batch',type=int,default=4);p.add_argument('--seed',type=int,default=8101);p.add_argument('--plan-limit',type=float,default=1.)
    p.add_argument('--proposal-batch',type=int,help='Parallel evaluation batch size only; does not change the proposal count')
    p.add_argument('--site',type=Path,default=m.physical.SITE);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--survey',type=Path,help='Explicit shared noisy pre-mission survey fixture; never an oracle field')
    p.add_argument('--matched-prior',action='store_true',help='Use survey-matched GP capacity/kernel/inducing geometry with zero survey observations')
    p.add_argument('--posterior-update-seconds',type=float,default=.2,help='Assimilation and execution-field refresh interval; measurements remain at 5 Hz')
    p.add_argument('--execution-fields',type=int,default=8,help='Execution construction sample count; planning count is unchanged')
    p.add_argument('--execution-margin',type=float,default=.12,help='Execution-only barrier margin in meters; physical geometry is unchanged')
    p.add_argument('--checkpoint-seconds',type=float,default=30.,help='Full execution-array checkpoint interval; all steps are retained in the final archive')
    p.add_argument('--resume',action='store_true',help='Continue an explicit planning-budget stop; preserve its rejected call and exact recorded prefix')
    args=p.parse_args();cfg=replace(legacy.Settings(),horizon=args.horizon,proposals=args.proposals,batch_size=args.proposal_batch or 7,iterations=args.levels,worlds=args.worlds,validation_worlds=args.validation,validation_batch_size=args.validation_batch);cfg.__post_init__();run(args,cfg)
