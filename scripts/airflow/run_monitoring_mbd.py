"""Run matched current-belief / wind-blind MBD missions, with raw audit files."""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import hashlib,json,time,shutil
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

import monitoring_mbd as m
import monitoring_routes
from voi_diffusion_ipp.wind.gp import InnerFields,WorldField
jax.config.update('jax_compilation_cache_dir',str(m.physical.ROOT/'.cache/monitoring_mbd_jax'))


def ready(value):
    if isinstance(value,dict):return {k:ready(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [ready(v) for v in value]
    if isinstance(value,(np.ndarray,jax.Array)):return ready(np.asarray(value).tolist())
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    return value


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(ready(value),separators=(',',':'),allow_nan=False));tmp.replace(path)


def sources():
    root=m.physical.ROOT
    paths=[Path(__file__),Path(m.__file__),Path(monitoring_routes.__file__),root/'scripts/airflow/robot_review.py',
           root/'src/voi_diffusion_ipp/clarity.py',root/'src/voi_diffusion_ipp/planning/mbd.py',
           root/'src/voi_diffusion_ipp/quarter/filter.py',
           *[root/f'src/voi_diffusion_ipp/safety/{name}.py' for name in ('flow_policy_value','rpcbf','qp','spline')],
           *[root/f'src/voi_diffusion_ipp/wind/{name}.py' for name in ('poam','poam_belief','gp')],
           root/'src/voi_diffusion_ipp/quadrotor/aerodynamics.py']
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(args):
    cfg=replace(m.Settings(),mission_seconds=args.seconds,horizon_seconds=args.horizon,commit_seconds=args.commit,
                diffusion_steps=args.levels,proposals=args.proposals,reference_tracking=True,multimode=True,tracking_kp=4.411764705882352,tracking_kv=7.352941176470588,reject_collisions=False,legacy_safety_costs=True,planning_clearance=m.physical.CFG.safety_margin)
    cfg.__post_init__()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    env,grid=m.environment(cfg);truth=m.physical.load_wind(args.scenario,args.site)
    ident=f's{args.scenario}-{args.method}';aware=args.method=='aware'
    resumed=None;prior_protocol=None
    if args.resume and (out/(ident+'.json')).exists():
        resumed=json.loads((out/(ident+'.json')).read_text());prior_protocol=json.loads((out/(ident+'-protocol.json')).read_text())
        for key,value in asdict(cfg).items():
            if key!='mission_seconds' and prior_protocol['settings'][key]!=ready(value):raise ValueError('Resume settings changed: '+key)
        for path,digest in sources().items():
            if path!='scripts/airflow/run_monitoring_mbd.py' and prior_protocol['source_sha256'].get(path)!=digest:raise ValueError('Resume source changed: '+path)
        if resumed['summary']['simulated_seconds']>=cfg.mission_seconds:raise ValueError('Resume requires a longer mission')
    protocol=dict(id=ident,scenario=args.scenario,method=args.method,seed=args.seed,settings=asdict(cfg),
        reference_curve_sha256=hashlib.sha256(np.stack([m.physical.route_curve(route,args.site).c for route in ('central','lower','upper')]).tobytes()).hexdigest(),
        method_definition='Expected frozen-current-belief MBD + relaxed RPCBF inside each forecast; full online POAM + RPCBF in execution' if aware else 'Still-air MBD with no belief or filter in its objective; full online POAM + RPCBF only after optimization',
        no_anticipated_updates=True,no_optimistic_objective=True,belief_evaluation='Direct full POAM predictive model and its existing coherent Matheron sampler; no lookup-table belief approximation',
        decisions='Two independent nominal reference-acceleration coordinates per decision interval; all coordinates optimized by original MBD, with a separate reverse chain per start. The original PD reference tracker and current-mean air-force compensation run before RPCBF. Coarse graph trajectories initialize only.',
        coarse_initialization='Online clarity-guided 64x37 graph, compiled Dijkstra, distinct free-space destinations and temporary cuts across geometrically discovered bottlenecks to retain distinct alternative passages. Approximate sensing ranking only; the full objective scores all candidates.',
        initialization='Untrained zero-mean POAM prior, plus one actual noisy observation at the initial position. No prior survey or supplied true field.',
        plant='p_dot=v; v_dot=u+(0.1/0.680)*norm(w_xy(p)-v)*(w_xy(p)-v)',
        clarity_equation='q_dot=C(p)^2/R*(1-q)^2-Q*q^2; C=exp(-||cell-p||^2/(2*sigma^2))',
        objective=f'Expected time integral of equal-area whole-free-space mean(max(target-q,0)), plus the old study clearance and speed terms: max({cfg.planning_clearance}-clearance,0)^2 + 2*max(speed-3.1,0)^2 while alive. The clearance preference uses this environment safety margin, not the larger historical environment threshold. Predicted contact stops motion and sensing; the full remaining-horizon decay cost is retained. Executed monitoring deficit is reported separately from planning penalties. No uncertainty bonus, tail selection, arrival reward, or injected dwell schedule.',
        source_sha256=sources(),wind_sha256=hashlib.sha256((args.site/f'data/scenario{args.scenario}/stationary-velocity.f32').read_bytes()).hexdigest(),
        limitations=['Finite Monte Carlo and diffusion budget; optimizer convergence is not established.','Existing POAM spectral residual sampling and finite inducing representation are retained, not replaced.','Forecast safety is evaluated every 0.2 s; actual RPCBF every 0.05 s. Both use 0.01 s RK4 physics.','Relaxed RPCBF permits constraint slack; finite-sample filtering is not a global safety guarantee.','Existing CFD is provisional; this is one paired development seed per environment.'])
    if prior_protocol:protocol['resumed_protocols']=[*prior_protocol.get('resumed_protocols',[]),dict(settings=prior_protocol['settings'],source_sha256=prior_protocol['source_sha256'],completed_prefix_s=resumed['summary']['simulated_seconds'])]
    write(out/(ident+'-protocol.json'),protocol)
    write(out/'environment.json',dict(grid=grid['grid'],dx_m=grid['dx_m'],free_indices=grid['free_indices'],cells=grid['cells'],
        membership=grid['membership'],targets=env.targets,process_noise=env.process_noise,initial_clarity=cfg.initial_q,
        bounds=env.bounds,obstacles=env.obstacles,regions=[dict(label='A',bounds=[[4.98,2.295],[8,4.625]]),dict(label='B',bounds=[[0,0],[2.9,1.755]])]))
    if args.preview:
        args.preview.mkdir(parents=True,exist_ok=True)
        for name in ('environment.json',ident+'-protocol.json'):shutil.copyfile(out/name,args.preview/name)
        write(args.preview/'manifest.json',dict(duration_s=cfg.mission_seconds,state='computing',source_directory=str(out)))
    rng=np.random.default_rng(args.seed)
    observation_noise=rng.normal(size=(round(cfg.mission_seconds/cfg.dt)+1,2))
    poam=m.estimator(cfg,jax.random.PRNGKey(args.seed+101))
    print(json.dumps(dict(event='initial_observation',id=ident)),flush=True)
    x=jnp.array([1.45,.875,0.,0.]);q=jnp.full(len(env.cells),cfg.initial_q)
    initial_y=np.asarray(truth.at(x[:2]))+cfg.observation_std*observation_noise[0]
    poam.update(x[None,:2],jnp.asarray(initial_y),jax.random.PRNGKey(args.seed+102),train_steps=cfg.poam_train_steps)
    optimizer=m.planner(env,cfg,aware);execute=m.execution_kernel(env,cfg)
    decay=jax.jit(lambda state,clarity:m.advance(state,clarity,jnp.zeros(2),truth,env,cfg,False,interval=cfg.execution_dt))
    predict=jax.jit(lambda th,x,q,b,inner,outer:jax.vmap(lambda c:m.forecast(th,x,q,b,inner,WorldField(outer.basis,c),env,cfg,True))(outer.coefficients)) if aware else jax.jit(lambda th,x,q,b,inner,outer:m.forecast(th,x,q,None,None,m.StillAir(),env,cfg,False))
    previous=None;compiled_optimizer=None;alive=True;contact=None;clock=0.;cumulative=0.;minimum=float(env.clearance(x))
    samples=[];q_frames=[];belief_frames=[];plans=[];raw_q=[];raw_x=[];actual_observations=[]
    start=time.perf_counter();infeasible=0;max_slack=0.;max_control=0.;fallbacks=0
    def snapshot_belief(b,t):
        mu,cov=b.predict(jnp.asarray(grid['all_cells']));mu=np.asarray(mu);std=np.sqrt(np.asarray(cov)[:,0,0])
        actual=np.asarray(truth.at(grid['cells']));pred=np.asarray(b.mean_at(env.cells))
        belief_frames.append(dict(t=t,mean=np.round(mu,5),std=np.round(std,5),
                                  rmse=float(np.sqrt(np.mean((pred-actual)**2))),observations=len(poam.data_x)))
    def record(t,u,nom,info):
        nonlocal minimum,max_control
        minimum=min(minimum,float(env.clearance(x)));max_control=max(max_control,float(np.max(np.abs(u))))
        means=np.array([np.mean(np.asarray(q)[grid['membership'][:,i]]) for i in range(2)])
        samples.append([t,*np.asarray(x),*np.asarray(u),*np.asarray(nom),float(info.slack) if info is not None else 0.,
                        float(info.correction) if info is not None else 0.,int(bool(info.feasible)) if info is not None else 1,
                        float(env.clearance(x)),float(m.clarity.deficit(q,env.targets)),cumulative,*means,float(np.mean(q)),int(alive)])
        raw_q.append(np.asarray(q));raw_x.append(np.asarray(x))
        if abs(t/.5-round(t/.5))<1e-6:q_frames.append(dict(t=t,q=np.round(np.asarray(q),6)))
    if resumed:
        raw=np.load(out/(ident+'-execution.npz'));samples=raw['samples'].tolist();raw_q=list(raw['clarity']);raw_x=list(raw['states'])
        actual_observations=raw['observation_rows'].tolist();clock=float(samples[-1][0]);x=jnp.asarray(raw_x[-1]);q=jnp.asarray(raw_q[-1]);alive=bool(samples[-1][18])
        q_frames=resumed['clarity_frames'];belief_frames=[f for f in resumed['belief_frames'] if f['t']<clock];plans=resumed['plans']
        summary=resumed['summary'];contact=summary['contact'];cumulative=summary['integrated_deficit'];minimum=summary['min_clearance_m']
        infeasible=summary['infeasible_execution_steps'];max_slack=summary['max_execution_slack'];max_control=summary['max_control_component_m_s2'];fallbacks=summary['fallback_periods']
        start-=summary['wall_seconds'];previous=jnp.asarray(np.load(out/f'{ident}-plan-{len(plans)-1:03d}.npz')['theta']) if plans else None
        obs=np.asarray(actual_observations).reshape(-1,5)
        for cycle in range(round(clock/cfg.commit_seconds)):
            batch=obs[(obs[:,0]>cycle*cfg.commit_seconds)&(obs[:,0]<=(cycle+1)*cfg.commit_seconds)]
            if len(batch):poam.update(jnp.asarray(batch[:,1:3]),jnp.asarray(batch[:,3:5]),jax.random.PRNGKey(args.seed+3000+cycle),train_steps=cfg.poam_train_steps,relocate=True)
        if resumed['summary']['completed']:
            b=poam.planning_belief();mu,cov=b.predict(jnp.asarray(grid['all_cells']));last=resumed['belief_frames'][-1]
            np.testing.assert_allclose(mu,last['mean'],atol=5.1e-6,rtol=0)
            np.testing.assert_allclose(np.sqrt(np.asarray(cov)[:,0,0]),last['std'],atol=5.1e-6,rtol=0)
        print(json.dumps(dict(event='resumed',id=ident,t=clock,observations=len(obs))),flush=True)
    else:record(0.,np.zeros(2),np.zeros(2),None)
    def publish(complete=False):
        elapsed=time.perf_counter()-start
        summary=dict(completed=complete,simulated_seconds=clock,planned_seconds=cfg.mission_seconds,contact=contact,
                     integrated_deficit=cumulative,min_clearance_m=minimum,infeasible_execution_steps=infeasible,
                     max_execution_slack=max_slack,max_control_component_m_s2=max_control,
                     fallback_periods=fallbacks,wall_seconds=elapsed,
                     passage_crossings=m.physical.passage_crossings(np.array(samples)[:,:5]),
                     final_region_clarity=samples[-1][15:17],final_whole_clarity=samples[-1][17])
        write(out/(ident+'.json'),dict(id=ident,scenario=args.scenario,method=args.method,protocol=ident+'-protocol.json',
            sample_columns=['t','x','y','vx','vy','ux','uy','nominal_x','nominal_y','slack','correction','feasible','clearance','deficit','cost','clarity_A','clarity_B','clarity_whole','alive'],
            samples=np.round(samples,7),clarity_frames=q_frames,belief_frames=belief_frames,plans=plans,summary=summary))
        np.savez_compressed(out/(ident+'-execution.npz'),samples=np.asarray(samples),states=raw_x,clarity=raw_q,
                            observation_rows=np.asarray(actual_observations).reshape(-1,5))
        if args.preview:
            tmp=args.preview/(ident+'.json.tmp');shutil.copyfile(out/(ident+'.json'),tmp);tmp.replace(args.preview/(ident+'.json'))
    for cycle in range(round(clock/cfg.commit_seconds),round(cfg.mission_seconds/cfg.commit_seconds)):
        print(json.dumps(dict(event='planning_belief',id=ident,cycle=cycle,t=clock)),flush=True)
        b=poam.planning_belief();snapshot_belief(b,clock)
        eps=jax.random.normal(jax.random.PRNGKey(args.seed+1000+cycle),(cfg.inner_samples+cfg.worlds,*b.sample_shape))
        all_fields=b.inner_fields(eps)
        inner=InnerFields(all_fields.basis,all_fields.coefficients[:cfg.inner_samples])
        outer=InnerFields(all_fields.basis,all_fields.coefficients[cfg.inner_samples:])
        if alive:
            seed_array,seed_metadata=monitoring_routes.seed_bank(x,q,env,cfg,previous)
            seeds=jnp.asarray(seed_array)
            wall=time.perf_counter()
            print(json.dumps(dict(event='optimize',id=ident,cycle=cycle,t=clock)),flush=True)
            plan_key=jax.random.PRNGKey(args.seed+2000+cycle)
            if compiled_optimizer is None:
                compiled_optimizer=optimizer.lower(x,q,b,inner,outer,seeds,plan_key).compile()
                print(json.dumps(dict(event='compiled',id=ident,seconds=time.perf_counter()-wall)),flush=True)
            result,seed_costs=compiled_optimizer(x,q,b,inner,outer,seeds,plan_key);jax.block_until_ready(result)
            seconds=time.perf_counter()-wall
            valid=bool(np.isfinite(result.cost));theta=result.vector if valid else jnp.zeros(cfg.steps*2)
            fallbacks+=int(not valid);previous=theta
            prediction=predict(theta,x,q,b,inner,outer);jax.block_until_ready(prediction)
            xs=np.asarray(prediction.states);xs=xs if aware else xs[None]
            plan=dict(t=clock,valid=valid,initialization=seed_metadata,expected_cost=float(result.cost),wall_seconds=seconds,evaluations=int(result.evaluations),
                      predicted_monitoring_cost=np.asarray(prediction.monitoring_cost),predicted_safety_cost=np.asarray(prediction.safety_cost),
                      trace=np.asarray(result.trace),seed_costs=np.asarray(seed_costs),predicted_states=np.round(xs,6),
                      predicted_contacts=np.asarray(prediction.collided),predicted_min_clearance=np.asarray(prediction.min_clearance),predicted_max_slack=np.asarray(prediction.max_slack))
            plans.append(plan)
            np.savez_compressed(out/f'{ident}-plan-{cycle:03d}.npz',theta=np.asarray(theta),initials=np.asarray(seeds),
                initial_state=np.asarray(x),initial_clarity=np.asarray(q),inner_coefficients=np.asarray(inner.coefficients),
                outer_coefficients=np.asarray(outer.coefficients),trace=np.asarray(result.trace),
                **{'history_'+name:np.asarray(value) for name,value in zip(result.history._fields,result.history)},
                **{f'belief_leaf_{i:03d}':np.asarray(value) for i,value in enumerate(jax.tree_util.tree_leaves(b))},
                belief_centers=np.asarray(b.basis.centers),belief_mean=np.asarray(b.mean),belief_covariance=np.asarray(b.covariance))
            print(json.dumps(ready(dict(event='plan',id=ident,cycle=cycle,t=clock,seconds=seconds,cost=result.cost,valid=valid))),flush=True)
        else:theta=jnp.zeros(cfg.steps*2)
        refs,accels=m.nominal_reference(theta,x,cfg)
        reference_initial=np.asarray(x)
        nominal=cfg.umax*np.tanh(np.asarray(theta).reshape(cfg.steps,2))
        observations=[];positions=[]
        for k in range(round(cfg.commit_seconds/cfg.execution_dt)):
            local=k*cfg.execution_dt;nom=jnp.asarray(nominal[min(int(round(local/cfg.dt,8)),cfg.steps-1)])
            if cfg.reference_tracking:
                index=min(int(round(local/cfg.dt,8)),cfg.steps-1);offset=local-index*cfg.dt
                ref=refs[index];a=accels[index]
                interpolated=jnp.r_[ref[:2]+offset*ref[2:]+.5*offset**2*a,ref[2:]+offset*a]
                nom=m.tracking_control(x,interpolated,a,b,cfg,True)
            if alive:
                end,u,info=execute(x,q,nom,b,inner,truth,jnp.array(alive))
                xn,qn,survived,incurred,gap,hit=end
                alive=bool(survived);cumulative+=float(incurred);minimum=min(minimum,float(gap))
                if not alive:contact=dict(time_s=clock+float(hit),position_m=np.asarray(xn[:2]).tolist())
                x,q=xn,qn;infeasible+=int(not bool(info.feasible));max_slack=max(max_slack,float(info.slack))
            else:
                end=decay(x,q);x,q=end[:2];cumulative+=float(end[3])
                u=jnp.zeros(2);info=None
            clock=round(clock+cfg.execution_dt,8);record(clock,u,nom,info)
            if alive and abs(clock/cfg.dt-round(clock/cfg.dt))<1e-6:
                noise=observation_noise[round(clock/cfg.dt)]
                y=np.asarray(truth.at(x[:2]))[0]+cfg.observation_std*noise
                positions.append(np.asarray(x[:2]));observations.append(y);actual_observations.append([clock,*np.asarray(x[:2]),*y])
        if positions:poam.update(jnp.asarray(positions),jnp.asarray(observations),jax.random.PRNGKey(args.seed+3000+cycle),train_steps=cfg.poam_train_steps,relocate=True)
        publish()
        print(json.dumps(dict(event='executed',id=ident,t=clock,cost=cumulative,contact=contact,wall_seconds=time.perf_counter()-start)),flush=True)
    snapshot_belief(poam.planning_belief(),clock);publish(True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scenario',type=int,choices=(1,2),required=True)
    p.add_argument('--method',choices=('aware','baseline'),required=True)
    p.add_argument('--site',type=Path,default=m.physical.SITE);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preview',type=Path,help='Optional live JSON-only preview directory; raw arrays stay in the research output.')
    p.add_argument('--resume',action='store_true',help='Extend an exact recorded prefix; reconstruct and verify POAM from its actual measurements.')
    p.add_argument('--seed',type=int,default=8101);p.add_argument('--seconds',type=float,default=60.)
    p.add_argument('--horizon',type=float,default=30.);p.add_argument('--levels',type=int,default=4)
    p.add_argument('--commit',type=float,default=5.,help='MPC commitment, matching the old successful 5 s runs; physical and low-level periods are unchanged.')
    p.add_argument('--proposals',type=int,default=16);run(p.parse_args())
