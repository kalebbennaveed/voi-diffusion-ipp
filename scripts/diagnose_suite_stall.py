"""One-call prediction/execution diagnostic from an archived reached state.

This does not produce mission performance claims. Every variant starts from the
same unrounded state and posterior, with the same search noise and geometry.
"""
import argparse,json,time
from dataclasses import replace
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from voi_diffusion_ipp.quarter.continuous import ControlSettings,continuation,nominal_reference
from voi_diffusion_ipp.quarter.environment import initial_belief,planning_sampler
from voi_diffusion_ipp.quarter.benchmark_environments import make_environment
from voi_diffusion_ipp.quarter.suite_engine import SuiteEngine
from voi_diffusion_ipp.quarter.modes import demonstrations,select_initial_modes

p=argparse.ArgumentParser();p.add_argument('record',type=Path);p.add_argument('--index',type=int,default=30);p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse',action='store_true');p.add_argument('--matched-only',action='store_true');args=p.parse_args()
d=json.loads(args.record.read_text());cfg=ControlSettings(**d['config']);env=make_environment(d['case'])
i=args.index
with np.load(args.record.parent/d['plans'][i-1]['execution_audit']) as a:
    packed=jnp.asarray(a['states'][-1]);q=jnp.asarray(a['clarity'][-1])
x=jnp.r_[packed[:2],packed[3:5]];b=initial_belief(cfg)
if args.matched_only:
    # Preserve the archived execution controller's exact projector, rather
    # than confounding model fidelity with a later numerical optimization.
    import importlib.util
    import voi_diffusion_ipp.safety.qp as qp
    legacy=args.record.parent.parent/'source_snapshot/src/voi_diffusion_ipp/safety/qp.py'
    spec=importlib.util.spec_from_file_location('archived_projector',legacy)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);qp.project=module.project
    with np.load(args.output/'soft_0.4.npz') as a:
        theta=jnp.asarray(a['theta']);mean=jnp.asarray(a['mean']);c=jnp.asarray(a['construction'])
        actual=a['actual'];earlier=a['predicted']
    cc=replace(cfg,prediction_filter='hard',prediction_substeps=5)
    fine=SuiteEngine(cc).prediction(env,False)(theta,x,q,mean,c,mean)
    np.savez_compressed(args.output/'same_decision.npz',actual=actual,earlier=earlier,
        fine_predicted=np.asarray(fine.x),theta=np.asarray(theta))
    print('Matched decision end-position error',float(np.linalg.norm(np.asarray(fine.x[cfg.commit_steps,:2])-actual[-1,:2])),flush=True)
    raise SystemExit
condition=jax.jit(lambda bb,p,y:bb.condition(p,y,relocate=cfg.wind_relocate))
for obs in d['observations'][:round(i*cfg.commit/cfg.observation_period)]:
    b=condition(b,jnp.array([obs['position']],jnp.float32),jnp.array([obs['measurement']],jnp.float32))
key=jax.random.PRNGKey(d['seed']);construction_key=jax.random.fold_in(key,271828)
for _ in range(i+1):key,ks,kp=jax.random.split(key,3)
mean,fields=planning_sampler(cfg)(b,ks,construction_key);fields.block_until_ready()
c=fields[:cfg.construction_worlds];w=fields[cfg.construction_worlds:cfg.construction_worlds+cfg.worlds];v=fields[-cfg.validation_worlds:]
bank,record=demonstrations(x,env,cfg,d['seed']+100003*i);initials,selection=select_initial_modes(bank,record,x,env,cfg,q)
previous=continuation(jnp.asarray(d['plans'][i-1]['policy']),cfg)
args.output.mkdir(parents=True,exist_ok=True);rows=[]
for margin,prediction_filter in [(.4,'soft'),(.7,'soft'),(.85,'soft'),(.7,'hard')]:
    cc=replace(cfg,safety_margin=margin,prediction_filter=prediction_filter);engine=SuiteEngine(cc)
    name=f'{prediction_filter}_{margin:g}'
    if args.reuse:
        with np.load(args.output/f'{name}.npz') as old:theta=jnp.asarray(old['theta'])
        r=(theta,jnp.nan);elapsed=0.
    else:
        start=time.perf_counter();r=engine.planner(env,'aware')(x,q,mean,c,w,v,previous,kp,jnp.array(1.),jnp.array(False),jnp.asarray(initials));r[0].block_until_ready();elapsed=time.perf_counter()-start
    forecast=engine.prediction(env,False)(r[0],x,q,mean,c,mean)
    fine_engine=SuiteEngine(replace(cc,prediction_substeps=5))
    fine=fine_engine.prediction(env,False)(r[0],x,q,mean,c,mean)
    # Five seconds from the same reached physical state, fixed current posterior.
    state=packed;path=[np.asarray(state[:3])]
    execute=engine.execution(env,d['case'])
    for k in range(round(cfg.commit/cfg.observation_period)):
        state,logs=execute(state,r[0],mean,c,x,k*round(cfg.observation_period/cfg.dt));state.block_until_ready();path.extend(np.asarray(logs[0][:,:3]))
    np.savez_compressed(args.output/f'{name}.npz',actual=np.asarray(path),predicted=np.asarray(forecast.x),fine_predicted=np.asarray(fine.x),nominal=np.asarray(nominal_reference(r[0],x,cc)[0]),theta=np.asarray(r[0]),mean=np.asarray(mean),construction=np.asarray(c))
    row=dict(name=name,predicted_cost=None if args.reuse else float(r[1]),cold_seconds=elapsed,initial_position=np.asarray(packed[:3]).tolist(),end_position=np.asarray(state[:3]).tolist(),displacement=float(jnp.linalg.norm(state[:2]-packed[:2])),max_slack=float(forecast.max_slack),coarse_prediction_at5s=np.asarray(forecast.x[cfg.commit_steps,:2]).tolist(),fine_prediction_at5s=np.asarray(fine.x[cfg.commit_steps,:2]).tolist())
    print(row,flush=True);rows.append(row)
(args.output/'diagnostic.json').write_text(json.dumps(rows,indent=2))
