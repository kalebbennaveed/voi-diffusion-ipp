"""Bounded deterministic MBD experiment; all outputs go to a new directory."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
import jax
import jax.numpy as jnp
import known_monitoring as known
import legacy_monitoring as legacy
import monitoring_mbd as m
from run_monitoring_mbd import write, sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--scenarios', default='1,2')
    parser.add_argument('--methods', default='aware,baseline')
    parser.add_argument('--wall-limit', type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob('s*-*.json')):
        raise RuntimeError('Refusing to overwrite existing recordings')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Experiment wall-time budget reached')))
    signal.alarm(args.wall_limit)
    cfg = known.settings(); cfg.__post_init__()
    pcfg = replace(legacy.physical_settings(cfg), mission_seconds=args.seconds,
                   safety_margin=cfg.safety_margin, inner_samples=1)
    pcfg.__post_init__()
    env, grid = m.environment(pcfg); env32 = legacy.float32(env)
    execute = known.execution_block(env, cfg, pcfg)
    with jax.experimental.disable_x64():
        planners = {method:known.optimizer(env32, cfg, method=='aware')
                    for method in args.methods.split(',')}
    write(args.output/'environment.json', dict(grid=grid['grid'], dx_m=.125,
        free_indices=grid['free_indices'], cells=grid['cells'], membership=grid['membership'],
        targets=env.targets, process_noise=env.process_noise, initial_clarity=pcfg.initial_q,
        bounds=env.bounds, obstacles=env.obstacles,
        regions=[dict(label='A',bounds=[[4.98,2.295],[8,4.625]]),dict(label='B',bounds=[[0,0],[2.9,1.755]])]))
    hashes = sources()
    for path in [Path(__file__), Path(known.__file__), Path(legacy.__file__),
                 *[m.physical.ROOT/f'src/voi_diffusion_ipp/quarter/{n}.py' for n in ('continuous','multimode','mbd_core','diffusion')],
                 m.physical.ROOT/'src/voi_diffusion_ipp/wind/oracle.py']:
        hashes[str(path.relative_to(m.physical.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    compiled = set(); execution_compiled = False
    reports = []
    for scenario in map(int,args.scenarios.split(',')):
        field = m.physical.load_wind(scenario); field32 = legacy.float32(field)
        for method, plan in planners.items():
            ident = f's{scenario}-{method}'; started = time.perf_counter()
            protocol = dict(id=ident,scenario=scenario,method=method,seed=8101,
                initial_wind_knowledge='known',uncertainty=0,settings=asdict(pcfg),
                legacy_settings=asdict(cfg),source_sha256=hashes,
                wind_sha256=hashlib.sha256((m.physical.SITE/f'data/scenario{scenario}/stationary-velocity.f32').read_bytes()).hexdigest(),
                wind_evaluation='Same FixedWind cell-center bilinear queries in forecast, RPCBF and plant; no resampling.',
                execution_filter='hard RPCBF, one deterministic field, 0.06 m margin',
                planning_filter='relaxed RPCBF for aware; none and still air for baseline',
                removed=['GP/posterior','survey','wind observations','wind sampling','duplicate deterministic validation','exploration objective'],
                runtime_limit_s=1.,numerics=dict(planning_dt=.25,planning_dtype='float32',physics_dt=.01,physics_dtype='float64',execution_filter_dt=.05))
            write(args.output/f'{ident}-protocol.json', protocol)
            x = jnp.array([1.45,.875,0.,0.]); q = jnp.full(len(env.cells),pcfg.initial_q)
            alive = jnp.array(True); previous = None; cost = 0.; clock = 0.
            samples=[]; qframes=[]; plans=[]; executions=[]; contact=None
            a_cells=np.asarray(grid['membership'][:,0]);b_cells=np.asarray(grid['membership'][:,1])
            def append(t,x,q,u,nominal,slack,correction,feasible,gap,alive):
                samples.append([t,*x,*u,*nominal,float(slack),float(correction),int(feasible),float(gap),
                    float(np.maximum(np.asarray(env.targets)-q,0).mean()),cost,
                    float(q[a_cells].mean()),float(q[b_cells].mean()),float(q.mean()),int(alive)])
                if abs(t/.5-round(t/.5))<1e-7:qframes.append(dict(t=t,q=np.round(q,6)))
            append(0.,np.asarray(x),np.asarray(q),np.zeros(2),np.zeros(2),0,0,True,env.clearance(x),True)
            for cycle in range(round(args.seconds/cfg.commit)):
                t0=time.perf_counter()
                seeds, shifted, metadata=legacy.seed_bank(x,q,env,cfg,previous)
                seeds,xp,qp,shifted=legacy.float32((seeds,x,q,shifted))
                initialization=time.perf_counter()-t0
                first=method not in compiled
                start=time.perf_counter()
                with jax.experimental.disable_x64():
                    result=plan(xp,qp,field32,seeds,shifted,jax.random.PRNGKey(8101+2000+cycle))
                    jax.block_until_ready(result)
                computation=time.perf_counter()-start; total=time.perf_counter()-t0
                compiled.add(method)
                theta,score,search,forecast=result
                if not np.isfinite(float(score)):raise RuntimeError('No finite plan')
                if not first and total>1.:
                    write(args.output/f'{ident}-runtime-stop.json',dict(t=clock,seconds=total,limit=1.))
                    raise RuntimeError(f'{ident}: warm planning {total:.3f} s exceeds 1 s limit')
                origin='initialization' if np.any(np.all(np.asarray(search.initial_parameters)==np.asarray(theta)[None],axis=1)) else 'diffusion'
                plans.append(dict(t=clock,wall_seconds=total,timing=dict(initialization=initialization,
                    field_sampling=0.,optimization_validation=computation,compilation_included=first),
                    expected_cost=float(score),selected_origin=origin,evaluations=int(search.evaluations),
                    validation_evaluations=0,initialization=metadata,seed_costs=np.asarray(search.initial_field_costs)[:,0],
                    trace=np.asarray(search.trace),predicted_states=np.asarray(forecast.x)[None],
                    predicted_contacts=[bool(forecast.terminated)],predicted_min_clearance=[float(forecast.min_clearance)],
                    predicted_monitoring_cost=[float(forecast.monitor)],predicted_safety_cost=[float(forecast.penalty)]))
                np.savez_compressed(args.output/f'{ident}-plan-{cycle:03d}.npz',theta=np.asarray(theta),
                    initial_state=np.asarray(x),initial_clarity=np.asarray(q),
                    **{'search_'+k:np.asarray(v) for k,v in zip(search._fields,search)})
                ts=time.perf_counter();end,rows=execute(x,q,theta.astype(jnp.float64),field,alive);jax.block_until_ready(end)
                executions.append(dict(t=clock,seconds=time.perf_counter()-ts,compilation_included=not execution_compiled))
                execution_compiled=True
                xs,qs,us,nominals,slacks,corrections,feasible,gaps,costs,living,hits=map(np.asarray,rows)
                for k in range(len(xs)):
                    if contact is None and np.isfinite(hits[k]):contact=dict(time_s=clock+k*pcfg.execution_dt+float(hits[k]),position_m=xs[k,:2].tolist())
                    cost+=float(costs[k])
                    append(round(clock+(k+1)*pcfg.execution_dt,8),xs[k],qs[k],us[k],nominals[k],slacks[k],corrections[k],feasible[k],gaps[k],living[k])
                x,q,alive=end; previous=theta;clock=round(clock+cfg.commit,8)
                if cycle==0 or clock%30==0 or clock==args.seconds:
                    print(json.dumps(dict(event='progress',id=ident,t=clock,planning_s=round(total,4),compile=first,
                        x=np.asarray(x[:2]).tolist(),contact=contact,wall_s=round(time.perf_counter()-started,2))),flush=True)
            samples=np.asarray(samples)
            visits={r['label']:next((float(s[0]) for s in samples if np.all(s[1:3]>=r['bounds'][0]) and np.all(s[1:3]<=r['bounds'][1])),None)
                for r in [dict(label='A',bounds=[[4.98,2.295],[8,4.625]]),dict(label='B',bounds=[[0,0],[2.9,1.755]])]}
            summary=dict(completed=True,simulated_seconds=clock,planned_seconds=args.seconds,contact=contact,
                integrated_deficit=cost,min_clearance_m=float(samples[:,12].min()),infeasible_execution_steps=int(np.sum(samples[:,11]==0)),
                max_execution_slack=float(samples[:,9].max()),max_control_component_m_s2=float(np.abs(samples[:,5:7]).max()),
                fallback_periods=0,wall_seconds=time.perf_counter()-started,first_visit_s=visits,
                passage_crossings=m.physical.passage_crossings(samples[:,:5]),
                final_region_clarity=samples[-1,15:17],final_whole_clarity=samples[-1,17])
            record=dict(id=ident,scenario=scenario,method=method,samples=np.round(samples,7),clarity_frames=qframes,
                plans=plans,summary=summary,execution_timing=executions)
            write(args.output/f'{ident}.json',record)
            np.savez_compressed(args.output/f'{ident}-execution.npz',samples=samples)
            reports.append(dict(id=ident,summary=summary))
            write(args.output/'summary.json',reports)
            print(json.dumps(dict(event='completed',id=ident,visits=visits,contact=contact,cost=cost,
                wall_s=time.perf_counter()-started)),flush=True)
    signal.alarm(0)

if __name__=='__main__':main()
