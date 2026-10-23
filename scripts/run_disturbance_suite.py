"""Paired actual nonlinear missions across distinct monitoring tasks.

Only disturbance-aware expectation and disturbance-blind planning are compared.
Both retain the same learned, filtered physical controller, route availability,
actuator limits, estimator, sensing model and optimizer budget. No optimism.
"""
import argparse
from dataclasses import replace,asdict
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
jax.config.update('jax_compilation_cache_dir',str(ROOT/'.cache/jax_suite'))
from run_compact_mbd_study import configuration
from run_full_control_quarter import run,write,source_hashes
from voi_diffusion_ipp.quarter.benchmark_environments import SCENARIOS,BY_NAME,describe,make_environment,flow,flow_parameters,survey_locations
from voi_diffusion_ipp.quarter.suite_engine import SuiteEngine


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',nargs='+',default=[s.name for s in SCENARIOS],choices=list(BY_NAME))
    p.add_argument('--seconds',type=float,default=300.)
    p.add_argument('--seed',type=int,default=9101)
    p.add_argument('--horizon',type=float,default=16.)
    p.add_argument('--dt',type=float,default=.25)
    p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--modes',type=int,default=13)
    p.add_argument('--proposals',type=int,default=26)
    p.add_argument('--worlds',type=int,default=2)
    p.add_argument('--validation-worlds',type=int,default=4)
    p.add_argument('--temperature',type=float,default=.06)
    p.add_argument('--prior-std',type=float,default=1.8)
    p.add_argument('--survey-count',type=int,default=64)
    p.add_argument('--inducing-count',type=int,default=64)
    p.add_argument('--short-length',type=float,default=.6)
    p.add_argument('--long-length',type=float,default=1.8)
    p.add_argument('--prediction-filter',choices=['soft','hard'],default='soft')
    p.add_argument('--prediction-substeps',type=int,default=1)
    p.add_argument('--validation-batch-size',type=int,default=0)
    p.add_argument('--planning-value-margin',type=float,default=.4)
    p.add_argument('--execution-margin',type=float,default=.7)
    p.add_argument('--execution-dt',type=float,default=.05)
    p.add_argument('--planning-clearance',type=float,default=.6)
    p.add_argument('--sensor-half-angle-degrees',type=float,default=25.)
    p.add_argument('--methods',nargs='+',choices=['aware','blind_filter'],default=['aware','blind_filter'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-directory',type=Path)
    args=p.parse_args()
    cfg=configuration(geometry='suite',seed=args.seed,horizon=args.horizon,dt=args.dt,
        iterations=args.iterations,mode_count=args.modes,proposals=args.proposals,
        batch_size=args.proposals,worlds=args.worlds,validation_worlds=args.validation_worlds,
        temperature=args.temperature,prior_std=args.prior_std,optimism=1.,mixture=0.,
        prediction_filter=args.prediction_filter,slack_weight=15.,planning_clearance=args.planning_clearance,
        safety_margin=args.planning_value_margin,execution_margin=args.execution_margin,rrt_visits=2,
        execution_dt=args.execution_dt,
        prediction_substeps=args.prediction_substeps,
        validation_batch_size=args.validation_batch_size,
        sensor_half_angle_degrees=args.sensor_half_angle_degrees,
        online_estimation=True,observation_period=1.,guidance_mode='off',
        execution_filter='hard',common_construction=True,wind_inducing_count=args.inducing_count,wind_kernel_lengths=(args.short_length,args.long_length))
    if abs(args.seconds/cfg.commit-round(args.seconds/cfg.commit))>1e-6:raise ValueError('Duration must be a commit multiple')
    args.output.mkdir(parents=True,exist_ok=True)
    sources={**source_hashes(),'scripts/run_disturbance_suite.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest={'config':asdict(cfg),'cases':[describe(n) for n in args.cases],'methods':args.methods,
        'mission_seconds':args.seconds,'seed':args.seed,'source_hashes':sources,
        'initial_information':f'{args.survey_count} synthetic noisy pre-mission stations selected only from known geometry; identical data shared by both methods',
        'comparison':'Expected disturbance-aware monitoring versus zero-disturbance planning; common wind-informed execution filter',
        'fidelity':'Unmodified production full-trajectory MBD, nonlinear 22-state quadrotor/motors at100Hz, control20Hz, finite visibility-limited footprint; all worlds are synthetic',
        'pad_notes':'Environment arrays have8 obstacle slots and6 goal slots. Padding is outside bounds. Original monitoring cells are duplicated to keep batch shapes, preserving their average cost.'}
    manifest_path=args.output/'experiment.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:raise ValueError('Preserve prior sources/configuration; use a fresh output folder')
    write(manifest_path,manifest)
    for name,digest in sources.items():
        data=(ROOT/name).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
        dest=args.output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    engine=SuiteEngine(cfg)
    for case in args.cases:
        env=make_environment(case)
        positions=survey_locations(env,args.survey_count)
        # Same measurement noise stream and geometry-only station rule for both.
        observed=np.asarray(flow(jnp.asarray(positions),flow_parameters(BY_NAME[case]))).copy()
        observed+=cfg.observation_std*np.random.default_rng(args.seed+7001).normal(size=observed.shape)
        data=np.c_[positions,observed]
        case_cfg=replace(cfg,wind_strength=BY_NAME[case].strength,initial_wind_observations=tuple(tuple(row) for row in data.tolist()))
        case_dir=args.output/case;case_dir.mkdir(exist_ok=True)
        write(case_dir/'initial_stations.json',{'locations_and_measurements':data.tolist(),'selection':'space-filling free geometry, no field values used for location selection'})
        for method in args.methods:
            dest=case_dir/f'two_{case}_{method}_{args.seed}.json'
            if dest.exists():
                old=json.loads(dest.read_text())
                if old['completed']:print('PRESERVED COMPLETE',case,method,flush=True);continue
                raise ValueError(f'Incomplete run at {dest}; explicit resume required')
            print('START',case,method,'seed',args.seed,'mission',args.seconds,flush=True)
            start=time.perf_counter()
            prefix=args.resume_directory/case/f'two_{case}_{method}_{args.seed}.json' if args.resume_directory else None
            result=run('two',case,method,args.seconds,args.seed,case_cfg,case_dir,engine=engine,resume=prefix)
            print('RESULT',case,method,'success',result['success'],'cost',result['cost'],'wall',time.perf_counter()-start,flush=True)
        rows=[]
        for method in args.methods:
            d=json.loads((case_dir/f'two_{case}_{method}_{args.seed}.json').read_text())
            warm=[r['complete_planning_seconds'] for r in d['plans'][1:]]
            rows.append({'method':method,'cost':d['cost'],'safe':d['success'],'completed_seconds':d['completed_seconds'],
                'warm_planning_median':float(np.median(warm)) if warm else None})
        write(case_dir/'paired_summary.json',{'case':describe(case),'results':rows,
            'reduction_percent':100*(rows[1]['cost']-rows[0]['cost'])/rows[1]['cost'] if len(rows)==2 else None})


if __name__=='__main__':main()
