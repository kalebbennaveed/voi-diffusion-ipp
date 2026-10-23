"""Matched optimistic/expected-cost flights with independent trajectory modes.

No archived value-of-learning objective is used. The complete posterior remains
fixed inside a plan, and actual 1 Hz observations update the runtime controller.
Full raw nonlinear execution records and mode search audits are preserved.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import jax

# Configure the persistent cache before importing modules that create JAX arrays.
if jax.config.jax_compilation_cache_dir is None:
    jax.config.update('jax_compilation_cache_dir',str(Path(__file__).resolve().parents[1]/'.cache/jax_compact'))

from run_full_control_quarter import run,source_hashes,write
from voi_diffusion_ipp.quarter.continuous import ControlSettings


def configuration(**changes):
    root=Path(__file__).resolve().parents[1]
    cfg=ControlSettings(**json.loads((root/'configs/compact_mbd.json').read_text())['config'])
    if 'proposals' in changes and 'batch_size' not in changes:
        changes['batch_size']=changes['proposals']
    return replace(cfg,**changes)


def main():
    defaults=configuration()
    p=argparse.ArgumentParser()
    p.add_argument('--cases',nargs='+',choices=['forward','reverse','crosswind'],default=['forward','reverse'])
    p.add_argument('--geometry',choices=['compact','compact_passages','compact_detour'],default='compact')
    p.add_argument('--detour-depth',type=float,default=2.)
    p.add_argument('--wind-at-start',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--initial-wind-data',type=Path)
    p.add_argument('--multimode',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--methods',nargs='+',choices=['proposed','aware','optimistic','fading','blind_filter'],default=['proposed','aware'])
    p.add_argument('--seconds',type=float,default=300.)
    p.add_argument('--seed',type=int,default=8101)
    p.add_argument('--iterations',type=int,default=defaults.iterations)
    p.add_argument('--proposals',type=int,default=defaults.proposals)
    p.add_argument('--horizon',type=float,default=10.)
    p.add_argument('--dt',type=float,default=.2)
    p.add_argument('--noise-end',type=float,default=.15)
    p.add_argument('--wind',type=float,default=3.1)
    p.add_argument('--temperature',type=float,default=defaults.temperature)
    p.add_argument('--prior-std',type=float,default=defaults.prior_std)
    p.add_argument('--optimism',type=float,default=.5)
    p.add_argument('--mixture',type=float,default=defaults.mixture)
    p.add_argument('--worlds',type=int,default=4)
    p.add_argument('--validation-worlds',type=int,default=8)
    p.add_argument('--modes',type=int,default=defaults.mode_count)
    p.add_argument('--initialization-effort',type=float,default=defaults.initialization_effort)
    p.add_argument('--prediction-filter',choices=['hard','soft'],default=None)
    p.add_argument('--slack-weight',type=float,default=None)
    p.add_argument('--planning-clearance',type=float,default=None)
    p.add_argument('--optimism-anneal-seconds',type=float,default=None)
    p.add_argument('--audit-all',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    cfg=configuration(seed=args.seed,iterations=args.iterations,proposals=args.proposals,
        horizon=args.horizon,dt=args.dt,cosine_end=args.noise_end,wind_strength=args.wind,
        temperature=args.temperature,prior_std=args.prior_std,optimism=args.optimism,
        mixture=args.mixture,worlds=args.worlds,validation_worlds=args.validation_worlds,
        mode_count=args.modes,initialization_effort=args.initialization_effort,
        optimism_anneal_seconds=args.optimism_anneal_seconds,geometry=args.geometry,multimode=args.multimode,detour_depth=args.detour_depth,wind_at_start=args.wind_at_start)
    overrides={name:value for name,value in dict(prediction_filter=args.prediction_filter,
        slack_weight=args.slack_weight,planning_clearance=args.planning_clearance).items() if value is not None}
    cfg=replace(cfg,**overrides)
    if args.initial_wind_data:
        data=json.loads(args.initial_wind_data.read_text())
        cfg=replace(cfg,initial_wind_observations=tuple(tuple(row) for row in data['observations']))
    if abs(args.seconds/cfg.commit-round(args.seconds/cfg.commit))>1e-6:
        raise ValueError('Mission duration must be divisible by the replanning interval')
    args.output.mkdir(parents=True,exist_ok=True)
    hashes={**source_hashes(),str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[1])):
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    config_path=Path(__file__).resolve().parents[1]/'configs/compact_mbd.json'
    hashes['configs/compact_mbd.json']=hashlib.sha256(config_path.read_bytes()).hexdigest()
    identity=dict(config=asdict(cfg),source_hashes=hashes,seconds=args.seconds,
        methods=args.methods,cases=args.cases,comparison='Proposed favorable-tail score versus expected-cost disturbance-aware score; same multi-mode search and vehicle')
    if args.initial_wind_data:
        payload=args.initial_wind_data.read_bytes()
        identity['initial_wind_data']=dict(source=str(args.initial_wind_data.resolve()),sha256=hashlib.sha256(payload).hexdigest(),saved='inputs/initial_wind_observations.json')
    manifest=args.output/f'experiment_{args.seed}_{"_".join(args.methods)}_{"_".join(args.cases)}.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:
        raise ValueError('Existing experiment has different source/settings; preserve it and use a new directory')
    write(manifest,identity)
    if args.initial_wind_data:
        destination=args.output/'inputs/initial_wind_observations.json'
        destination.parent.mkdir(exist_ok=True);destination.write_bytes(payload)
    root=Path(__file__).resolve().parents[1]
    for name,digest in identity['source_hashes'].items():
        data=(root/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Source changed during experiment setup')
        target=args.output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    for case in args.cases:
        for method in args.methods:
            path=args.output/f'two_{case}_{method}_{args.seed}.json'
            if path.exists():raise FileExistsError(f'Existing recording is preserved: {path}')
            print('START',case,method,args.seed,flush=True)
            result=run('two',case,method,args.seconds,args.seed,cfg,args.output,audit_all=args.audit_all)
            print('RESULT',case,method,args.seed,result['success'],result['cost'],result['completed_seconds'],flush=True)


if __name__=='__main__':main()
