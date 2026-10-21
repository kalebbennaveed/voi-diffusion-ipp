"""Reproduce the review configuration without requiring existing local data."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from run_full_control_quarter import run, write
from voi_diffusion_ipp.quarter.continuous import ControlSettings


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--case',choices=['forward','reverse','crosswind','commitment','calm_a'],default='forward')
    p.add_argument('--methods',nargs='+',choices=['proposed','blind','blind_filter','aware','optimistic'],default=['proposed'])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    profile=json.loads((root/'docs/compute_review/current45_dynamic.json').read_text())
    cfg=ControlSettings(**profile['config'])
    if abs(args.seconds/cfg.commit-round(args.seconds/cfg.commit))>1e-6:
        raise ValueError('Recording duration must be divisible by the replanning interval')
    args.output.mkdir(parents=True,exist_ok=True)
    for method in args.methods:
        path=args.output/f'two_{args.case}_{method}_{cfg.seed}.json'
        if path.exists():raise FileExistsError(f'Preserving existing recording: {path}')
    write(args.output/'review_configuration.json',dict(config=asdict(cfg),reference_profile=str(profile['record'])))
    for method in args.methods:
        run('two',args.case,method,args.seconds,cfg.seed,cfg,args.output,audit_all=True)


if __name__=='__main__':main()
