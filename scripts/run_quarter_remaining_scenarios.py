"""Run the missing matched 300 s cases with the recorded runtime configuration.

No scenario-dependent planner weights or prescribed route choices are used.
Completed failures are retained. Running again skips completed recordings only.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import jax

from run_full_control_quarter import run
from voi_diffusion_ipp.quarter.continuous import ControlSettings


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=Path('output/european_quarter/rrt_mbd_runtime_300'))
    args=parser.parse_args()
    source=json.loads((args.results/'two_forward_proposed_8101.json').read_text())
    cfg=ControlSettings(**source['config'])
    assert source['completed'] and source['mission_seconds']==300
    for layout,case in [('two','crosswind'),('two','commitment'),('multi','crosswind')]:
        for method in ['proposed','blind','blind_filter']:
            path=args.results/f'{layout}_{case}_{method}_{source["seed"]}.json'
            if path.exists():
                previous=json.loads(path.read_text())
                if previous.get('completed') and previous['config']==asdict(cfg) and previous['mission_seconds']==300:
                    print('RETAIN COMPLETED RECORDING',path,flush=True)
                    continue
                raise RuntimeError(f'Preserve and inspect the incomplete or mismatched recording before retrying: {path}')
            print('START',layout,case,method,'300 s',flush=True)
            run(layout,case,method,300.,source['seed'],cfg,args.results)
            jax.clear_caches()


if __name__=='__main__':main()
