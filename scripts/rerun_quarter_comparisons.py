"""Fresh matched five-minute experiments after the execution-filter correction.

Every method/environment is rerun from t=0. Previous runs and failed attempts
remain in separate directories; no trajectory or performance is reused.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import jax
from run_full_control_quarter import run,source_hashes,write
from voi_diffusion_ipp.quarter.continuous import ControlSettings

CASES=[('two','crosswind'),('multi','crosswind'),('two','forward'),
       ('two','reverse'),('two','commitment')]
METHODS=['proposed','blind','blind_filter']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='output/european_quarter/rrt_mbd_execution_300_v1')
    p.add_argument('--resume',action='store_true',help='Skip only completed runs with identical configuration and source hashes')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    old=json.loads(Path('output/european_quarter/rrt_mbd_runtime_300/two_forward_proposed_8101.json').read_text())
    cfg=ControlSettings(**old['config'])
    write(out/'experiment.json',dict(description='Fresh 300 s missions: common hard execution filter for proposed and filtered-blind; unchanged soft planning filter',
        cases=CASES,methods=METHODS,config=asdict(cfg),source_hashes=source_hashes(),seconds=300,seed=8101))
    for layout,case in CASES:
        for method in METHODS:
            target=out/f'{layout}_{case}_{method}_8101.json'
            if target.exists():
                d=json.loads(target.read_text())
                if not a.resume:raise FileExistsError(f'Refusing to overwrite {target}')
                if d['config']!=asdict(cfg) or d['source_hashes']!=source_hashes():raise ValueError('Cannot mix changed configuration/source in one experiment')
                if d.get('completed'):
                    print('PRESERVED COMPLETED RUN',target,flush=True);continue
                raise ValueError(f'Incomplete run must be preserved before restarting: {target}')
            print(f'START {layout}/{case}/{method}',flush=True)
            result=run(layout,case,method,300.,8101,cfg,out)
            print(f'END {layout}/{case}/{method}: success={result["success"]}, time={result["completed_seconds"]:.2f}, cost={result["cost"]:.6f}',flush=True)
            jax.clear_caches()

if __name__=='__main__':main()
