"""Rerun all three four-site methods with additional execution clearance.

The 0.55 m trial contacts a wall at 264.44 s for filtered blind. Replaying that
same control sequence/state/posterior with 0.70 m leaves 0.164 m physical
clearance. This common adjustment is applied to every method from t=0, and
the failed trial is retained. It is development tuning, not held-out validation.
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import jax
from run_full_control_quarter import run,source_hashes,write
from voi_diffusion_ipp.quarter.continuous import ControlSettings

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('output/european_quarter/rrt_mbd_execution_four_site_300_v2'))
    parser.add_argument('--source',type=Path,default=Path('output/european_quarter/rrt_mbd_execution_300_v1/multi_crosswind_blind_filter_8101.json'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);source=args.source
    old=json.loads(source.read_text());cfg=replace(ControlSettings(**old['config']),execution_margin=.70)
    write(out/'experiment.json',dict(reason=__doc__,source=str(source),source_hashes=source_hashes(),
        changed_parameters=dict(execution_margin=dict(previous=.55,current=.70)),
        timing_note='This development sweep may overlap the other environment reruns; measured latency is not an isolated hardware benchmark.'))
    for method in ('proposed','blind','blind_filter'):
        path=out/f'multi_crosswind_{method}_8101.json'
        if path.exists():raise FileExistsError(f'Refusing to overwrite {path}')
        run('multi','crosswind',method,300.,8101,cfg,out)
        jax.clear_caches()

if __name__=='__main__':main()
