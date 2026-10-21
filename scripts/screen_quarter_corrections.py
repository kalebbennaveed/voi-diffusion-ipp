"""Controlled fresh-flight ablations; no renderer and no prerecorded paths.

Every run starts from t=0 and retains raw search/physics logs. Variants are
development experiments, not held-out evidence. Output directories never mix
source revisions or silently overwrite prior results.
"""
import argparse,json,hashlib
from pathlib import Path
from dataclasses import replace,asdict
import jax
from run_full_control_quarter import run,write,source_hashes
from voi_diffusion_ipp.quarter.continuous import ControlSettings

def variants(base):
 legacy=replace(base,penalty_aggregation='tail',validate_initials=False,rrt_diversity=False,prediction_substeps=1,prediction_filter='soft',observation_period=5.)
 corrected=replace(base,penalty_aggregation='mean',validate_initials=True,rrt_diversity=True,prediction_substeps=1,prediction_filter='hard',observation_period=1.)
 return {
  'legacy':legacy,
  'matched_controller':replace(legacy,prediction_substeps=1,prediction_filter='hard'),
  'corrected':corrected,
  'mean_objective':replace(corrected,optimism=1.,mixture=0.),
  'eta75':replace(corrected,optimism=.75),
  'budget':replace(corrected,iterations=12,proposals=32,temperature=.3),
  'horizon60':replace(corrected,horizon=60.),
  'wind39':replace(corrected,wind_strength=3.9),
  'scale45':replace(corrected,environment_scale=.45),
  'noise10':replace(corrected,cosine_end=.10,iterations=12,proposals=32,temperature=.3),
  'no_penalty':replace(corrected,violation_weight=0.),
  'soft_penalty':replace(corrected,violation_weight=1.),
  'sub100ms':replace(corrected,prediction_substeps=2),
  'model_weights':replace(corrected,guidance_mode='off'),
  'executed_guidance':replace(corrected,guidance_mode='executed'),
  'prior12':replace(corrected,prior_std=1.2),
  'tuned':replace(corrected,environment_scale=.45,wind_strength=3.9,prior_std=1.2,guidance_mode='off',violation_weight=1.,temperature=.3),
  'calibrated':replace(corrected,prior_std=1.2,guidance_mode='off',violation_weight=1.,temperature=.3),
  'coarser':replace(corrected,dt=.5,horizon=45.,prediction_substeps=5),
  'online':replace(corrected,prior_std=1.2,guidance_mode='off',violation_weight=1.,temperature=.3,online_estimation=True),
  'online_margin':replace(corrected,prior_std=1.2,guidance_mode='off',violation_weight=1.,temperature=.3,online_estimation=True,execution_margin=.7),
  'supported':replace(corrected,prior_std=1.2,guidance_mode='off',violation_weight=1.,temperature=.3,online_estimation=True,wind_support_nodes=25,wind_relocate=True,terminate_predictions=True),
  'supported_margin':replace(corrected,prior_std=1.2,guidance_mode='off',violation_weight=1.,temperature=.3,online_estimation=True,wind_support_nodes=25,wind_relocate=True,execution_margin=.7,terminate_predictions=True),
 }

def main():
 p=argparse.ArgumentParser();p.add_argument('--variants',nargs='+',default=['matched_controller','corrected','mean_objective']);p.add_argument('--cases',nargs='+',default=['crosswind']);p.add_argument('--methods',nargs='+',default=['proposed']);p.add_argument('--seconds',type=float,default=120);p.add_argument('--seed',type=int,default=8101);p.add_argument('--layout',default='two');p.add_argument('--output',type=Path,default=Path('output/european_quarter/planner_investigation/screen120'));a=p.parse_args()
 old=json.loads(Path('output/european_quarter/rrt_mbd_execution_300/two_forward_proposed_8101.json').read_text());base=ControlSettings(**old['config']);bank=variants(base)
 for name in a.variants:
  cfg=replace(bank[name],seed=a.seed);out=a.output/name;out.mkdir(parents=True,exist_ok=True)
  manifest=out/'experiment.json';identity=dict(config=asdict(cfg),source_hashes=source_hashes(),seconds=a.seconds,seed=a.seed,variant=name)
  if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError(f'Source/configuration changed for {out}; use a new directory')
  write(manifest,identity)
  snapshot=out/'source_snapshot';root=Path(__file__).resolve().parents[1]
  for filename,digest in identity['source_hashes'].items():
   data=(root/filename).read_bytes()
   if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Source files changed during sweep; preserve outputs and start a new process/directory')
   target=snapshot/filename;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  for case in a.cases:
   for method in a.methods:
    path=out/f'{a.layout}_{case}_{method}_{a.seed}.json'
    if path.exists():
     d=json.loads(path.read_text())
     if d.get('completed'):print('PRESERVE',path,flush=True);continue
     raise ValueError(f'Incomplete experiment preserved at {path}; use a new output directory')
    print('START',name,case,method,flush=True)
    d=run(a.layout,case,method,a.seconds,a.seed,cfg,out,audit_all=True)
    print('RESULT',name,case,method,d['success'],d['cost'],d['completed_seconds'],flush=True)
    jax.clear_caches()

if __name__=='__main__':main()
