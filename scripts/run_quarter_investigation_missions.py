"""Fresh matched physical missions for the planner investigation, no viewer build."""
import argparse,json,hashlib,shutil
from pathlib import Path
from dataclasses import replace,asdict
import jax
from run_full_control_quarter import run,source_hashes,write
from screen_quarter_corrections import variants
from voi_diffusion_ipp.quarter.continuous import ControlSettings

CASES={'forward':('two','forward'),'reverse':('two','reverse'),'crosswind':('two','crosswind'),
       'large_east':('two','commitment'),'calm_a':('two','calm_a'),'multi':('multi','crosswind')}

def main():
 p=argparse.ArgumentParser();p.add_argument('--preset',default='calibrated');p.add_argument('--cases',nargs='+',default=list(CASES));p.add_argument('--methods',nargs='+',default=['proposed','blind','blind_filter']);p.add_argument('--seconds',type=float,default=300.);p.add_argument('--seed',type=int,default=8101);p.add_argument('--resume-source',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 old=json.loads(Path('output/european_quarter/rrt_mbd_execution_300/two_forward_proposed_8101.json').read_text());base=replace(variants(ControlSettings(**old['config']))[a.preset],seed=a.seed);a.output.mkdir(parents=True,exist_ok=True)
 root=Path(__file__).resolve().parents[1];hashes=source_hashes();snapshot=a.output/'source_snapshot'
 # A directory represents one loaded source revision. Never overwrite another
 # experiment's provenance before noticing an incompatible existing run.
 for name,digest in hashes.items():
  path=snapshot/name
  if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
   raise FileExistsError(f'Preserve the existing source snapshot; use a new output directory: {path}')
 for name,digest in hashes.items():
  data=(root/name).read_bytes()
  if hashlib.sha256(data).hexdigest()!=digest:raise RuntimeError('Source changed during sweep')
  path=snapshot/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
 manifest=a.output/f'experiment_{"_".join(a.methods)}_{"_".join(a.cases)}.json'
 write(manifest,dict(preset=a.preset,base_config=asdict(base),cases=a.cases,methods=a.methods,seconds=a.seconds,seed=a.seed,source_hashes=hashes,
    notes='Development tuning, single seed. Six physical cases include preserved all-windy larger east and a separately identified calm-A larger-east case. Four-site execution margin is 0.70 m for all methods; other cases use 0.55 m.'))
 for name in a.cases:
  layout,case=CASES[name];cfg=replace(base,execution_margin=max(base.execution_margin,.7 if layout=='multi' else .55))
  for method in a.methods:
   target=a.output/f'{layout}_{case}_{method}_{a.seed}.json'
   if target.exists():
    d=json.loads(target.read_text())
    if d.get('completed') and d['config']==asdict(cfg):print('PRESERVED',target,flush=True);continue
    raise FileExistsError(f'Preserve incomplete/different run: {target}')
   if shutil.disk_usage(a.output).free<120_000_000:raise RuntimeError('Less than 120 MB free; preserve checkpoint and recover duplicate storage before continuing')
   resume=None
   if a.resume_source:
    prior=json.loads(a.resume_source.read_text())
    if (prior['layout'],prior['case'],prior['method'],prior['seed'])==(layout,case,method,a.seed):resume=a.resume_source
   print('START',a.preset,layout,case,method,flush=True)
   d=run(layout,case,method,a.seconds,a.seed,cfg,a.output,resume=resume)
   print('RESULT',layout,case,method,d['success'],d['cost'],d['completed_seconds'],flush=True)
   jax.clear_caches()

if __name__=='__main__':main()
