"""Audit and package the deterministic runs without replacing older data."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--site',type=Path,required=True);args=p.parse_args()
    out=args.site/'data/monitoring-known-300'
    if (out/'manifest.json').exists():raise RuntimeError('Refusing to replace a packaged comparison')
    out.mkdir(parents=True,exist_ok=True)
    checks=[];runs=[]
    env=json.loads((args.directory/'environment.json').read_text())
    for scenario in (1,2):
        for method in ('aware','baseline'):
            ident=f's{scenario}-{method}';record=json.loads((args.directory/f'{ident}.json').read_text())
            protocol=json.loads((args.directory/f'{ident}-protocol.json').read_text())
            raw=np.load(args.directory/f'{ident}-execution.npz')['samples']
            assert record['summary']['completed'] and raw[-1,0]==300 and len(raw)==6001
            assert np.all(np.isfinite(raw)) and np.max(np.abs(raw[:,5:7]))<=1.6+1e-8
            assert np.allclose(np.diff(raw[:,0]),.05)
            q=np.array([f['q'] for f in record['clarity_frames']])
            assert q.shape==(601,len(env['cells'])) and q.min()>=0 and q.max()<=1
            mean_deficit=np.maximum(np.array(env['targets'])[None]-q,0).mean(1)
            assert np.max(abs(mean_deficit-raw[::10,13]))<1e-6
            geometry=np.array(env['obstacles']);positions=raw[:,1:3];bounds=np.array(env['bounds'])
            clear=np.minimum(np.min(np.linalg.norm(positions[:,None]-geometry[None,:,:2],axis=2)-geometry[:,2],axis=1),np.min(np.c_[positions-bounds[0],bounds[1]-positions],axis=1))
            # Column 12 is the minimum over the preceding physical substeps,
            # including their initial state; it need not equal endpoint clearance.
            assert np.all(clear>=raw[:,12]-1e-8)
            if record['summary']['contact'] is None:assert np.all(clear>0) and np.all(raw[:,18]==1)
            warm=[r['wall_seconds'] for r in record['plans'] if not r['timing']['compilation_included']]
            assert len(record['plans'])==60 and max(warm)<1
            assert protocol['uncertainty']==0 and protocol['legacy_settings']['construction_worlds']==1
            sha=hashlib.sha256((args.site/f'data/scenario{scenario}/stationary-velocity.f32').read_bytes()).hexdigest()
            assert sha==protocol['wind_sha256']
            checks.append(dict(id=ident,seconds=300,finite=True,actuator_bound=True,clarity_range=[float(q.min()),float(q.max())],
                clarity_deficit_consistent=True,unchanged_wind_sha256=sha,first_visit_s=record['summary']['first_visit_s'],
                contact=record['summary']['contact'],infeasible_execution_steps=record['summary']['infeasible_execution_steps']))
            timing=dict(min_s=min(warm),median_s=float(np.median(warm)),max_s=max(warm),calls=len(warm),
                initialization_median_s=float(np.median([r['timing']['initialization'] for r in record['plans'] if not r['timing']['compilation_included']])),
                mbd_forecast_median_s=float(np.median([r['timing']['optimization_validation'] for r in record['plans'] if not r['timing']['compilation_included']])),
                first_compilation_s=[r['wall_seconds'] for r in record['plans'] if r['timing']['compilation_included']],
                execution_block_median_s=float(np.median([r['seconds'] for r in record['execution_timing'] if not r['compilation_included']])))
            payload=json.dumps(record,separators=(',',':'),allow_nan=False).encode()
            (out/f'{ident}.json.gz').write_bytes(gzip.compress(payload,mtime=0))
            shutil.copyfile(args.directory/f'{ident}-protocol.json',out/f'{ident}-protocol.json')
            runs.append(dict(id=ident,scenario=scenario,method=method,file=f'{ident}.json.gz',decoded_sha256=hashlib.sha256(payload).hexdigest(),summary=record['summary'],timing=timing))
    audit=dict(checks=checks,model='Original physical double integrator plus quadratic air force; exact stored stationary wind known; full spline RPCBF; whole-environment clarity ODE.',
        limits='One development seed per scenario. Relaxed forecasting and hard execution have different control periods. No global safety guarantee is inferred from no contact.')
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    shutil.copyfile(args.directory/'environment.json',out/'environment.json')
    manifest=dict(duration_s=300,state='complete',implementation='known-wind-mbd',initial_wind_knowledge='known',uncertainty=0,
        planning_horizon_s=15,replan_s=5,control_coordinates=120,runtime_limit_s=1,runs=runs,
        planning='Original 7-mode MBD, 2 levels and 14 proposals per level; one known field. Select the exact deterministic search incumbent. One selected forecast retained for display.',
        execution='Full hard RPCBF, one known field, 0.06 m margin, 0.05 s filter period, 0.01 s physics and clarity.',
        no_learning=True,no_exploration=True,source_available=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    root=Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in (root/'src').rglob('*.py'):z.write(f,str(f.relative_to(root)))
        for name in ('known_monitoring','run_known_monitoring','package_known_monitoring','legacy_monitoring','monitoring_mbd','monitoring_routes','robot_review','run_monitoring_mbd'):
            f=root/f'scripts/airflow/{name}.py';z.write(f,str(f.relative_to(root)))
        z.write(root/'pyproject.toml','pyproject.toml')
        for scenario in (1,2):
            for name in ('metadata.json','stationary-velocity.f32'):
                z.write(args.site/f'data/scenario{scenario}/{name}',f'scenario-site/data/scenario{scenario}/{name}')
        z.writestr('REPRODUCE.txt','Install pyproject.toml dependencies. Run from the extracted project:\nROBOT_REVIEW_SITE=scenario-site PYTHONPATH=src:scripts/airflow python scripts/airflow/run_known_monitoring.py --output reproduced --seconds 300 --wall-limit 300\nWarm planning stops above 1 s; compilation is separate. Existing outputs are never overwritten.\n')
    print(json.dumps(dict(runs=[dict(id=r['id'],timing=r['timing'],summary=r['summary']) for r in runs]),indent=2))

if __name__=='__main__':main()
