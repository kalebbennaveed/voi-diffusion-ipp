"""Publish completed matched missions and their exact reproduction inputs."""
import argparse,hashlib,json,shutil,zipfile,io
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def package(directory,site):
    out=site/'data/monitoring';out.mkdir(parents=True,exist_ok=True)
    audit=json.loads((directory/'audit.json').read_text())
    runs=[];duration=None
    for scenario in (1,2):
        for method in ('aware','baseline'):
            ident=f's{scenario}-{method}';record=json.loads((directory/(ident+'.json')).read_text())
            assert record['summary']['completed']
            protocol=json.loads((directory/(ident+'-protocol.json')).read_text())
            run_duration=protocol['settings']['mission_seconds']
            assert record['summary']['simulated_seconds']==run_duration
            if duration is None:duration=run_duration
            assert duration==run_duration
            for path,digest in protocol['source_sha256'].items():
                assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
            assert protocol['wind_sha256']==hashlib.sha256((site/f'data/scenario{scenario}/stationary-velocity.f32').read_bytes()).hexdigest()
            for suffix in ('.json','-protocol.json'):shutil.copyfile(directory/(ident+suffix),out/(ident+suffix))
            runs.append(dict(id=ident,scenario=scenario,method=method,summary=record['summary'],protocol=ident+'-protocol.json'))
    for name in ('environment.json','audit.json'):shutil.copyfile(directory/name,out/name)
    if (directory/'comparison-diagnostics.json').exists():shutil.copyfile(directory/'comparison-diagnostics.json',out/'comparison-diagnostics.json')
    manifest=dict(schema=1,comparison='Frozen-current-belief expected-cost MBD with relaxed RPCBF planning versus still-air MBD with RPCBF only after optimization',
        duration_s=duration,runs=runs,verification=audit,planner_settings=protocol['settings'],method_version='online-coarse-multistart-tracking',initializer_version='bottleneck-cuts',
        preservation='Existing wind animations, three-route robot comparison, both directions and both scenarios remain above this section.',
        simulation_clock='Simulation pauses during optimization. Wall time is recorded; no real-time claim.',
        download='Source, field inputs, protocols, every held control and all post-start wind observations. The initial local measurement is reproduced from the supplied field and seed. Full-precision 20 Hz state/clarity records, candidate histories and derived POAM matrices remain in the local raw NPZ files; the supplied source reproduces them. Physics and clarity are integrated at 100 Hz.')
    for name in ('coarse_initialization_benchmark.json','monitoring_implementation_checks.json','coarse_method_review.json','monitoring_physical_margin_diagnostic.json'):
        path=directory.parent/name
        if path.exists():shutil.copyfile(path,out/name)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT/'src').rglob('*.py')):z.write(p,str(p.relative_to(ROOT)))
        for name in ('monitoring_mbd.py','monitoring_routes.py','run_monitoring_mbd.py','audit_monitoring_mbd.py','package_monitoring_mbd.py','summarize_monitoring_comparison.py','robot_review.py'):
            p=ROOT/'scripts/airflow'/name;z.write(p,str(p.relative_to(ROOT)))
        z.write(ROOT/'pyproject.toml','pyproject.toml')
        z.write(ROOT/'tests/test_monitoring_mbd.py','tests/test_monitoring_mbd.py')
        for name in ('test_mbd.py','test_mbd_target.py'):
            z.write(ROOT/'tests'/name,'tests/'+name)
        z.write(ROOT/'docs/monitoring_mbd_audit.md','docs/monitoring_mbd_audit.md')
        for p in [site/'app.js',*[site/f'data/scenario{s}/{name}' for s in (1,2) for name in ('metadata.json','stationary-velocity.f32')]]:
            z.write(p,'scenario-site/'+str(p.relative_to(site)))
        for p in out.glob('*.json'):z.write(p,'results/'+p.name)
        for run in runs:
            ident=run['id'];raw=np.load(directory/(ident+'-execution.npz'))
            b=io.BytesIO();np.savez_compressed(b,samples=raw['samples'],observation_rows=raw['observation_rows'])
            z.writestr('results/'+ident+'-controls-observations.npz',b.getvalue())
        z.writestr('REPRODUCE.txt',
            'Use Python 3.11 and install pyproject.toml dependencies. Set PYTHONPATH=src.\n'
            'For each S=1,2 and M=aware,baseline:\n'
            f'python scripts/airflow/run_monitoring_mbd.py --scenario S --method M --site scenario-site --output reproduced --seconds {duration:g} --horizon 30 --commit 5 --levels 4 --proposals 16 --seed 8101\n'
            'python scripts/airflow/audit_monitoring_mbd.py --directory reproduced --site scenario-site\n'
            'This is computationally intensive: full direct POAM and spline-filtered rollouts are retained.\n'
            'The compact controls-observations NPZ files contain every held control and post-start measurement; the initial local measurement is reproduced from the protocol seed and supplied wind field.\n'
            'Inspect every contact, infeasible step, invalid candidate and fallback in the exported records.\n')
    print(json.dumps(dict(published=str(out),runs=len(runs),archive_bytes=(out/'source.zip').stat().st_size)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--site',type=Path,required=True)
    args=p.parse_args();package(args.directory,args.site)
