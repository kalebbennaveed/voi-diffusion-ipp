"""Audit all declared unknown-wind runs and add a separate web dataset."""
import argparse,gzip,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np


def source_archive(dest,root,protocol,site):
    """Bundle recorded sources plus the local modules and fixed plant inputs."""
    files=set(protocol['source_sha256'])
    files.update(str(p.relative_to(root)) for p in (root/'scripts/airflow').glob('*.py'))
    files.update(['pyproject.toml','tests/test_implicit_monitoring.py',
                  'results/airflow/cluster_revision_20261103/v2/environment-spec.json'])
    index={}
    with zipfile.ZipFile(dest/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            path=root/name
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            if name in protocol['source_sha256']:assert sha==protocol['source_sha256'][name],name
            index[name]=dict(sha256=sha,recorded_before_run=name in protocol['source_sha256'])
            z.write(path,name)
        for scenario in [1,2]:
            for name in ['metadata.json','stationary-velocity.f32']:
                path=site/f'data/scenario{scenario}'/name
                z.write(path,f'airflow-site/revised/data/scenario{scenario}/{name}')
        z.writestr('source-index.json',json.dumps(index,indent=2))
        z.writestr('REPRODUCE.txt',
            'Run from this extracted source directory with the project dependencies installed.\n'
            'Set ROBOT_REVIEW_SITE to the absolute path of airflow-site in this archive.\n'
            'python scripts/airflow/run_implicit_monitoring.py --output results/new-implicit-run '
            '--seconds 300 --scenarios 1,2 --methods aware,baseline --cases center --wall-limit 300\n'
            'Choose a new output directory; the runner refuses to replace existing records.\n'
            'Warm calls above 1 second and the overall wall-time limit stop the experiment.\n'
            'First compilation is separately reported. Different hardware can exceed the limit.\n'
            'The fixed wind files are plant/sensor inputs only. Never pass them to planning or control.\n'
            'source-index.json distinguishes pre-run hashes from dependencies packaged afterwards.\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--site',type=Path,required=True);a=p.parse_args()
    dest=a.site/'data/implicit-monitoring-300'
    if (dest/'manifest.json').exists():raise RuntimeError('Preserving existing published experiment')
    dest.mkdir(exist_ok=True,parents=True)
    env=json.loads((a.directory/'environment.json').read_text());free=np.asarray(env['free_indices'])
    runs=[];checks=[]
    for scenario in [1,2]:
        for method in ['aware','baseline']:
            ident=f's{scenario}-center-{method}';path=a.directory/f'{ident}.json.gz'
            with gzip.open(path,'rt') as f:r=json.load(f)
            protocol=json.loads((a.directory/f'{ident}-protocol.json').read_text())
            s=np.asarray(r['samples']);assert r['summary']['completed'] and s[-1,0]==300 and len(s)==6001
            assert np.isfinite(s).all() and np.max(np.abs(s[:,5:7]))<=1.6+1e-6
            assert r['plans'][0]['observations']==0 and r['plans'][0]['planning_field_max_abs']==0
            assert r['belief_frames'][0]['observations']==0
            assert max(r['belief_frames'][0]['planning_std'])==0 and min(r['belief_frames'][0]['std'])>1.79
            assert len(r['observations'])>0 and min(o[0] for o in r['observations'])>0
            for frame in r['belief_frames']:
                assert frame['observations']==sum(o[0]<=frame['t']+1e-8 for o in r['observations'])
                assert np.all(np.asarray(frame['planning_std'])<=np.asarray(frame['std'])+2e-5)
            for plan in r['plans']:
                assert plan['observations']==sum(o[0]<=plan['t']+1e-8 for o in r['observations'])
                assert len(plan['predicted_states'])==2 and len(plan['seed_costs'])==8
                assert plan['evaluations']==43
                if not plan['timing']['compilation_included']:assert plan['wall_seconds']<=1
            q=np.array([f['q'] for f in r['clarity_frames']]);assert q.shape==(601,len(free)) and q.min()>=0 and q.max()<=1
            deficit=np.maximum(np.array(env['targets'])[None]-q,0).mean(axis=1)
            assert np.max(abs(deficit-s[::10,13]))<1e-6
            warm=[p for p in r['plans'] if not p['timing']['compilation_included']]
            timings={k:dict(median_s=float(np.median([p['timing'][k] for p in warm])),max_s=max(p['timing'][k] for p in warm)) for k in ['initialization','field_sampling','optimization_validation','actual_posterior_update']}
            timings['total']=dict(median_s=float(np.median([p['wall_seconds'] for p in warm])),p95_s=float(np.percentile([p['wall_seconds'] for p in warm],95)),max_s=max(p['wall_seconds'] for p in warm),calls=len(warm))
            timings['startup_s']=[p['wall_seconds'] for p in r['plans'] if p['timing']['compilation_included']]
            before=np.asarray(r['belief_frames'][0]['std'])[free];after=np.asarray(r['belief_frames'][-1]['std'])[free]
            check=dict(id=ident,initial_performance_wind_exactly_zero=True,initial_safety_uncertainty_retained=True,
                no_future_observations=True,finite=True,actuator_limits=True,clarity_consistent=True,
                mean_full_sigma_before=float(before.mean()),mean_full_sigma_after=float(after.mean()),
                observations=len(r['observations']),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            checks.append(check)
            shutil.copyfile(path,dest/path.name);shutil.copyfile(a.directory/f'{ident}-protocol.json',dest/f'{ident}-protocol.json')
            runs.append(dict(id=ident,scenario=scenario,method=method,file=path.name,summary=r['summary'],timing=timings,belief=check))
    # First aware plans must be identical across hidden scenarios: no sensor
    # readings exist yet, and the same geometric and inference priors are used.
    for method in ['aware','baseline']:
        first=[np.load(a.directory/f's{s}-center-{method}-plan-000.npz') for s in [1,2]]
        np.testing.assert_array_equal(first[0]['theta'],first[1]['theta'])
        np.testing.assert_array_equal(first[0]['planning_fields'],0.)
        np.testing.assert_array_equal(first[1]['planning_fields'],0.)
    manifest=dict(duration_s=300,planning_horizon_s=15,replan_s=5,performance_fields=2,safety_fields=2,
        state='complete',runs=runs,initial_wind='zero in performance; uncertain in safety',
        method='Measurement-supported POAM sampling; expected monitoring cost; no information reward or hypothetical updates',
        numerical_notes='Existing C1 field cache at 0.2 m, 32-node POAM spectral residual, float64 inference and physics, float32 planning. Two antithetic performance fields per call; posterior frozen in each rollout.',
        safety_notes='Full posterior plus a pointwise 3-sigma force bound; finite sampled backup horizon. Infeasibility is logged. No deterministic global safety guarantee.',
        hardware=str(protocol['device']))
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (dest/'audit.json').write_text(json.dumps(dict(checks=checks,first_plan_identical_across_hidden_scenarios=True,all_declared_runs_included=True),indent=2))
    shutil.copyfile(a.directory/'environment.json',dest/'environment.json')
    root=Path(__file__).resolve().parents[2]
    source_archive(dest,root,protocol,a.site)
    print(json.dumps([dict(id=r['id'],summary=r['summary'],timing=r['timing']['total']) for r in runs],indent=2))


if __name__=='__main__':main()
