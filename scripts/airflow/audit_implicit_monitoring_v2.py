"""Audit completed recordings without re-running or changing the experiments."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np


def audit(directory):
    env=json.loads((directory/'environment.json').read_text())
    free=np.asarray(env['free_indices']);runs=[]
    for path in sorted(directory.glob('s*-*.json.gz')):
        r=json.load(gzip.open(path,'rt'));s=np.asarray(r['samples']);summary=r['summary']
        protocol=json.loads(path.with_name(r['id']+'-protocol.json').read_text())
        assert np.isfinite(s).all() and np.max(np.abs(s[:,5:7]))<=1.6+1e-6
        assert r['plans'][0]['planning_field_max_abs']==0 and r['plans'][0]['observations']==0
        assert min(r['belief_frames'][0]['std'])>1.79
        assert max(r['belief_frames'][0]['planning_std'])==0
        assert r['observations'][0][0]==0
        for p in r['plans']:
            expected=0 if p['t']==0 else sum(o[0]<=p['t']+1e-8 for o in r['observations'])
            assert p['observations']==expected
            assert p['evaluations']==43 and len(p['seed_costs'])==8
            assert len(p['predicted_states'])==2
            assert p['validation']['unique_evaluations']<=3
            if not p['timing']['compilation_included'] and p['wall_seconds']>1:
                assert not summary['completed'] and p is r['plans'][-1] and summary['stop_reason']=='Planning time exceeded one-second budget'
            if not p['validation']['no_feasible']:
                choice=p['validation']['selected']
                assert p['validation']['feasible'][choice]
                assert not any(p['predicted_contacts'])
        q=np.asarray([f['q'] for f in r['clarity_frames']])
        assert q.min()>=0 and q.max()<=1
        deficit=np.maximum(np.asarray(env['targets'])-q,0).mean(1)
        assert np.max(abs(deficit-s[::10,13]))<1e-6
        warm=[p for p in r['plans'] if not p['timing']['compilation_included']]
        timings={k:dict(median_s=float(np.median([p['timing'][k] for p in warm])),max_s=max(p['timing'][k] for p in warm)) for k in ['initialization','field_sampling','optimization_validation','mbd_search','execution_validation','actual_posterior_update']}
        timings['total']=dict(median_s=float(np.median([p['wall_seconds'] for p in warm])),max_s=max(p['wall_seconds'] for p in warm),p95_s=float(np.percentile([p['wall_seconds'] for p in warm],95)),calls=len(warm))
        timings['startup_s']=[p['wall_seconds'] for p in r['plans'] if p['timing']['compilation_included']]
        error=[]
        for p in r['plans']:
            target=p['t']+5
            if target>s[-1,0]:continue
            actual=s[round(target/.05),1:3]
            prediction=np.asarray(p['predicted_states'])[:,20,:2]
            error.append(float(np.linalg.norm(prediction-actual,axis=1).mean()))
        checks=dict(cold_performance_zero=True,full_safety_uncertainty_retained=True,actual_measurements_only=True,
                    finite_and_actuator_limits=True,clarity_consistent=True,
                    validation_failures=sum(p['validation']['no_feasible'] for p in r['plans']),
                    validation_failures_after_initial=sum(p['validation']['no_feasible'] for p in r['plans'][1:]),
                    forecast_contact_plans=sum(any(p['predicted_contacts']) for p in r['plans']),
                    five_second_position_error_median_m=float(np.median(error)),five_second_position_error_max_m=max(error),
                    mean_full_sigma_initial=float(np.asarray(r['belief_frames'][0]['std'])[free].mean()),
                    mean_full_sigma_final=float(np.asarray(r['belief_frames'][-1]['std'])[free].mean()))
        # Recorded runtime inputs must still be the exact source used by each run.
        root=Path(__file__).resolve().parents[2]
        for name,expected in protocol['source_sha256'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
        runs.append(dict(id=r['id'],scenario=r['scenario'],method=r['method'],case=r['initial_condition']['id'],file=path.name,summary=summary,timing=timings,checks=checks,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    first_identical=[]
    for case in set(r['case'] for r in runs):
        for method in ['aware','baseline']:
            files=[directory/f's{s}-{case}-{method}-plan-000.npz' for s in [1,2]]
            if not all(p.exists() for p in files):continue
            a,b=map(np.load,files)
            np.testing.assert_array_equal(a['theta'],b['theta'])
            np.testing.assert_array_equal(a['validation_fields'],b['validation_fields'])
            np.testing.assert_array_equal(a['validation_fields'],0.)
            first_identical.append(case+'/'+method)
    report=dict(runs=runs,first_plans_identical_across_hidden_scenarios=first_identical,
        caution='No contact is not a proof of robust safety. Full posterior pointwise confidence disks and sampled finite backup values are retained; no true-wind coverage guarantee is asserted.')
    (directory/'audit.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    report=audit(a.directory)
    print(json.dumps([dict(id=r['id'],summary=r['summary'],checks=r['checks'],timing=r['timing']['total']) for r in report['runs']],indent=2))
