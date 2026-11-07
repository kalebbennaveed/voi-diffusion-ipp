"""Audit saved executed learning experiments, including unsuccessful runs."""
import argparse,gzip,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runs=[];records={}
    for f in sorted(a.directory.glob('s*-*.json.gz')):
        r=json.load(gzip.open(f));records[r['scenario'],r['method']]=r
        x=np.asarray(r['samples'],dtype=float);o=np.asarray(r['observations']);q=np.asarray([f['q'] for f in r['clarity_frames']]);s=r['summary']
        assert np.isfinite(np.delete(x,9,axis=1)).all(), 'Nonfinite recorded state/control'
        assert np.isfinite(x[:,:9]).all() and np.isfinite(q).all()
        assert np.max(np.abs(x[:,5:7]))<=1.6+1e-6
        assert np.all((q>=0)&(q<=1))
        assert np.allclose(np.diff(x[:,0]),.05,atol=1e-7)
        assert len(o)==s['actual_measurements']
        indices=np.rint(o[:,0]/.05).astype(int)
        np.testing.assert_allclose(o[:,1:3],x[indices,1:3],atol=1e-6,rtol=0)
        if s['contact'] is not None:assert np.max(o[:,0])<s['contact']['time_s']
        times={}
        for phase in ['learning','monitoring']:
            plans=[z for z in r['plans'] if z['phase']==phase and not z['compilation_included']]
            # The first optional kernel EM compilation occurs at t=2 s.
            steady=[z for z in plans if z['t']!=2.]
            if steady:
                vals=np.array([z['wall_seconds'] for z in steady]);times[phase]=dict(median_s=float(np.median(vals)),max_s=float(vals.max()),calls=len(vals))
        row=dict(id=r['id'],scenario=r['scenario'],method=r['method'],summary=s,
                 timing=times,finite_states_controls_clarity=True,actuator_limits_respected=True,
                 measurements_at_executed_positions=True,
                 reached_A_and_completed_without_contact=bool(s['completed'] and s['contact'] is None and s['first_visit_s']['A'] is not None),
                 full_filter_feasibility=bool(s['infeasible_execution_steps']==0),
                 infinite_slack_steps=int(np.sum(~np.isfinite(x[:,9]))),
                 actual_wind_rmse_improved=bool(s['final_rmse_m_s']<s['initial_rmse_m_s']))
        runs.append(row)
    handoffs=[]
    for scenario in [1,2]:
        survey=records[scenario,'survey'];baseline=records[scenario,'baseline1'];h=baseline['handoff'];assert h
        prefix=lambda r:[z for z in r['samples'] if z[0]<=h['t']]
        assert prefix(survey)==prefix(baseline)
        np.testing.assert_allclose(h['state'],prefix(baseline)[-1][1:5],atol=1e-7,rtol=0)
        frame=next(z for z in baseline['clarity_frames'] if z['t']==h['t'])
        np.testing.assert_allclose(h['clarity'],frame['q'],atol=5.1e-7,rtol=0)
        assert h['observations']==len([o for o in baseline['observations'] if o[0]<=h['t']])
        saved=np.load(a.directory/f'{baseline["id"]}-handoff.npz')
        np.testing.assert_array_equal(saved['mean'],np.asarray(h['posterior_mean']))
        np.testing.assert_array_equal(saved['covariance'],np.asarray(h['posterior_covariance']))
        handoffs.append(dict(scenario=scenario,t=h['t'],identical_learning_execution=True,
                             state_clarity_and_posterior_preserved=True,measurements=h['observations']))
    report=dict(runs=runs,handoffs=handoffs,note='Failed and stopped runs are retained. Variance reduction is a model statistic, not an accuracy certificate. Partial accumulated costs are not comparable with full 300 s costs.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(runs=len(runs),complete_no_contact_A=sum(r['reached_A_and_completed_without_contact'] for r in runs),
                         contacts=sum(r['summary']['contact'] is not None for r in runs),handoffs=handoffs)))


if __name__=='__main__':main()
