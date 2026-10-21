"""Explain completed comparisons using measured execution and planning logs."""
import json
import argparse
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'output/european_quarter/rrt_mbd_runtime_300'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,default=RESULTS);args=parser.parse_args()
    results=args.results
    analysis=json.loads((results/'analysis.json').read_text())
    findings=[]
    for row in analysis['runs']:
        if row['method']!='proposed':continue
        methods={r['method'] for r in analysis['runs'] if (r['layout'],r['case'],r['seed'])==(row['layout'],row['case'],row['seed'])}
        if not {'blind','blind_filter'}<=methods:continue
        identity=dict(layout=row['layout'],case=row['case'],seed=row['seed'])
        items=[]
        record=json.loads((results/row['file']).read_text())
        if row['deficit_reduction_percent'] is not None:
            items.append(f"Proposed records {row['cost']:.4f} deficit·s, {row['deficit_reduction_percent']:+.2f}% reduction versus blind planning.")
        else:
            items.append("This terminated flight is excluded from monitoring-performance comparisons; the recording is retained for failure diagnosis.")
        if not row['success']:
            items.append(f"Proposed ends at {record['completed_seconds']:.2f} s: {row['failure']}. The remaining mission contributes deficit with sensing stopped.")
            if not row['observations']:
                items.append(f"The failure occurs before the first scheduled wind observation at {record['config']['commit']:g} s; the controller still uses the zero-mean initial belief.")
        else:
            items.append(f"Proposed completes {len(row['transfers'])} cross-district transfers and collects {row['observations']} wind observations.")
            items.append('Recorded time inside inspection regions: '+', '.join(f"{r['name']} {r['seconds']:.1f} s" for r in row['regions'])+'.')
            filtered_row=next(r for r in analysis['runs'] if r['layout']==row['layout'] and r['case']==row['case'] and r['seed']==row['seed'] and r['method']=='blind_filter')
            if filtered_row['success']:
                gain=100*(1-row['cost']/filtered_row['cost'])
                items.append(f"Against blind planning with the same execution filter, proposed has {abs(gain):.2f}% {'less' if gain>=0 else 'more'} total clarity deficit.")
            baseline_row=next(r for r in analysis['runs'] if r['layout']==row['layout'] and r['case']==row['case'] and r['seed']==row['seed'] and r['method']=='blind')
            if row['late_deficit_reduction_percent'] is not None:
                gain=row['late_deficit_reduction_percent']
                items.append(f"Over 60–300 s, proposed has {abs(gain):.2f}% {'less' if gain>=0 else 'more'} deficit than blind planning. The first minute remains included in the full-mission cost.")
            for direction in ('east','west'):
                descriptions=[]
                for title,item in [('proposed',row),('blind',baseline_row)]:
                    transfers=[t for t in item['transfers'] if t['direction']==direction]
                    counts={name:sum(t['passage']==name for t in transfers) for name in 'ABC'}
                    if transfers:descriptions.append(f"{title}: "+', '.join(f"{n} through {name}" for name,n in counts.items() if n)+f", median {np.median([t['seconds'] for t in transfers]):.1f} s")
                if descriptions:items.append(f"Recorded {direction}bound transfers — "+'; '.join(descriptions)+'. These describe the observed behavior, not an isolated causal effect of learning.')
        if row['case']=='commitment' and row['success']:
            east=row['regions'][1]
            if east['first_entry'] is not None:
                items.append(f"Proposed first enters the larger eastern region at {east['first_entry']:.1f} s. This run does not demonstrate the requested attempt-and-remain-west behavior.")
            baseline=next(r for r in analysis['runs'] if r['layout']==row['layout'] and r['case']==row['case'] and r['seed']==row['seed'] and r['method']=='blind')
            items.append(f"Eastern inspection time is {east['seconds']:.1f} s versus {baseline['regions'][1]['seconds']:.1f} s for blind planning. Mean western deficit is {row['regions'][0]['mean_deficit']:.3f} versus {baseline['regions'][0]['mean_deficit']:.3f}; eastern deficit is {east['mean_deficit']:.3f} versus {baseline['regions'][1]['mean_deficit']:.3f}.")
            if row['deficit_reduction_percent'] is not None and row['deficit_reduction_percent']<0 and row['regions'][0]['mean_deficit']>baseline['regions'][0]['mean_deficit'] and east['mean_deficit']<baseline['regions'][1]['mean_deficit']:
                items.append('The extra eastern monitoring reduces eastern deficit, but the deterioration in western monitoring outweighs that benefit in the mission objective.')
        evidence={}
        if record['failure'] and record['observations']:
            last=record['plans'][-1]
            with np.load(results/last['execution_audit']) as audit:
                elapsed=len(audit['states'])*.01
                actual=audit['states'][-1,:2].astype(float)
                points=np.asarray(last['points'])
                predicted=np.array([np.interp(elapsed,np.arange(len(points))*record['config']['dt'],points[:,k]) for k in range(2)])
                rectangles=np.asarray(record['environment']['buildings'])
                delta=np.abs(predicted-(rectangles[:,:2]+rectangles[:,2:4])/2)-(rectangles[:,2:4]-rectangles[:,:2])/2
                distances=np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(delta.max(axis=1),0)
                bounds=np.asarray(record['environment']['bounds'])
                margin=float(min(distances.min(),(predicted-bounds[0]).min(),(bounds[1]-predicted).min())-.275)
                error=float(np.linalg.norm(predicted-actual))
                evidence['last_execution']=dict(plan_time=last['t'],failure_time=record['completed_seconds'],
                    actual_position=actual.tolist(),posterior_mean_predicted_position=predicted.tolist(),
                    position_error=error,predicted_surface_clearance=margin,
                    actual_surface_clearance=float(audit['clearances'][-1]),maximum_slack=float(audit['slacks'].max()))
                items.append(f"At contact, the last posterior-mean rollout differs from the measured position by {error:.3f} m and predicts {margin:.3f} m surface clearance. The execution has {row['minimum_clearance']:.3f} m clearance. The execution controller and the remaining prediction error did not preserve clearance, despite {row['observations']} earlier observations.")
        if row['case']=='crosswind':
            blind=json.loads((results/f"{row['layout']}_crosswind_blind_{row['seed']}.json").read_text())
            filtered=json.loads((results/f"{row['layout']}_crosswind_blind_filter_{row['seed']}.json").read_text())
            same=np.array_equal(blind['plans'][0]['policy'],filtered['plans'][0]['policy'])
            evidence['blind_baselines_first_nominal_controls_identical']=bool(same)
            if same and not filtered['success'] and not filtered['observations'] and blind['success']:
                items.append(f"The two blind baselines start with identical nominal controls. Unfiltered execution completes; enabling the shared execution filter leads to contact at {filtered['completed_seconds']:.2f} s. This isolates an execution-filter failure under the initial uncertain belief, rather than a difference in their initial planned trajectory.")
            for d in (record,blind,filtered):
                with np.load(results/d['plans'][0]['execution_audit']) as audit:
                    i=min(449,len(audit['states'])-1)
                    t=(i+1)*.01
                    points=np.asarray(d['plans'][0]['points'])
                    prediction=np.array([np.interp(t,np.arange(len(points))*d['config']['dt'],points[:,k]) for k in range(2)])
                    evidence[d['method']]=dict(time=t,actual_position=audit['states'][i,:2].tolist(),
                        predicted_position=prediction.tolist(),position_error=float(np.linalg.norm(prediction-audit['states'][i,:2])),
                        applied_acceleration=audit['applied_controls'][i].tolist(),
                        first_execution_maximum_slack=float(audit['slacks'].max()))
        findings.append(dict(**identity,statements=items,evidence=evidence))
    analysis['findings']=findings
    (results/'analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False))
    print('Mechanisms documented for',len(findings),'completed comparisons.')


if __name__=='__main__':main()
