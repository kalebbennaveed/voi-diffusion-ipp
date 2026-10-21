"""Evaluate the time dependence of a matched, completed 300 s comparison."""
import argparse,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--case',choices=['forward','reverse','crosswind'],default='forward');args=p.parse_args();root=Path(args.results)
records=[json.loads((root/f'two_{args.case}_{m}_8101.json').read_text()) for m in ('proposed','blind','blind_filter')]
reference=records[0]
assert all(r['completed'] and not r.get('plan_only') and r['mission_seconds']==300 and r['config']==reference['config'] for r in records)
summary=[]
for r in records:
    values=[]
    for plan in r['plans']:
        with np.load(root/plan['execution_audit']) as audit:
            values.extend(audit['cost_increments'].astype(float))
            q=audit['clarity'][-1].copy()
    cumulative=np.r_[0,np.cumsum(values,dtype=float)];times=np.arange(len(cumulative))*.01
    if r['failure']:
        target=np.asarray(r['environment']['targets'],np.float32)
        noise=np.asarray(r['environment']['process_noise'],np.float32)
        tail=[];accumulated=float(cumulative[-1])
        for _ in range(round((300-times[-1])/.1)):
            q=q/(1+noise*q*.1)
            accumulated+=.1*float(np.maximum(target-q,0).mean())
            tail.append(accumulated)
        times=np.r_[times,times[-1]+np.arange(1,len(tail)+1)*.1]
        cumulative=np.r_[cumulative,tail]
    assert abs(cumulative[-1]-r['cost'])<2e-4
    checkpoints=np.interp([0,60,120,180,240,300],times,cumulative)
    summary.append(dict(method=r['method'],success=r['success'],physical_seconds=r['completed_seconds'],
        cumulative_cost=checkpoints[1:].tolist(),minute_mean_deficit=(np.diff(checkpoints)/60).tolist(),
        first_minute_mean_deficit=float(checkpoints[1]/60),later_four_minutes_mean_deficit=float((checkpoints[-1]-checkpoints[1])/240),
        continuation_at=(r.get('continuation') or {}).get('continued_at_seconds')))
blind=next(r for r in summary if r['method']=='blind')
for r in summary:
    r['cumulative_reduction_vs_blind_percent']=(100*(1-np.array(r['cumulative_cost'])/blind['cumulative_cost'])).tolist()
    r['minute_reduction_vs_blind_percent']=(100*(1-np.array(r['minute_mean_deficit'])/blind['minute_mean_deficit'])).tolist()
    r['later_reduction_vs_blind_percent']=100*(1-r['later_four_minutes_mean_deficit']/blind['later_four_minutes_mean_deficit'])
report=dict(description='Matched 300 s physical simulations. Any continuation is identified per run. No flight extrapolation. Deficit means are derived from recorded objective increments; failed flights retain the unsensed remainder. One development seed, not independent confirmation.',seed=8101,window_seconds=60,runs=summary)
(root/f'mission_duration_{args.case}.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps(report,indent=2))
