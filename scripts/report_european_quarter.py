"""Audit completed recordings and explain observed behavior using logged states."""
import json,argparse
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--results',default='output/european_quarter/validated_v4');args=parser.parse_args();root=Path(args.results);rows=[]
for path in sorted(root.glob('*.json')):
    d=json.loads(path.read_text())
    if not d.get('completed') or d.get('plan_only') or d.get('revision') not in ('quarter-optimistic-cosine-soft-v4','quarter-optimistic-cosine-soft-v5','quarter-full-control-mbd-v1'):continue
    frames=d['frames'];t=np.array([f['t'] for f in frames]);p=np.array([f['p'] for f in frames]);v=np.array([f['velocity'] for f in frames]);w=np.array([f['wind'] for f in frames]);q=np.array([f['q'] for f in frames]);targets=np.array(d['environment']['targets']);dt=np.diff(t,prepend=t[0]);deficit=np.maximum(targets-q,0).mean(axis=1)
    assert np.isfinite(p).all() and np.isfinite(q).all()
    assert np.all(np.diff(t)>0) and np.all((q>=0)&(q<=1.0001))
    if d['success']:
        assert abs(t[-1]-d['mission_seconds'])<.011
        assert abs(np.trapezoid(deficit,t)-d['cost'])<.03
    intervals=np.diff(t)
    assert np.allclose(intervals[:-1],.1,atol=1e-4)
    assert 0<intervals[-1]<=.1001
    assert 0<d['config']['dt']<=.5 and d['model']['execution_step']==.01
    assert abs(d['config']['dt']/.01-round(d['config']['dt']/.01))<1e-6
    assert d['model']['schedule'].startswith('cosine')
    speed=np.linalg.norm(v[:,:2],axis=1);headwind=(np.sum(w*v[:,:2],axis=1)<-.5)&(speed>.1)
    scale=d.get('config',{}).get('environment_scale',1.)
    passages=np.argmin(abs(p[:,1,None]-np.array(d['environment'].get('passages',np.array([8,-23,27])*scale))[None]),axis=1);central=abs(p[:,0])<19*scale
    transfers=[];last_side=None;entry=None
    for i in range(len(t)):
        side=-1 if p[i,0]<-22*scale else 1 if p[i,0]>22*scale else 0
        if side:
            if last_side is not None and side!=last_side and entry is not None:
                sample=np.arange(entry,i+1);mid=sample[central[sample]]
                if len(mid):
                    code=int(np.bincount(passages[mid],minlength=3).argmax());transfers.append(dict(start=float(t[entry]),end=float(t[i]),passage='ABC'[code],direction='east' if side==1 else 'west',seconds=float(t[i]-t[entry])))
            last_side=side;entry=i
    regions=[]
    for k,r in enumerate(d['environment']['regions']):
        inside=np.all(p[:,:2]>=r[:2],axis=1)&np.all(p[:,:2]<=r[2:],axis=1)
        regions.append(dict(name=['west','east','northwest','southeast'][k],seconds=float(dt[inside].sum()),first_entry=float(t[inside][0]) if inside.any() else None,final_mean_clarity=float(q[-1,k*42:(k+1)*42].mean()),mean_deficit=float(np.trapezoid(np.maximum(targets[k*42:(k+1)*42]-q[:,k*42:(k+1)*42],0).mean(axis=1),t)/max(t[-1],1e-6))))
    belief=d['beliefs'][-1];grid=np.array(d['environment']['grid']);valid=np.ones(len(grid),bool)
    for b in d['environment']['buildings']:valid&=~(np.all(grid>=b[:2],axis=1)&np.all(grid<=b[2:4],axis=1))
    times=np.array([plan.get('total_planning_seconds',plan['seconds']) for i,plan in enumerate(d['plans']) if i>=2 and not plan.get('compilation_included') and not d['plans'][i-1].get('compilation_included')])
    # Early versus later task performance is measured from the same recording,
    # not an extra reward for visiting an uncertain region.
    cutoff=min(60.,d['mission_seconds']);mask=t<=cutoff
    early=float(np.trapezoid(deficit[mask],t[mask]))
    if all(plan.get('execution_audit') and (root/plan['execution_audit']).exists() for plan in d['plans']):
        # Use the actual 100 Hz objective increments, before rounding for display.
        early=0.;elapsed=0.
        for plan in d['plans']:
            with np.load(root/plan['execution_audit']) as execution:
                count=min(len(execution['cost_increments']),round(max(0.,cutoff-elapsed)/.01))
                early+=float(execution['cost_increments'][:count].sum(dtype=np.float64))
                elapsed+=len(execution['cost_increments'])*.01
            if elapsed>=cutoff-1e-6:break
    if t[-1]<cutoff:
        tail=np.arange(t[-1]+.1,cutoff+.001,.1)
        tail_q=q[-1]/(1+np.array(d['environment']['process_noise'])*q[-1]*(tail[:,None]-t[-1]))
        early+=float(.1*np.maximum(targets-tail_q,0).mean(axis=1).sum())
    late=(d['cost']-early)/max(d['mission_seconds']-cutoff,1e-9)
    measured_frames={round(f['t'],2):f for f in frames};prediction_errors=[];later_prediction_errors=[]
    for plan in d['plans']:
        endpoint=round(plan['t']+d['config']['commit'],2)
        if endpoint not in measured_frames:continue
        predicted=plan['points'][round(d['config']['commit']/d['config']['dt'])]
        error=float(np.linalg.norm(np.asarray(predicted)-np.asarray(measured_frames[endpoint]['p'][:2])))
        prediction_errors.append(error)
        if plan['t']>=30:later_prediction_errors.append(error)
    rows.append(dict(revision=d['revision'],file=path.name,layout=d['layout'],case=d['case'],method=d['method'],seed=d['seed'],cost=d['cost'],success=d['success'],failure=d['failure'],mission_seconds=d['mission_seconds'],first_minute_cost=early,later_mean_deficit=late,headwind_seconds=float(dt[headwind].sum()),median_headwind_speed=float(np.median(speed[headwind])) if headwind.any() else None,passage_seconds={name:float(dt[central&(passages==k)].sum()) for k,name in enumerate('ABC')},transfers=transfers,regions=regions,initial_mean_wind_std=1.8,final_mean_wind_std=float(np.mean(np.array(belief['std'])[valid])),final_wind_rmse=belief['rmse'],observations=len(d['observations']),minimum_clearance=d['min_clearance'],tracking_rms=d['rms_tracking_error'],median_planning_seconds=float(np.median(times)) if len(times) else None))
    rows[-1].update(mean_position_prediction_error=float(np.mean(prediction_errors)) if prediction_errors else None,
                    position_prediction_error_after_30s=float(np.mean(later_prediction_errors)) if later_prediction_errors else None,
                    prediction_error_interval=d['config']['commit'])
for r in rows:
    blind=next((b for b in rows if b['layout']==r['layout'] and b['case']==r['case'] and b['seed']==r['seed'] and b['method']=='blind' and b['mission_seconds']==r['mission_seconds']),None)
    r['early_deficit_reduction_percent']=100*(blind['first_minute_cost']-r['first_minute_cost'])/blind['first_minute_cost'] if blind and blind['first_minute_cost']>0 else None
    r['late_deficit_reduction_percent']=100*(blind['later_mean_deficit']-r['later_mean_deficit'])/blind['later_mean_deficit'] if blind and blind['later_mean_deficit']>0 else None
    r['deficit_reduction_percent']=100*(blind['cost']-r['cost'])/blind['cost'] if blind else None
    if blind:
        # A percentage is meaningful only under the same physical experiment.
        # Optimizer settings may differ in explicitly identified ablations.
        current=json.loads((root/r['file']).read_text())
        baseline=json.loads((root/blind['file']).read_text())
        for field in ('bounds','buildings','regions','cells','targets','process_noise','wind'):
            np.testing.assert_array_equal(current['environment'][field],baseline['environment'][field])
        assert current['vehicle']==baseline['vehicle']
        for field in ('p','velocity','q'):
            np.testing.assert_array_equal(current['frames'][0][field],baseline['frames'][0][field])
out=dict(description='Measured from completed flight recordings; each row identifies its source revision. Individual seeds, not statistical claims. Median timing excludes the first two calls of each process, including a resumed process, because type specialization can compile twice.',runs=rows)
(root/'analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False))
for r in rows:print(r['layout'],r['case'],r['method'],round(r['cost'],3),r['deficit_reduction_percent'],r['transfers'])
