"""Publish all declared ablations, preserving raw results and honest censoring."""
import gzip,hashlib,json,shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/final_comparison_20261106'
SITE=ROOT.parent/'sites/belief-adaptive-monitoring/dist/revised'
DEST=SITE/'data/final-comparison-20261106'
METHODS=[
 ('unfiltered','Baseline 1 · MBD without RPCBF','#b23a48','Still-air MBD; no execution safety filter or learned-wind compensation. Sensor readings are logged; the unused wind belief is not updated.'),
 ('filtered','Baseline 2 · MBD + execution RPCBF','#b5751b','Still-air MBD; learned-belief tracking and full RPCBF only during execution.'),
 ('aware','Baseline 3 · Disturbance-aware MBD','#7355a2','Full learned-posterior MBD and RPCBF, beginning immediately without a survey.'),
 ('global','Baseline 4 · Global learning → DA-MBD','#367c9b','Uniform free-space information acquisition; posterior vector RMS ≤ 1.10 m/s, then disturbance-aware MBD.'),
 ('proposed','Proposed · Candidate-path learning → DA-MBD','#087f74','Clarity-valued geometric-path information acquisition; weighted posterior vector RMS ≤ 1.10 m/s, then the same MBD.')]

def load(path):
    with gzip.open(path,'rt') if path.suffix=='.gz' else path.open() as f:return json.load(f)

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(data,allow_nan=False,separators=(',',':'))
    temporary=path.with_name(path.name+'.tmp')
    if path.suffix=='.gz':
        with gzip.open(temporary,'wt',compresslevel=5) as f:f.write(text)
    else:temporary.write_text(text)
    temporary.replace(path)

def source(s,method):
    continuation=OUT/f'{method}-s{s}-completed'
    mode='pathmask' if method=='proposed' else method
    if (continuation/'summary.json').exists():
        rows=load(continuation/'summary.json')
        if any(r['id']==f's{s}-center-{mode}' and (r['completed'] or r.get('contact') or (r.get('stop_reason') or '').startswith('Learning accuracy threshold not reached')) for r in rows):
            return continuation/f's{s}-center-{mode}.json.gz'
    if method=='proposed' and s<3:
        folder='safety-aware-proposed' if s==1 else 'safety-aware-proposed-s2-recovered'
        return ROOT/'output/path_mask_20261105'/folder/f's{s}-center-pathmask.json.gz'
    folders={'unfiltered':'unfiltered' if s<3 else 'unfiltered-s3','aware':'aware' if s<3 else 'aware-s3',
             'filtered':f'filtered-s{s}','global':f'global-s{s}','proposed':'proposed-s3'}
    mode='pathmask' if method=='proposed' else method
    return OUT/folders[method]/f's{s}-center-{mode}.json.gz'

def audit_reuse(record):
    frames=record['belief_frames'];gate=record['summary']['gate_time_s']
    candidates=[f['t'] for f in frames if f.get('task_vector_std_m_s') is not None and f['task_vector_std_m_s']<=1.1]
    first=min(candidates) if candidates else None
    checks=record.get('handoff_validation',[])
    result=dict(original_gate_s=gate,first_uncertainty_qualified_s=first,
        unchanged_recorded_transition=first==gate,
        development_forecast_check_passed_at_same_time=any(c['t']==gate and c['passed'] for c in checks),
        limitation='Retrospective criterion audit of an existing recording, not a newly executed uncertainty-only run. The original development forecast check passed when uncertainty first crossed the threshold; it did not delay the recorded transition.')
    if not result['unchanged_recorded_transition'] or not result['development_forecast_check_passed_at_same_time']:
        raise ValueError('Existing result does not match new stopping transition')
    return result

def main():
    DEST.mkdir(parents=True,exist_ok=True);scenarios=[];allrows=[]
    incident=OUT/'runtime-pressure-incident.json'
    if incident.exists():shutil.copy2(incident,DEST/incident.name)
    previous=load(DEST/'manifest.json') if (DEST/'manifest.json').exists() else {}
    cached={(s['id'],r['method']):r for s in previous.get('scenarios',[]) for r in s['runs']}
    methods=[dict(id=m,label=l,color=c,definition=d) for m,l,c,d in METHODS]
    for s in (1,2,3):
        flow=(SITE/f'data/scenario{s}') if s<3 else OUT/'scenario3-flow-v2/data/scenario3'
        if not (flow/'metadata.json').exists():
            scenarios.append(dict(id=s,title=f'Scenario {s}',status='Computing stationary airflow',runs=[]));continue
        meta=load(flow/'metadata.json');spec=meta.get('environment_spec')
        if spec is None:spec=load(ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
        dx=spec['dx'];bounds=np.asarray(spec['bounds']);nx,ny=np.rint((bounds[1]-bounds[0])/dx).astype(int)
        xx,yy=np.meshgrid(bounds[0,0]+(np.arange(nx)+.5)*dx,bounds[0,1]+(np.arange(ny)+.5)*dx)
        cells=np.c_[xx.ravel(),yy.ravel()];free=np.ones(len(cells),bool)
        for x,y,r in spec['obstacles']:free&=np.sum((cells-[x,y])**2,axis=1)>r*r
        cells=cells[free];important=np.zeros(len(cells),bool)
        for region in spec['regions']:important|=np.all(cells>=region['bounds'][0],axis=1)&np.all(cells<=region['bounds'][1],axis=1)
        targets=np.where(important,spec.get('region_target',.7),spec.get('background_target',.35))
        field=np.fromfile(flow/meta['data_file'],dtype='<f4').reshape(meta['grid'][1],meta['grid'][0],3)
        env=dict(bounds=spec['bounds'],obstacles=spec['obstacles'],regions=spec['regions'],cells=cells.tolist(),
            grid=[int(nx),int(ny)],free_indices=np.flatnonzero(free).tolist(),targets=targets.tolist(),dx=dx,
            wind=field[...,:2].tolist(),wind_dx=meta['dx_m'],wind_region=spec.get('wind_region'),metadata=meta)
        write(DEST/f'environment-s{s}.json',env)
        # The preserved S1/S2 proposed recordings determine their shared
        # comparison windows. In S3, compare at the global learner's completed
        # survey + 300 s MBD horizon; retain longer recordings in full.
        comparison_window={1:532,2:560,3:600}[s]
        if s==3:
            baseline_summary=OUT/'global-s3/summary.json'
            if baseline_summary.exists():
                completed=[z for z in load(baseline_summary) if z['completed']]
                if completed:comparison_window=min(comparison_window,completed[0]['simulated_seconds'])
        scenario=dict(id=s,title=f'Scenario {s}',status='Recorded comparisons',environment=f'environment-s{s}.json',
            comparison_window_s=comparison_window,runs=[])
        for method,label,color,definition in METHODS:
            path=source(s,method)
            if not path.exists():
                scenario['runs'].append(dict(method=method,label=label,status='Pending',color=color));continue
            fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()
            old=cached.get((s,method),{})
            if (old.get('source_sha256')==fingerprint and old.get('status') in ('Completed','Contact','Checkpoint','Learning limit')
                and old.get('source')==str(path.relative_to(ROOT))
                and old.get('compute') is not None and 'learning_footprint' in old and old.get('comparison_window_s')==scenario['comparison_window_s']
                and (DEST/old['record']).exists()):
                scenario['runs'].append(old);allrows.append(dict(scenario=s,**old));continue
            r=load(path);summary=r['summary'];arr=np.asarray(r['samples']);contact=summary.get('contact')
            # An old timestamp or checkpoint is never declared completed.
            summaries=load(path.parent/'summary.json') if (path.parent/'summary.json').exists() else []
            finished=any(z['id']==r['id'] for z in summaries)
            status='Contact' if contact else 'Completed' if summary['completed'] and finished else 'Stopped' if finished else 'Running'
            if finished and summary.get('stop_reason') in ('Explicit experiment wall-time limit reached',
                    'Stopped at a requested experiment checkpoint; recordings and posterior retained'):status='Checkpoint'
            if finished and (summary.get('stop_reason') or '').startswith('Learning accuracy threshold not reached'):status='Learning limit'
            reused=method=='proposed' and s<3
            audit=audit_reuse(r) if reused else None
            budget=scenario['comparison_window_s'];at=arr[arr[:,0]<=budget+1e-8][-1]
            ends_at_contact=bool(contact and contact['time_s']<=budget)
            complete_window=not ends_at_contact and summary['simulated_seconds']>=budget
            learning=summary.get('gate_time_s');duration=summary['simulated_seconds']
            footprint=None
            if s==3 and learning is not None and learning>0:
                segment=arr[arr[:,0]<learning]
                region=np.asarray(spec['wind_region'])
                inside=np.all(segment[:,1:3]>=region[0],axis=1)&np.all(segment[:,1:3]<=region[1],axis=1)
                footprint=dict(learning_time_s=learning,
                    lower_right_time_s=float(np.sum(inside)*.05),
                    minimum_learning_y_m=float(segment[:,2].min()),
                    definition='Actual recorded positions during dedicated learning, at 0.05 s resolution. Lower-right box is used only for this offline diagnostic.')
            warm=[p for p in r['plans'] if p['phase']=='monitoring' and not p['compilation_included']]
            request_times=np.asarray([p['wall_seconds'] for p in warm])
            compute=dict(horizon_s=15,replan_s=5,requests=len(warm),
                median_s=float(np.median(request_times)) if len(warm) else None,
                p95_s=float(np.percentile(request_times,95)) if len(warm) else None,
                max_s=float(request_times.max()) if len(warm) else None,
                definition='Recorded complete warm MBD requests: fields, geometry, optimization/validation, immediately preceding posterior update and safety-field refresh. Startup compilation excluded; concurrent CPU load included.')
            curve=arr[::20][:,[0,13]].tolist()
            if curve[-1][0]!=arr[-1,0]:curve.append(arr[-1,[0,13]].tolist())
            row=dict(method=method,label=label,color=color,status=status,total_time_s=duration,
                learning_time_s=learning if learning is not None else duration,learning_completed=learning is not None,
                monitoring_time_s=summary.get('monitoring_duration_s',max(0,duration-(learning or 0))),
                final_clarity_deficit=float(arr[-1,13]),comparison_deficit=float(at[13]) if complete_window else None,
                comparison_window_s=budget,comparison_complete=complete_window,contact=contact,
                common_window_mean_deficit=float(at[14]/budget) if complete_window else None,
                learning_footprint=footprint,
                cumulative_deficit=float(arr[-1,14]),curve=curve,record=f's{s}-{method}.json.gz',
                source=str(path.relative_to(ROOT)),source_sha256=fingerprint,
                original_summary=summary,reused=reused,reuse_audit=audit,
                compute=compute,
                uncertainty_stopping='path-weighted posterior vector RMS ≤ 1.10 m/s' if method=='proposed' else 'whole-map posterior vector RMS ≤ 1.10 m/s' if method=='global' else 'No dedicated learning stage')
            # Display archive: physical data are untouched; optimization logs remain in raw source.
            view=dict(samples=r['samples'][::2],clarity_frames=r['clarity_frames'],belief_frames=r['belief_frames'],
                task_frames=r.get('task_frames',[]),summary=summary,
                references=[dict(t=p['t'],phase=p['phase'],paths=p.get('initialization',{}).get('coarse_paths',[]) or ([p['path']] if p.get('path') else [])) for p in r['plans'] if p['phase']=='monitoring' or abs(p['t']-round(p['t']))<1e-7])
            if method=='global' and not view['task_frames']:
                weights=np.zeros(len(free));weights[free]=1/free.sum()
                view['task_frames']=[dict(t=0.,weights=weights.tolist(),geometric_paths=[],source='Exact uniform free-space acquisition domain')]
            if view['samples'][-1]!=r['samples'][-1]:view['samples'].append(r['samples'][-1])
            write(DEST/row['record'],view);scenario['runs'].append(row);allrows.append(dict(scenario=s,**row))
        scenarios.append(scenario)
    manifest=dict(version='20261106-uncertainty-only',methods=methods,scenarios=scenarios,
        stopping='Posterior vector RMS uncertainty ≤ 1.10 m/s over each acquisition domain. No truth or forecast-validation gate in new trials.',
        time_definition='Total time means recorded simulated mission time, not compute time. Full-trajectory curves stop at contact; no fabricated continuation. Common-window deficit is reported only for surviving recordings that reach that window. S1/S2 windows preserve the existing proposed horizons; the S3 window equals the global learner\'s completed learning + 300 s MBD. Longer recordings remain visible, but their own-end deficits are not equal-time comparisons.',
        scenario3_prior='Unknown wind everywhere, zero prior mean and the same full spatial uncertainty. Calm truth is not prior knowledge; some local learning can therefore be necessary.',
        scenario3_task='Upper band has clarity target 0.70 and decay 0.16. Background target is zero, while its clarity dynamics continue. No wind support is given to the method.',
        limitations=['Existing Scenario 1/2 proposed runs are reused with an explicit stopping-criterion audit.',
            'Single declared initial condition and seed per scenario; all outcomes shown.',
            'Safety-aware finite-sample RPCBF can be infeasible or collide.',
            'A geometric-path mask is a relevance proxy, not exact value of information.',
            'Stationary CFD mean fields are numerical scenarios, not measured or grid-converged truth.'])
    write(DEST/'manifest.json',manifest);write(OUT/'comparison-results.json',manifest)
    print(json.dumps(dict(recorded=len(allrows),states=[(r['scenario'],r['method'],r['status'],r['total_time_s']) for r in allrows])))

if __name__=='__main__':main()
