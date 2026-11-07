"""Check recorded evidence, including failures, before declaring the demo final."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from package_final_comparison import ROOT,OUT,DEST,source,load,write,audit_reuse

def main():
    p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    m=load(OUT/'comparison-results.json');checks=[];unfinished=[]
    for s in m['scenarios']:
        env=load(DEST/s['environment']);targets=np.asarray(env['targets'])
        initials=[];configurations=[]
        for row in s['runs']:
            ident=f"s{s['id']}-{row['method']}"
            if row['status'] not in ('Completed','Contact','Learning limit'):
                unfinished.append(ident)
                # Active recordings are atomically replaced by the runner;
                # verify their final immutable snapshot after completion.
                if a.allow_partial or 'record' not in row:continue
            path=source(s['id'],row['method']);r=load(path);z=r['summary']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['source_sha256'],ident+' changed during audit'
            continuation=r.get('checkpoint_continuation')
            if continuation:
                assert continuation['posterior_matches_saved_arrays'] and continuation['observations_match_exactly'],ident+' invalid continuation'
                assert hashlib.sha256(Path(continuation['source']).read_bytes()).hexdigest()==continuation['sha256'],ident+' original checkpoint changed'
            prefix=r.get('prefix_verification')
            if prefix:
                assert prefix.get('full_precision_endpoint_verified') and prefix.get('full_precision_posterior_verified'),ident+' prefix verification incomplete'
                assert prefix['checked_observations']==round(prefix['replay_through_s']/.2)+1,ident+' incomplete prefix observations'
                assert hashlib.sha256(Path(prefix['source']).read_bytes()).hexdigest()==prefix['sha256'],ident+' original prefix changed'
            arr=np.asarray(r['samples']);ts=arr[:,0]
            assert np.all(np.isfinite(arr)) and np.all(np.diff(ts)>0),ident
            assert abs(ts[-1]-z['simulated_seconds'])<1e-7,ident
            assert np.max(np.abs(arr[:,5:7]))<=1.600001,ident+' actuator bound'
            assert int((arr[1:,11]==0).sum())==z['infeasible_execution_steps'],ident+' safety-step count'
            assert abs(arr[:,12].min()-z['min_clearance_m'])<6e-8,ident+' clearance summary'
            assert abs(arr[:,9].max()-z['max_execution_slack'])<6e-8,ident+' slack summary'
            assert abs(arr[-1,13]-row['final_clarity_deficit'])<1e-7,ident
            assert np.all(np.diff(arr[:,14])>=-1e-7),ident+' accumulated deficit decreased'
            assert abs(arr[-1,14]-z['integrated_deficit'])<1e-6,ident+' accumulated deficit differs'
            warm=np.asarray([p['wall_seconds'] for p in r['plans'] if p['phase']=='monitoring' and not p['compilation_included']])
            assert len(warm)==row['compute']['requests'],ident+' MBD timing count includes another phase'
            if len(warm):
                np.testing.assert_allclose([np.median(warm),np.percentile(warm,95),warm.max()],
                    [row['compute'][k] for k in ('median_s','p95_s','max_s')],rtol=0,atol=1e-10)
            if row['comparison_complete']:
                last=arr[ts<=s['comparison_window_s']+1e-8][-1]
                assert abs(row['common_window_mean_deficit']-last[14]/s['comparison_window_s'])<1e-10,ident+' shared-time mean'
            frame=r['clarity_frames'][0];q=np.asarray(frame['q'])
            assert q.shape==targets.shape and np.all((q>=0)&(q<=1)),ident
            np.testing.assert_allclose(np.maximum(targets-q,0).mean(),arr[0,13],atol=6e-7,rtol=0)
            for frame in r['clarity_frames']:
                field=np.asarray(frame['q'])
                assert field.shape==targets.shape and np.isfinite(field).all(),ident+' clarity map'
                assert np.all((field>=0)&(field<=1)),ident+' clarity range'
                k=int(np.argmin(np.abs(ts-frame['t'])))
                assert abs(ts[k]-frame['t'])<1e-7,ident+' clarity time alignment'
                np.testing.assert_allclose(np.maximum(targets-field,0).mean(),arr[k,13],atol=6e-7,rtol=0,
                    err_msg=ident+' plotted deficit does not match its clarity map')
            initials.append((ident,arr[0,1:5],q))
            contact=z.get('contact')
            if contact:assert 0<=ts[-1]-contact['time_s']<=.200001,ident+' continued after contact'
            proto=load(path.with_name(r['id']+'-protocol.json'))
            physical=dict(proto['physical_settings']);physical.pop('mission_seconds',None)
            # Historical recordings predate the spec-file hash field, but
            # contain the complete executed geometry and clarity grid. Compare
            # those numerical inputs directly for every method, rather than
            # assuming that a missing historical hash matches the new trials.
            executed_environment=load(path.parent/'environment.json')
            configurations.append((ident,dict(settings=proto['settings'],physical=physical,
                gp_profile=proto['gp_profile'],environment=executed_environment)))
            if row['reused']:
                audit_reuse(r)
            else:
                assert proto['uncertainty_only'] and not proto['validated_handoff'],ident
                assert proto['initial_measurements']==0,ident+' prior survey'
                assert proto['rmse_threshold_m_s'] is None,ident+' oracle gate'
                assert proto['gp_profile']['prior_std_m_s']==1.8,ident+' prior changed'
                if row['status']=='Learning limit':
                    assert z['gate_time_s'] is None and z['simulated_seconds']>=proto['learning_limit_s'],ident+' stopped before declared learning limit'
            gate=z.get('gate_time_s')
            if row['method'] in ('global','proposed') and gate is not None:
                metric='task_vector_std_m_s' if row['method']=='proposed' else 'mean_variance'
                def sigma(f):return f[metric] if metric.startswith('task') else np.sqrt(2*f[metric])
                eligible=[f for f in r['belief_frames'] if f.get(metric) is not None and sigma(f)<=1.1+1e-10]
                assert eligible and abs(min(f['t'] for f in eligible)-gate)<1e-8,ident+' first threshold crossing differs'
            checks.append(dict(id=ident,outcome=row['status'],t=float(ts[-1]),
                first_visit=z['first_visit_s'],gate_s=gate,
                final_deficit=row['final_clarity_deficit'],observations=len(r['observations']),
                executed_environment_sha256=hashlib.sha256(json.dumps(executed_environment,sort_keys=True).encode()).hexdigest(),
                source_sha256=row['source_sha256']))
        for ident,x,q in initials[1:]:
            np.testing.assert_allclose(x,initials[0][1],atol=1e-7,rtol=0,err_msg=ident)
            np.testing.assert_allclose(q,initials[0][2],atol=1e-6,rtol=0,err_msg=ident)
        for ident,config in configurations[1:]:
            assert config==configurations[0][1],ident+' physical, inference or MBD settings differ'
    archive_manifest=DEST.parent/'full-learning-20261104/manifest.json'
    archive=load(archive_manifest)
    archived_files={f for group in archive['groups'] for f in group['runs']}
    assert len(archive['groups'])>=52 and len(archived_files)>=57,'Prior demo groups were removed'
    assert all((archive_manifest.parent/f).is_file() for f in archived_files),'Prior recording is missing'
    preservation=dict(groups=len(archive['groups']),recordings=len(archived_files),all_recordings_present=True)
    report=dict(final=len(checks)==15 and not unfinished,checked=checks,unfinished=unfinished,
        previous_demo=preservation,
        checks='Finite monotonically timed recordings; identical initial states/clarity, physical parameters, GP configuration and MBD budget within each scenario; actuator bounds; contact termination; every recorded clarity map agrees with the plotted deficit; displayed metrics from raw data; uncertainty-only first-crossing handoff and unknown prior; explicit historical reuse audit.')
    write(OUT/'final-integrity-audit.json',report);write(DEST/'final-integrity-audit.json',report)
    print(json.dumps(dict(final=report['final'],checked=len(checks),unfinished=unfinished)))
    if unfinished and not a.allow_partial:raise RuntimeError('Comparison still contains unfinished declared trials')

if __name__=='__main__':main()
