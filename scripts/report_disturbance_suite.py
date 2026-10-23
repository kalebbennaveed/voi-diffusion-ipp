"""Report paired actual-flight evidence. Numerical results use dense audit logs."""
import argparse,hashlib,json,zlib
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT=Path(__file__).resolve().parents[1]
from voi_diffusion_ipp.quarter.benchmark_environments import SCENARIOS
PLAYBACK_VERSION='recorded-spatial-maps-v3'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def examine(path):
    path=path.resolve()
    d=json.loads(path.read_text());dense=0.;logged_dense=0.;count=0;minimum=np.inf;max_quality=np.array(d['frames'][0]['q'])
    last_quality=max_quality.copy();target=np.array(d['environment']['targets']);step_error=0.
    snapshot=path.parent.parent/'source_snapshot'
    for name,digest in d['source_hashes'].items():assert sha(snapshot/name)==digest,(path,name)
    if d.get('continuation'):
        prefix_path=Path(d['continuation']['source']);assert sha(prefix_path)==d['continuation']['source_sha256']
        prefix=json.loads(prefix_path.read_text())
        assert d['frames'][:len(prefix['frames'])]==prefix['frames']
        assert d['observations'][:len(prefix['observations'])]==prefix['observations']
        assert d['plans'][:len(prefix['plans'])]==prefix['plans']
    last_state=None
    for p in d['plans']:
        with np.load(path.parent/p['execution_audit']) as a:
            if last_state is not None:np.testing.assert_array_equal(a['initial_state'],last_state)
            if len(a['states']):last_state=a['states'][-1].astype(float)
            quality=np.asarray(a['clarity'],float);logged=np.asarray(a['cost_increments'],float)
            after=np.maximum(target-quality,0).mean(axis=1)
            before=np.r_[np.maximum(target-last_quality,0).mean(),after[:-1]]
            recomputed=.005*(before+after)
            np.testing.assert_allclose(logged,recomputed,rtol=0,atol=2e-8)
            step_error=max(step_error,float(np.max(np.abs(logged-recomputed))))
            dense+=float(recomputed.sum());logged_dense+=float(logged.sum());count+=len(a['states'])
            if len(quality):last_quality=quality[-1]
            if len(a['states']):minimum=min(minimum,float(np.min(a['clearances'])))
            if len(a['clarity']):max_quality=np.maximum(max_quality,np.max(a['clarity'],axis=0))
    if d['success']:assert abs(dense-d['cost'])<1e-4,(path,dense,d['cost'])
    assert abs(count*.01-d['completed_seconds'])<.011
    f=d['frames'];velocity=np.array([x['velocity'][:2] for x in f]);w=np.array([x['wind'] for x in f]);speed=np.linalg.norm(velocity,axis=1)
    e=d['environment'];grid=np.array(e['grid']).reshape(*e['grid_shape'],2)
    grid_wind=np.array(e['wind']).reshape(*e['grid_shape'],2)
    interpolate=RegularGridInterpolator((grid[:,0,1],grid[0,:,0]),grid_wind,bounds_error=False,fill_value=0.)
    reference=np.array([row['ref'][:2] for row in f]);reference_wind=interpolate(reference[:,::-1])
    tracking=np.linalg.norm(np.array([x['p'] for x in f])-np.array([x['ref'] for x in f]),axis=1)
    held=(speed<.2)&(tracking>1.)
    cells=np.array(d['environment']['cells']);targets=np.array(d['environment']['targets'])
    regional=[]
    for i,r in enumerate(np.array(d['environment']['regions'])):
        mask=np.all((cells>=r[:2])&(cells<=r[2:]),axis=1)
        if mask.any():regional.append({'site':i+1,'fraction_ever_reaching_target':float(np.mean(max_quality[mask]>=targets[mask]))})
    along=np.sum(w*velocity,axis=1)/np.maximum(speed,.1)
    head=(along<-1.5)&(speed>.2)
    warm_plans=[x for x in d['plans'] if not x.get('compilation_included',False)]
    warm=[x['complete_planning_seconds'] for x in warm_plans]
    early=min(f,key=lambda x:abs(x['t']-60))['cost']
    out={'method':d['method'],'cost':d['cost'],'cost_first60':early,'cost_after60':d['cost']-early,
        'completed_seconds':d['completed_seconds'],'success':d['success'],'failure':d['failure'],
        'minimum_surface_gap':minimum if np.isfinite(minimum) else None,'dense_cost_before_failure':dense,'dense_steps':count,
        'saved_cost_increments_sum':logged_dense,'maximum_integral_step_discrepancy':step_error,
        'source_snapshot_hashes_verified':True,'physical_state_continuity_verified':True,
        'continued_prefix_verified':True if d.get('continuation') else None,
        'dense_cost_matches_report':bool(abs(dense-d['cost'])<1e-4) if d['success'] else None,
        'unobserved_remainder_penalty':d['cost']-dense if not d['success'] else 0.,
        'wind_exposure_fraction':float(np.mean(np.linalg.norm(w,axis=1)>1.5)),
        'nominal_reference_wind_exposure_fraction_approx':float(np.mean(np.linalg.norm(reference_wind,axis=1)>1.5)),
        'headwind_fraction':float(np.mean(head)),
        'mean_speed_when_opposing_wind':float(np.mean(speed[head])) if head.any() else None,
        'rms_reference_tracking_error':d['rms_tracking_error'],
        'nearly_stationary_with_reference_error_over_1m_fraction':float(np.mean(held)),
        'cells_ever_reaching_target_fraction':float(np.mean(max_quality>=targets)),
        'regional_target_attainment':regional,
        'cells_ever_above_0_7_fraction':float(np.mean(max_quality>.7)),
        'warm_planning_median':float(np.median(warm)) if warm else None,
        'warm_planning_p95':float(np.quantile(warm,.95)) if warm else None,
        'warm_planning_max':float(np.max(warm)) if warm else None,
        'first_call_seconds':d['plans'][0]['complete_planning_seconds'] if d['plans'] else None,
        'process_start_calls':[{'mission_time':p['t'],'recorded_seconds':p['complete_planning_seconds']} for p in d['plans'] if p.get('compilation_included')],
        'median_fields_seconds':float(np.median([p['field_seconds'] for p in warm_plans])) if warm else None,
        'median_geometry_seconds':float(np.median([p['rrt_seconds'] for p in warm_plans])) if warm else None,
        'median_search_selection_seconds':float(np.median([p['seconds'] for p in warm_plans])) if warm else None,
        'source':str(path.relative_to(ROOT)),'sha256':sha(path),'source_hashes':d['source_hashes']}
    assert all(p.get('optimism_fraction')==1 and p.get('archive_mixture')==0 for p in d['plans'])
    playback={'frames':[f[i] for i in range(0,len(f),5)]+([f[-1]] if (len(f)-1)%5 else []),
        'trail':[[x['t'],*x['p'][:2]] for x in f],'beliefs':d['beliefs'],'environment':d['environment'],
        'method':d['method']}
    return d,out,playback


def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',type=Path,nargs='+');p.add_argument('--selection',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if bool(a.roots)==bool(a.selection):p.error('Supply either explicit cohort roots or a case-selection manifest')
    selection=json.loads(a.selection.read_text()) if a.selection else None
    rows=[];pending=[]
    (a.out/'data').mkdir(exist_ok=True)
    for s in SCENARIOS:
        paths=[]
        roots=[ROOT/selection['cases'][s.name]['root']] if selection else a.roots
        for method in ['aware','blind_filter']:
            found=[f for root in roots for f in (root/s.name).glob(f'two_{s.name}_{method}_*.json') if json.loads(f.read_text()).get('completed')]
            if len(found)>1:raise ValueError(f'Explicitly select one paired cohort: {found}')
            paths.append(found[0] if found else None)
        if not all(paths):pending.append(s.name);continue
        da,ma,pa=examine(paths[0]);db,mb,pb=examine(paths[1])
        manifest=json.loads((paths[0].parent.parent/'experiment.json').read_text())
        task=next(item for item in manifest['cases'] if item['name']==s.name)
        assert da['config']==db['config'] and da['source_hashes']==db['source_hashes']
        assert da['mission_seconds']==db['mission_seconds'] and da['seed']==db['seed']
        assert da['environment']==db['environment'] and da['frames'][0]==db['frames'][0]
        safe=bool(da['success'] and db['success'] and
            min(da['completed_seconds'],db['completed_seconds'])>=da['mission_seconds']-.001)
        if safe:assert min(ma['minimum_surface_gap'],mb['minimum_surface_gap'])>=-2e-5
        value=100*(mb['cost']-ma['cost'])/mb['cost']
        row={'case':s.name,'title':task['title'],'topology':task['topology'],'task':task['task'],
            'mechanism':task['disturbance_mechanism'],'scenario_revision':task.get('revision','initial-development-suite'),
            'real_obstacles':sum(r[0]<da['environment']['bounds'][1][0] for r in da['environment']['buildings']),
            'mission_seconds':da['mission_seconds'],'seed':da['seed'],'aware':ma,'blind_filter':mb,
            'reduction_percent':value,'valid_safe_comparison':safe,
            'substantial_reduction':bool(safe and value>=15),
            'settings':da['config']}
        if s.name=='interchange':
            row['title']='Four-branch inspection network'
            row['metadata_note']='The recorded geometry has west, east, north and south branches; an earlier title called it three-branch.'
        digest=hashlib.sha256((ma['sha256']+mb['sha256']+PLAYBACK_VERSION).encode()).hexdigest()[:12]
        data_name=f'data/{s.name}-{digest}.json.deflate'
        data_path=a.out/data_name
        if not data_path.exists():
            packed=zlib.compress(json.dumps({'aware':pa,'blind_filter':pb},separators=(',',':'),allow_nan=False).encode(),3)
            temporary=data_path.with_suffix('.tmp');temporary.write_bytes(packed);temporary.replace(data_path)
        row['playback_url']=data_name
        rows.append(row)
    summary={'playback_version':PLAYBACK_VERSION,
        'comparison':'Disturbance-aware expected-cost MBD versus disturbance-blind planning, common learned physical execution filter',
        'protocol':{'aware':'Predicts posterior-field-dependent motion under the execution-filter model; minimizes expected monitoring cost plus soft penalties.',
            'blind_filter':'Predicts nominal zero-wind motion without execution-filter intervention during planning; uses the same wind-informed controller and filter during physical-model execution.'},
        'status':'complete' if not pending else 'running','rows':rows,'pending':pending,
        'complex_environment_criterion':'at least6 real obstacles, at least4 monitored areas, distinct geometry and task distribution',
        'numerical_evidence':'Cost is independently recomputed from saved100Hz clarity values using the trapezoidal integral of mean positive target shortfall. Contact cases are not counted as successful performance contrasts.',
        'initial_information':'64 noisy shared pre-mission measurements selected only from free geometry; this isolates planning awareness with an initially mapped field.',
        'limitations':['Development-selected synthetic spatial fields, not CFD or hardware validation.',
            'One paired optimizer/measurement seed unless otherwise labeled; these cases are not a held-out benchmark.',
            'The initial field is mapped using64 shared synthetic noisy measurements; this is not a wind-map-from-scratch experiment.',
            'Planning is planar at3.2m; physical execution uses the nonlinear3D quadrotor.',
            'Display frames every0.5s; quantitative checks use dense100Hz archives.',
            'Missions use simulated execution time. The simulator pauses its mission clock while planning.',
            'Planning timers include competing jobs; no hard deadline guarantee.'],
        'definitions':{'substantial_reduction':'at least15 percent lower integrated monitoring deficit, both full missions completed without contact'}}
    if selection:summary['case_selection']={'manifest':str(a.selection),'sha256':sha(a.selection),'declaration':selection}
    timing=ROOT/'output/disturbance_suite/runtime_benchmark.json'
    if selection and timing.exists():summary['single_process_timing']=json.loads(timing.read_text())
    (a.out/'results.json').write_text(json.dumps(summary,indent=2))
    template=(ROOT/'docs/disturbance_suite/report_template.html').read_text()
    template=template.replace('__REPORT__',json.dumps(summary,separators=(',',':')))
    temporary=a.out/'index.tmp';temporary.write_text(template);temporary.replace(a.out/'index.html')
    print(json.dumps({'completed_pairs':len(rows),'pending':pending,'safe_pairs':sum(r['valid_safe_comparison'] for r in rows),'substantial':sum(r['substantial_reduction'] for r in rows),'html_bytes':(a.out/'index.html').stat().st_size},indent=2))
    for r in rows:print(r['case'],f"{r['reduction_percent']:+.1f}%",'SAFE' if r['valid_safe_comparison'] else 'CONTACT - excluded from success claims')


if __name__=='__main__':main()
