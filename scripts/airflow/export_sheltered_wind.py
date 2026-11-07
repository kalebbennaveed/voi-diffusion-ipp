"""Export physically redesigned inlet/outlet CFD fields; never mask interior wind."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
from export_stationary_wind import mean_between, metrics, relative_changes

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'results/airflow'
SITE=ROOT.parent/'sites/belief-adaptive-monitoring/dist'

def regional_metrics(meta,field):
    xx,yy=np.meshgrid((np.arange(meta['grid'][0])+.5)*meta['dx_m'],(np.arange(meta['grid'][1])+.5)*meta['dx_m'])
    speed=np.linalg.norm(field[...,:2],axis=-1)
    masks={'A':(xx>=4.98)&(yy>=2.295),'B':(xx<=2.9)&(yy<=1.755)}
    return {name:dict(mean_m_s=float(speed[m].mean()),p95_m_s=float(np.quantile(speed[m],.95)),max_m_s=float(speed[m].max())) for name,m in masks.items()}

def export(args):
    dest=args.site;data=dest/'data';data.mkdir(parents=True,exist_ok=True)
    if not (dest/'app.js').exists():shutil.copyfile(SITE/'app.js',dest/'app.js')
    report={'environment_revision':'directional-sheltered-ports-v8','time_independent':True,
        'physical_change':'Physical inlet and pressure-outlet conditions are recorded separately for each scenario. Same columns, monitoring regions, air properties, solver, force law and robot controller. Strong lower inflow is present only in Scenario 2.',
        'interior_velocity_modification':'None: exported vector mean of the newly solved CFD field, no ROI blending, clipping, or speed cap.',
        'full_scene_grid_convergence_established':False,'scenarios':[]}
    for scenario in (1,2):
        prefix=args.source_prefixes[scenario-1] if args.source_prefixes else args.prefix
        directory=OLD/f'{prefix}_s{scenario}_n{args.nx}'
        m=json.loads((directory/'metadata.json').read_text());raw=(directory/'velocity.f32').read_bytes()
        if scenario==1:report['environment_revision']=m['environment_revision']
        assert m['environment_revision']==report['environment_revision']
        assert hashlib.sha256(raw).hexdigest()==m['velocity_sha256']
        v=np.frombuffer(raw,dtype='<f4').reshape(m['data_shape'])
        mean=mean_between(m,v,args.start,args.end).astype('<f4')
        middle=(args.start+args.end)/2
        first,second=[mean_between(m,v,a,b) for a,b in [(args.start,middle),(middle,args.end)]]
        baseline=OLD/f'pre_shelter_stationary/scenario{scenario}'
        previous=json.loads((baseline/'metadata.json').read_text())
        before=np.fromfile(baseline/'stationary-velocity.f32',dtype='<f4').reshape(previous['data_shape'])
        row={'scenario':scenario,'grid':m['grid'],'monitoring_regions_before':regional_metrics(previous,before),
            'openings':m['openings'],'boundary_conditions':m['boundary_conditions'],
            'monitoring_regions':regional_metrics(m,mean),'fixed_field_mean_speeds_m_s':metrics(m,mean),
            'fixed_horizontal_mean_speeds_m_s':metrics(m,mean[...,:2]),
            'sampling_windows_s':[[args.start,middle],[middle,args.end]],
            'sampling_vector_relative_L2':float(np.linalg.norm(second-first)/np.linalg.norm(second)),
            'sampling_region_means_m_s':[regional_metrics(m,f) for f in (first,second)],
            'max_recorded_density_change':max(d['max_relative_density_change'] for d in m['diagnostics'])}
        report['scenarios'].append(row)
        meta=dict(m)
        for key in ('times_s','data_shape','data_file','data_axes','velocity_sha256'):meta.pop(key)
        meta.update(time_dependence='none',field_definition=f'Fixed vector mean of redesigned 3D CFD horizontal slice over {args.start:g}–{args.end:g} s',
            averaging_interval_s=[args.start,min(args.end,m['times_s'][-1])],
            averaging_method='Trapezoidal vector average with interpolated endpoints; no interior masking, rescaling, or clipping.',
            data_shape=list(mean.shape),data_file='stationary-velocity.f32',data_axes=['y from south','x from west','u/v/w'],
            velocity_sha256=hashlib.sha256(mean.tobytes()).hexdigest(),
            generator={'source_directory':str(directory.relative_to(ROOT)),'transient_velocity_sha256':m['velocity_sha256']},
            stationary_model_limit='Prescribed stationary mean-flow environment, not an exact steady Navier–Stokes solution.',
            validation_status='Provisional CFD; monitor-region speeds measured in the computed field. Full-scene grid convergence not established.',
            intended_use='Designed scenario and conditional robot-response review',monitoring_region_metrics=row['monitoring_regions'])
        out=data/f'scenario{scenario}';out.mkdir(exist_ok=True)
        (out/'stationary-velocity.f32').write_bytes(mean.tobytes());(out/'metadata.json').write_text(json.dumps(meta,indent=2))
    report['analytic_solver_check']=json.loads((OLD/'solver_verification.json').read_text())
    report['summary']='The fixed fields are vector means of new 3D CFD solves with the inlet and outlet geometry recorded in each scenario configuration. Monitoring-region speeds are measured from the resulting fields; no calm-zone mask is applied. Taylor–Green solver verification remains applicable. These full scenes have not established grid convergence or experimental accuracy.'
    (data/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',default='sheltered_v2');parser.add_argument('--source-prefixes',nargs=2);parser.add_argument('--nx',type=int,default=64);parser.add_argument('--start',type=float,default=16);parser.add_argument('--end',type=float,default=32);parser.add_argument('--site',type=Path,default=OLD/'sheltered_review');export(parser.parse_args())
