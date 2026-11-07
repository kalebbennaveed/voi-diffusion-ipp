"""Export one fixed vector field per scenario, with numerical provenance.

Time averaging defines the requested stationary mean-flow environment. It does
not claim that the instantaneous LES flow is steady or fully grid converged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path('voi-diffusion-ipp/results/airflow')
WEB=Path('sites/belief-adaptive-monitoring/dist/data')
SOURCE=Path('voi-diffusion-ipp/scripts/airflow')

def load(scenario,n):
    p=ROOT/f'stationary_s{scenario}_n{n}'
    m=json.loads((p/'metadata.json').read_text())
    raw=(p/'velocity.f32').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==m['velocity_sha256']
    assert all('junction_halfway_no_slip_links' in q for q in m['curved_boundary_checks'])
    assert m['slice_height_m']==.75
    return m,np.frombuffer(raw,dtype='<f4').reshape(m['data_shape'])

def mean_between(m,v,start,end):
    t=np.asarray(m['times_s'])
    # Solver duration is rounded to an integral step. Never extrapolate a
    # missing endpoint; a sub-step shortfall uses the actual final time.
    if end>t[-1]:
        assert end-t[-1]<=m['dt_s']
        end=float(t[-1])
    assert t[0]<=start<end<=t[-1]
    def endpoint(time):
        i=np.searchsorted(t,time)-1
        a=(time-t[i])/(t[i+1]-t[i])
        return (1-a)*v[i].astype(float)+a*v[i+1]
    keep=(t>start)&(t<end)
    samples=np.concatenate([endpoint(start)[None],v[keep],endpoint(end)[None]])
    times=np.r_[start,t[keep],end]
    return np.trapezoid(samples,times,axis=0)/(end-start)

def metrics(m,v):
    xx,yy=np.meshgrid((np.arange(m['grid'][0])+.5)*m['dx_m'],(np.arange(m['grid'][1])+.5)*m['dx_m'])
    valid=np.ones_like(xx,dtype=bool)
    for x,y,r,h in m['obstacles']:valid&=(xx-x)**2+(yy-y)**2>r*r
    dx,dy=xx-4.5425,yy-1.8975
    along=-.671*dx-.741*dy;across=.741*dx-.671*dy
    masks={'central_passage':(abs(along)<.65)&(abs(across)<.18)&valid,
           'lower_bypass':(xx>4.7)&(xx<6.1)&(yy>.06)&(yy<.26)&valid,
           'upper_bypass':(xx>1.3)&(xx<3.0)&(yy>4.38)&(yy<4.57)&valid,
           'whole_fluid_slice':valid}
    speed=np.linalg.norm(v,axis=-1)
    return {name:float(speed[mask].mean()) for name,mask in masks.items()}

def relative_changes(first,second):
    return {key:abs(second[key]-first[key])/max(second[key],.1) for key in first}

def export(n):
    WEB.mkdir(exist_ok=True)
    rows=[];sources={}
    for scenario in (1,2):
        m,v=load(scenario,n)
        mean=mean_between(m,v,8.,16.).astype('<f4')
        first=mean_between(m,v,8.,12.);second=mean_between(m,v,12.,16.)
        block_change=relative_changes(metrics(m,first),metrics(m,second))
        row={'scenario':scenario,'grid':m['grid'],'slice_height_m':.75,
             'fixed_field_mean_speeds_m_s':metrics(m,mean),
             'sampling_comparison':'Mean vector fields over 8–12 s versus 12–16 s',
             'sampling_passage_relative_changes':block_change,
             'sampling_vector_relative_L2':float(np.linalg.norm(second-first)/np.linalg.norm(second)),
             'boundary_checks':m['curved_boundary_checks']}
        if n!=64:
            cm,cv=load(scenario,64);coarse=mean_between(cm,cv,8.,16.)
            row['coarse_grid']=cm['grid'];row['coarse_fixed_field_mean_speeds_m_s']=metrics(cm,coarse)
            row['grid_relative_changes']=relative_changes(metrics(cm,coarse),metrics(m,mean))
        rows.append(row)
        sources[f'scenario{scenario}']={'source_directory':f'stationary_s{scenario}_n{n}',
          'transient_velocity_sha256':m['velocity_sha256']}
        metadata=dict(m)
        for key in ['times_s','data_shape','data_file','data_axes','velocity_sha256']:metadata.pop(key)
        metadata.update(time_dependence='none',field_definition='Fixed vector mean of the 3D CFD horizontal slice over solver time 8–16 s; no temporal interpolation or gusts in the environment',
          averaging_interval_s=[8,min(16,m['times_s'][-1])],averaging_method='Trapezoidal vector average, linearly interpolated window endpoints; no extrapolation, speed clipping, or minimum wind',
          data_shape=list(mean.shape),data_file='stationary-velocity.f32',data_axes=['y from south','x from west','u/v/w'],
          velocity_sha256=hashlib.sha256(mean.tobytes()).hexdigest(),generator=sources[f'scenario{scenario}'],
          stationary_model_limit='A mean-flow environment, not the instantaneous turbulent flow and not an exact steady Navier–Stokes solution',
          validation_status='Provisional numerical field; full-scene accuracy and grid convergence not established',
          intended_use='Scenario visualization; not validated for robot performance or research conclusions')
        out=WEB/f'scenario{scenario}';out.mkdir(exist_ok=True)
        (out/metadata['data_file']).write_bytes(mean.tobytes())
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
        # Only the stationary field is served; superseded transient records are
        # retained in results/airflow for audit rather than shipped in the site.
        old=out/'velocity.f32'
        if old.exists():old.unlink()
    sampling=max(max(r['sampling_passage_relative_changes'].values()) for r in rows)
    grid=max((max(r.get('grid_relative_changes',{'none':0}).values()) for r in rows))
    report={'time_independent':True,'analytic_solver_check':json.loads((ROOT/'solver_verification.json').read_text()),
      'scenarios':rows,'full_scene_grid_convergence_established':False,
      'comparison_note':'Both resolutions use the same 0.75 m slice and 8–16 s averaging window. Passage masks are sampled on each grid; these are sensitivity checks, not a rigorous grid-convergence or experimental validation study.',
      'summary':f'The field is constant in time by construction. Taylor–Green solver check passed (1.96% → 0.49% relative velocity error). Passage mean-speed changes between two averaging blocks reach {100*sampling:.1f}%.'+(f' The 64→{n} grid change reaches {100*grid:.1f}%.' if n!=64 else ' This is the coarse computation while refinement runs.')+' Full scene grid convergence and wall-resolved turbulence are not established.'}
    if (ROOT/'refinement_failure.json').exists():
        report['refinement_failure']=json.loads((ROOT/'refinement_failure.json').read_text())
        report['summary']=f'The displayed fields are fixed in time. Taylor–Green solver check passed (1.96% → 0.49% relative velocity error). The 128-cell Scenario 2 run failed its numerical stability gate and was rejected. Both displayed fields use the completed 64-cell computations. Their passage speeds change by up to {100*sampling:.1f}% between averaging blocks, but this does not establish accuracy. This is a provisional scenario preview, not a validated wind model.'
        if (ROOT/'stationary_s1_n128'/'metadata.json').exists():
            fm,fv=load(1,128);cm,cv=load(1,64)
            report['scenario1_refinement_sensitivity']=relative_changes(metrics(cm,mean_between(cm,cv,8,16)),metrics(fm,mean_between(fm,fv,8,16)))
            largest=max(report['scenario1_refinement_sensitivity'].values())
            report['summary']+=f' Scenario 1 also shows up to {100*largest:.0f}% passage-speed change with refinement, so its numerical accuracy is unresolved.'
    (WEB/'validation.json').write_text(json.dumps(report,indent=2));(ROOT/'stationary_validation.json').write_text(json.dumps(report,indent=2))
    files=[SOURCE/'simulate_sketch_airflow.py',SOURCE/'verify_flow_solver.py',SOURCE/'export_stationary_wind.py',SOURCE/'verify_stationary_adapter.cjs',SOURCE/'README.md']
    with zipfile.ZipFile(WEB/'airflow-source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for file in files:archive.write(file,file.name)
        archive.write(ROOT/'EXPERIMENT_STATUS.md','EXPERIMENT_STATUS.md')
        if (ROOT/'refinement_failure.json').exists():archive.write(ROOT/'refinement_failure.json','refinement_failure.json')
        archive.writestr('requirements.txt','lbmpy==1.4.1\npystencils==1.4\nnumpy==2.4.6\nscipy==1.17.1\nsympy==1.12.1\nCython==3.3.1\n')
        archive.writestr('REPRODUCE.txt','Python 3.11.15. Install requirements in an isolated environment.\nSet PYSTENCILS_CACHE_DIR to a writable directory and PYSTENCILS_CONFIG to compiler.json.\nCompiler configuration is included. Serial CPU, no OpenMP required.\nPlace scripts under voi-diffusion-ipp/scripts/airflow and run from the workspace root. Create sites/belief-adaptive-monitoring/dist/data before export.\nFor each scenario S=1/2 and grid N=64:\npython simulate_sketch_airflow.py --nx N --scenario S --seconds 16 --sample .1 --output voi-diffusion-ipp/results/airflow/stationary_sS_nN\nRun verify_flow_solver.py, then export_stationary_wind.py --nx 64 to reproduce the displayed provisional fields.\nThe analogous N=128 Scenario 2 check failed the numerical stability gate. Its result must not be substituted for a completed simulation.\nCopy the included refinement_failure.json and EXPERIMENT_STATUS.md into voi-diffusion-ipp/results/airflow for the report/source archive.\nThe fixed field is a vector average over 8–16 s; it does not replay transient CFD frames.\n')
        archive.writestr('compiler.json',json.dumps({'compiler':{'command':'clang++','flags':'-O3 -DNDEBUG -fPIC -std=c++11','os':'darwin','restrict_qualifier':'__restrict__'},'cache':{'object_cache':'/private/tmp/airflow-pystencils-object'}}))
        archive.writestr('source-sha256.json',json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--nx',type=int,choices=[64,128],default=128)
    export(parser.parse_args().nx)
