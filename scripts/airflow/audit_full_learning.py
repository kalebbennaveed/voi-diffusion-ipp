"""Independent numerical and provenance checks of the new full recordings."""
from pathlib import Path
import gzip,json
import numpy as np
from package_repaired_validation import wind
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/full_learning_20261104'

def audit(folder):
    environment=json.loads((folder/'environment.json').read_text());free=np.array(environment['free_indices']);nx,ny=environment['grid'];dx=environment['dx_m']
    xx,yy=np.meshgrid((np.arange(nx)+.5)*dx,(np.arange(ny)+.5)*dx);cells=np.c_[xx.ravel(),yy.ravel()]
    reports=[]
    for source in sorted(folder.glob('s*-center-*.json.gz')):
        r=json.load(gzip.open(source));s=np.array(r['samples']);obs=np.array(r['observations']);sm=r['summary'];field=wind(r['scenario'])
        assert np.isfinite(s).all();assert np.abs(s[:,5:7]).max()<=1.6+2.1e-5
        assert np.allclose(np.diff(s[:,0]),.05,atol=1e-7)
        np.testing.assert_allclose(obs[:,1:3],s[np.rint(obs[:,0]/.05).astype(int),1:3],rtol=0,atol=1e-6)
        def derivative(x,u):
            rel=field(x[:2])-x[2:];return np.r_[x[2:],u+(.1/.680)*np.linalg.norm(rel)*rel]
        def step(x,u):
            h=.01;k1=derivative(x,u);k2=derivative(x+h*k1/2,u);k3=derivative(x+h*k2/2,u);k4=derivative(x+h*k3,u)
            return x+h*(k1+2*k2+2*k3+k4)/6
        eligible=np.flatnonzero((s[:-1,18]>0)&(s[1:,18]>0));errors=[]
        for k in eligible[::max(1,len(eligible)//120)]:
            x=s[k,1:5].copy()
            for _ in range(5):x=step(x,s[k+1,5:7])
            errors.append(float(np.max(np.abs(x-s[k+1,1:5]))))
        assert max(errors,default=0)<1e-6
        # Independent clarity integration on actual moving-robot segments.
        # Saved q frames are rounded to 1e-6, so compare at that precision.
        protocol=json.loads(source.with_name(source.name.replace('.json.gz','-protocol.json')).read_text())
        settings=protocol['physical_settings'];sigma=settings['sensor_sigma'];noise=settings['sensor_R']
        Q=np.asarray(environment['process_noise']);sensed_cells=np.asarray(environment['cells'])
        frames={f['t']:np.asarray(f['q']) for f in r['clarity_frames']}
        starts=[t for t in frames if t+1 in frames and s[round((t+1)/.05),18]>0]
        clarity_errors=[]
        for t in starts[::max(1,len(starts)//5)][:5]:
            qq=frames[t].copy()
            for k in range(round(t/.05),round((t+1)/.05)):
                xx=s[k,1:5].copy();u=s[k+1,5:7]
                for _ in range(5):
                    xn=step(xx,u)
                    sensing=lambda p:np.exp(-np.sum((sensed_cells-p)**2,axis=1)/sigma**2)/noise
                    c0,cm,c1=sensing(xx[:2]),sensing((xx[:2]+xn[:2])/2),sensing(xn[:2])
                    f=lambda z,C:C*(1-z)**2-Q*z*z
                    k1=f(qq,c0);k2=f(qq+.005*k1,cm);k3=f(qq+.005*k2,cm);k4=f(qq+.01*k3,c1)
                    qq=np.clip(qq+.01*(k1+2*k2+2*k3+k4)/6,0,1);xx=xn
            clarity_errors.append(float(np.max(np.abs(qq-frames[t+1]))))
        assert max(clarity_errors,default=0)<2e-6
        b=r['belief_frames'][-1];mean=np.array(b['mean']);truth=np.array([field(p) for p in cells]);error=np.linalg.norm(mean-truth,axis=1)
        rmse=float(np.sqrt(np.mean(error[free]**2)));assert abs(rmse-sm['final_rmse_m_s'])<1e-5
        if sm['gate_reached']:
            frames=[b for b in r['belief_frames'] if b['t']<=sm['gate_time_s']]
            if sm.get('gate_kind')=='route uncertainty':
                gate=r['handoff']['route_precision']
                precision=float(np.sqrt(2*np.mean(np.asarray(gate['std'])**2)))
                assert abs(precision-gate['vector_std_m_s'])<1e-10
                assert precision<=sm['task_std_threshold_m_s']
                positions=np.asarray(gate['positions'])
                assert len(np.unique(positions,axis=0))==len(positions)
                assert gate['handoff_route_selected']
                assert r['handoff']['t']==sm['gate_time_s']
            elif sm.get('gate_kind')=='task uncertainty':
                assert min(b['task_vector_std_m_s'] for b in frames if b.get('task_vector_std_m_s') is not None)<=sm['task_std_threshold_m_s']
            else:assert min(b['rmse_m_s'] for b in frames)<=sm['rmse_threshold_m_s']
        monitor=[p for p in r['plans'] if p['phase']=='monitoring'];warm=[p for p in monitor if not p['compilation_included']]
        if protocol.get('model_trust'):
            for plan in monitor:
                validation=plan['validation'];selected=validation['selected']
                if not validation['no_feasible']:
                    assert validation['forecast_error_bounds'][selected]<=.4+1e-6
                    assert validation['forecast_authority_margins'][selected]>=-1.01e-5
        longest_geometry=max((sum(np.linalg.norm(np.diff(np.asarray(p['global_geometric_path']),axis=0),axis=1)) for p in r['plans'] if p.get('global_geometric_path')),default=0.)
        if sm.get('gate_kind')=='route uncertainty':
            # The recorded finite route sampler must not have truncated a
            # long path. Its actual sample count includes each segment end.
            for plan in r['plans']:
                if plan.get('global_geometric_path'):
                    lengths=np.linalg.norm(np.diff(np.asarray(plan['global_geometric_path']),axis=0),axis=1)
                    assert 1+np.maximum(1,np.ceil(lengths/.05)).sum()<=1024
        checkpoint=source.with_name(source.name.replace('.json.gz','-posterior.npz'))
        covariance_check=None
        if checkpoint.exists():
            z=np.load(checkpoint);C=z['covariance'];e=np.linalg.eigvalsh((C+C.T)/2)
            assert np.isfinite(C).all() and e.min()>=-1e-9
            assert np.max(np.abs(C-C.T))<1e-9
            covariance_check=dict(minimum_eigenvalue=float(e.min()),maximum_eigenvalue=float(e.max()),symmetric=True)
        timing={}
        for key in ['fields','geometry','optimization','posterior']:
            vals=[p['timing'][key] for p in warm]
            timing[key]=dict(median=float(np.median(vals)),p95=float(np.percentile(vals,95)),max=float(max(vals))) if vals else None
        reports.append(dict(id=r['id'],completed_300=sm['completed'] and sm['simulated_seconds']==300,
            no_contact=sm['contact'] is None,finite_states=True,commands_in_bounds_with_solver_tolerance=True,max_actuator_overshoot_m_s2=float(max(0.,np.abs(s[:,5:7]).max()-1.6)),measurements_at_robot=True,
            independently_reintegrated_segments=len(errors),max_state_reintegration_error=max(errors,default=0),
            independently_reintegrated_clarity_windows=len(clarity_errors),max_clarity_reintegration_error=max(clarity_errors,default=0),
            rmse_recalculation_error=abs(rmse-sm['final_rmse_m_s']),
            final_inducing_covariance_check=covariance_check,longest_recorded_geometric_path_m=float(longest_geometry),
            monitoring_plans=len(monitor),unvalidated_plan_fallbacks=sum('execution_fallback' in p for p in monitor),
            warm_mbd_deadline_misses=sum(p['wall_seconds']>1 for p in warm),warm_mbd_timing_s=timing,
            summary=sm))
    return reports

if __name__=='__main__':
    reports={}
    for name in ['cold','cold_guard','cold_recovery','cold_final','cold_checked','learned']:
        folder=OUT/name
        if (folder/'summary.json').exists():reports[name]=audit(folder)
    for name in ['matched','information_frontier','reachable','global_goals','adaptive_routes','diversified','route_precision','model_support','adaptive_information','opportunity_information']:
        folder=ROOT/'output/task_learning_20261105'/name
        if (folder/'summary.json').exists():reports[name]=audit(folder)
    (OUT/'audit.json').write_text(json.dumps(reports,indent=2,allow_nan=False));print(json.dumps(reports,indent=2))
