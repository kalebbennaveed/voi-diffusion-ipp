"""Independently audit recorded physics, monitoring cost and mission behavior."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from voi_diffusion_ipp.experiments.urban_airflow import REVISION,wind_velocity

parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
root=Path('output/urban_quad_airflow');summary={};missing=[]
for case in ['forward','reverse','crosswind','commitment']:
    summary[case]={}
    for method in ['proposed','aware','blind']:
        path=root/f'{case}_{method}_8101.json'
        if not path.exists():missing.append(path.name);continue
        d=json.loads(path.read_text())
        if d.get('model_revision')!=REVISION:missing.append(path.name);continue
        assert d['mission_seconds']==360
        assert d['config']['dt']*(d['config']['stage_steps']+d['config']['future_steps'])==(240 if case=='commitment' else 120)
        np.testing.assert_allclose(wind_velocity(np.asarray(d['environment']['grid']),case),d['environment']['wind'],atol=1e-6)
        assert all(p['predicted_cost'] is not None for p in d['plans'])
        frames=d['frames'];p=np.array([f['p'] for f in frames]);q=np.array([f['q'] for f in frames]);t=np.array([f['t'] for f in frames])
        v=np.array([f['velocity'] for f in frames]);w=np.array([f['wind'] for f in frames]);rot=np.array([f['quat'] for f in frames]);motors=np.array([f['motors'] for f in frames])
        assert all(np.isfinite(a).all() for a in [p,q,t,v,w,rot,motors])
        np.testing.assert_allclose(wind_velocity(p[:,:2],case),w,atol=2e-4)
        assert np.all(np.diff(t)>0)
        assert np.all((q>=0)&(q<=1));assert np.max(np.abs(np.linalg.norm(rot,axis=1)-1))<2e-6
        assert motors.min()>=0 and motors.max()<=d['vehicle']['max_motor_speed']+.01
        cost=np.trapezoid(np.maximum(.75-q,0).mean(axis=1),t)
        if d['success']:
            assert abs(d['completed_seconds']-360)<1e-6 and d['min_clearance']>=0
            assert len(d['plans'])==180 and abs(cost-d['cost'])<.04
        else:
            assert d['failure'] and d['completed_seconds']<360
            # The failure convention explicitly charges no further sensing.
            last=q[-1].copy()
            for _ in range(round((360-d['completed_seconds'])/.1)):
                last=last/(1+.025*last*.1);cost+=.1*np.maximum(.75-last,0).mean()
            assert abs(cost-d['cost'])<.06
        std=np.array([b['std'] for b in d['beliefs']]);assert np.isfinite(std).all() and std.min()>=0
        assert np.isfinite(np.array([b['mean'] for b in d['beliefs']])).all()
        survey=d['preflight_survey'];assert len(survey['positions'])==10 and survey['noise_std']==.25
        # Require full, identical serialized preflight data within each case.
        if method!='proposed' and summary[case].get('proposed'):
            parent=json.loads((root/f'{case}_proposed_8101.json').read_text())
            assert survey==parent['preflight_survey']
            assert d['config']==parent['config'] and d['vehicle']==parent['vehicle']
            assert frames[0]==parent['frames'][0]
        crossings=[];side=0;last_side_index=0
        for i,point in enumerate(p):
            new=0 if point[0]<13 else 1 if point[0]>29 else None
            if new is not None:
                if new!=side:
                    segment=p[last_side_index:i+1];core=segment[(segment[:,0]>=14)&(segment[:,0]<=28)]
                    if len(core):
                        y=float(np.median(core[:,1]));passage=['C','A','B'][0 if y<4 else 1 if y<16 else 2]
                        crossings.append(dict(time=float(t[i]),direction='right' if new else 'left',passage=passage))
                side=new;last_side_index=i
        region_costs=[];occupancy=[]
        cells=np.array(d['environment']['cells'])
        for region in d['environment']['regions']:
            a,b,c,e=region;mask=(cells[:,0]>=a)&(cells[:,0]<=c)&(cells[:,1]>=b)&(cells[:,1]<=e)
            region_costs.append(float(np.trapezoid(np.maximum(.75-q[:,mask],0).mean(axis=1),t)))
            inside=(p[:,0]>=a)&(p[:,0]<=c)&(p[:,1]>=b)&(p[:,1]<=e)
            occupancy.append(float(np.sum(np.diff(t)*inside[1:])))
        speed=np.linalg.norm(v[:,:2],axis=1);wind_speed=np.linalg.norm(w,axis=1)
        cosine=np.sum(v[:,:2]*w,axis=1)/np.maximum(speed*wind_speed,1e-8)
        core=(p[:,0]>16)&(p[:,0]<26)&(wind_speed>1.5)&(speed>.05)
        air={}
        for label,mask in [('upwind',core&(cosine<-.85)),('downwind',core&(cosine>.85))]:
            air[label]=dict(samples=int(mask.sum()),median_ground_speed=float(np.median(speed[mask])) if mask.any() else None)
        row={k:d[k] for k in ['success','failure','cost','min_clearance','rms_tracking_error','max_tracking_error','completed_seconds','wall_seconds']}
        row.update(independent_cost=cost,mean_std_start=float(std[0].mean()),mean_std_end=float(std[-1].mean()),
            region_costs=region_costs,region_occupancy_seconds=occupancy,crossings=crossings,wind_traversal=air,
            altitude_range=[float(p[:,2].min()),float(p[:,2].max())],sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        summary[case][method]=row
    if 'blind' in summary[case]:
        for method in ['proposed','aware']:
            if method in summary[case]:summary[case][method]['reduction_vs_blind_percent']=100*(1-summary[case][method]['cost']/summary[case]['blind']['cost'])
if missing and not args.partial:raise RuntimeError('Missing current complete records: '+', '.join(missing))
if all(summary[c] for c in ['forward','reverse']):
    f=json.loads((root/'forward_proposed_8101.json').read_text());r=json.loads((root/'reverse_proposed_8101.json').read_text())
    np.testing.assert_allclose(f['environment']['wind'],-np.asarray(r['environment']['wind']),atol=1e-6)
if summary['crosswind']:
    d=json.loads((root/'crosswind_proposed_8101.json').read_text());grid=np.array(d['environment']['grid'])
    assert not np.any(np.asarray(d['environment']['wind'])[grid[:,1]>=23])
sources=[
 'src/voi_diffusion_ipp/experiments/urban_airflow.py',
 'src/voi_diffusion_ipp/planning/urban_cycle.py',
 'src/voi_diffusion_ipp/planning/mbd.py',
 'src/voi_diffusion_ipp/safety/flow_policy_value.py',
 'src/voi_diffusion_ipp/quadrotor/aerodynamics.py',
 'src/voi_diffusion_ipp/quadrotor/config.py',
 'src/voi_diffusion_ipp/quadrotor/controller.py',
 'src/voi_diffusion_ipp/quadrotor/dynamics.py',
 'src/voi_diffusion_ipp/quadrotor/execution.py',
 'src/voi_diffusion_ipp/wind/poam.py',
 'src/voi_diffusion_ipp/wind/poam_belief.py',
 'scripts/run_urban_quad_airflow.py',
]
source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
(root/'validation.json').write_text(json.dumps(dict(revision=REVISION,missing=missing,source_sha256_at_audit=source_hashes,results=summary),indent=2))
for case,methods in summary.items():
    for method,row in methods.items():print(case,method,round(row['cost'],3),row['success'],len(row['crossings']),'crossings',round(row['min_clearance'],3),'m')
print('Verified',sum(map(len,summary.values())),'records;',len(missing),'outstanding')
