"""Validate the complete exported comparison, independently of the renderer."""
import hashlib
import json
from pathlib import Path

import numpy as np

root=Path('output/urban_quad')
summary={}
for case in ['forward','reverse','crosswind','commitment']:
    summary[case]={}
    for method in ['proposed','aware','blind']:
        path=root/f'{case}_{method}_8101.json';d=json.loads(path.read_text())
        assert d['model_revision']=='matched_limits_v3'
        assert d['mission_seconds']==120 and d['success'],(case,method,d['failure'])
        assert abs(d['completed_seconds']-120)<1e-6
        assert all(p['predicted_cost'] is not None for p in d['plans'])
        assert len(d['plans'])==60
        p=np.array([f['p'] for f in d['frames']]);q=np.array([f['q'] for f in d['frames']])
        rotations=np.array([f['quat'] for f in d['frames']]);motors=np.array([f['motors'] for f in d['frames']])
        times=np.array([f['t'] for f in d['frames']])
        assert np.isfinite(p).all() and np.isfinite(q).all() and np.all((q>=0)&(q<=1))
        assert np.max(np.abs(np.linalg.norm(rotations,axis=1)-1))<2e-6
        assert motors.min()>=0 and motors.max()<=d['vehicle']['max_motor_speed']+.01
        assert d['min_clearance']>0
        cost=np.trapezoid(np.maximum(.75-q,0).mean(axis=1),times)
        assert abs(cost-d['cost'])<.025,(case,method,cost,d['cost'])
        std=np.array([b['std'] for b in d['beliefs']])
        assert np.isfinite(std).all() and std.min()>=0 and std[-1].mean()<std[0].mean()
        summary[case][method]={k:d[k] for k in ['cost','min_clearance','rms_tracking_error','max_tracking_error','wall_seconds']}
        summary[case][method].update(mean_std_start=float(std[0].mean()),mean_std_end=float(std[-1].mean()),
                                    altitude_range=[float(p[:,2].min()),float(p[:,2].max())],sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for method in ['proposed','aware']:
        summary[case][method]['reduction_vs_blind_percent']=100*(1-summary[case][method]['cost']/summary[case]['blind']['cost'])
forward=json.loads((root/'forward_proposed_8101.json').read_text())
reverse=json.loads((root/'reverse_proposed_8101.json').read_text())
np.testing.assert_allclose(forward['environment']['wind'],-np.asarray(reverse['environment']['wind']),atol=1e-6)
cross=json.loads((root/'crosswind_proposed_8101.json').read_text());points=np.asarray(cross['environment']['grid'])
assert not np.any(np.asarray(cross['environment']['wind'])[points[:,1]>22])
(root/'validation.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print('PASS: 12 full missions; finite selected policies; collision-free recorded motion; normalized attitudes; motor bounds; independent cost integration; decreased uncertainty; reversed field; calm B.')
