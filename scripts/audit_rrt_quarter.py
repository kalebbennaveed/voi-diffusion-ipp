"""Reconstruct published flight samples and mission costs from 100 Hz records.

This checks that the replay is the nonlinear integration output, not an RRT or
nominal-path animation. It does not establish generalization beyond these runs.
"""
import argparse
import json
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation


def audit(path):
    d=json.loads(path.read_text())
    if not d.get('completed') or d.get('plan_only') or d.get('config',{}).get('initialization')!='rrt':
        return None
    root=path.parent
    elapsed=0.; cost=0.; positions=[np.array(d['frames'][0]['p'])]; quats=[np.array(d['frames'][0]['quat'])]; times=[0.]
    minimum=float('inf'); maximum_orthogonality=0.; last_q=np.array(d['frames'][0]['q'])
    continuation_check=None
    if d.get('continuation'):
        provenance=d['continuation'];source=Path(provenance['source'])
        assert hashlib.sha256(source.read_bytes()).hexdigest()==provenance['source_sha256']
        prefix=json.loads(source.read_text());n=len(prefix['plans'])
        assert d['frames'][:len(prefix['frames'])]==prefix['frames']
        assert d['observations'][:len(prefix['observations'])]==prefix['observations']
        assert d['beliefs'][:len(prefix['beliefs'])]==prefix['beliefs']
        assert d['plans'][:n]==prefix['plans']
        for plan in prefix['plans']:
            name=plan['execution_audit']
            assert (root/name).read_bytes()==(source.parent/name).read_bytes()
        continuation_check=dict(verified=True,prefix_seconds=prefix['completed_seconds'],
                                unchanged_prefix_frames=len(prefix['frames']),prefix_planning_calls=n,
                                source_sha256=provenance['source_sha256'])
    theta_changes=[]
    for p in d['plans']:
        with np.load(root/p['execution_audit']) as e:
            np.testing.assert_allclose(e['theta'],p['policy'],atol=1e-7,rtol=0)
            states=e['states'];n=len(states)
            assert states.shape[1]==22 and np.isfinite(states).all()
            assert np.all(e['motor_commands']>=0)
            np.testing.assert_allclose(e['initial_state'][:3],positions[-1],atol=7e-6,rtol=0)
            # State pack: position, velocity, flattened rotation, angular rate,
            # four motor angular velocities.
            rotations=states[:,6:15].reshape(-1,3,3)
            maximum_orthogonality=max(maximum_orthogonality,float(np.max(np.abs(rotations.transpose(0,2,1)@rotations-np.eye(3)))))
            for j in range(9,n,10):
                positions.append(states[j,:3]);quats.append(Rotation.from_matrix(rotations[j]).as_quat());times.append(elapsed+.01*(j+1))
            if n%10:
                positions.append(states[-1,:3]);quats.append(Rotation.from_matrix(rotations[-1]).as_quat());times.append(elapsed+.01*n)
            elapsed+=n*.01
            cost+=float(e['cost_increments'].sum(dtype=np.float64))
            minimum=min(minimum,float(e['clearances'].min()))
            last_q=e['clarity'][-1].astype(float)
        if p.get('audit'):
            with np.load(root/p['audit']) as a:
                np.testing.assert_allclose(a['chosen_theta'],p['policy'],atol=1e-7,rtol=0)
                theta_changes.append(float(np.min(np.linalg.norm(a['initial_parameters']-a['chosen_theta'],axis=1))))
    np.testing.assert_allclose(times,[f['t'] for f in d['frames']],atol=1e-5,rtol=0)
    np.testing.assert_allclose(positions,[f['p'] for f in d['frames']],atol=6e-6,rtol=0)
    q0=np.asarray(quats);q1=np.asarray([f['quat'] for f in d['frames']]);sign=np.sign(np.sum(q0*q1,axis=1));sign[sign==0]=1
    np.testing.assert_allclose(q0*sign[:,None],q1,atol=6e-7,rtol=0)
    if d['failure']:
        # Include deterioration after termination exactly as in the mission
        # definition, instead of rewarding a short failed recording.
        for _ in range(round(max(0,d['mission_seconds']-elapsed)/.1)):
            last_q=last_q/(1+np.array(d['environment']['process_noise'])*last_q*.1)
            cost+=.1*np.maximum(np.array(d['environment']['targets'])-last_q,0).mean()
    np.testing.assert_allclose(cost,d['cost'],atol=2e-4,rtol=0)
    np.testing.assert_allclose(minimum,d['min_clearance'],atol=1e-6,rtol=0)
    assert maximum_orthogonality<2e-4
    return dict(file=path.name,case=d['case'],method=d['method'],seed=d['seed'],success=d['success'],
        continuation_check=continuation_check,recorded_seconds=elapsed,reconstructed_cost=cost,reported_cost=d['cost'],
        replay_frames=len(times),maximum_rotation_orthogonality_error=maximum_orthogonality,
        first_plan_distance_to_nearest_initial_control_sequence=theta_changes[0] if theta_changes else None,
        decision_origins=dict(Counter(p['decision_origin'] for p in d['plans'])),
        maximum_rrt_seconds=max(p['rrt_seconds'] for p in d['plans']),
        mean_guidance_used_fraction=float(np.mean([p['guidance_used_fraction'] for p in d['plans']])))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',required=True);a=parser.parse_args();root=Path(a.results)
    records=[r for p in sorted(root.glob('*.json')) if (r:=audit(p)) is not None]
    direction_checks=[]
    for r in records:
        if r['case']!='forward':continue
        reverse=next((v for v in records if v['case']=='reverse' and v['method']==r['method'] and v['seed']==r['seed']),None)
        if reverse:
            forward_data=json.loads((root/r['file']).read_text());reverse_data=json.loads((root/reverse['file']).read_text())
            assert forward_data['config']==reverse_data['config']
            np.testing.assert_array_equal(forward_data['plans'][0]['policy'],reverse_data['plans'][0]['policy'])
            direction_checks.append(dict(method=r['method'],seed=r['seed'],first_plan_identical=True,
                meaning='Before any observations, reversing the hidden physical wind does not change the initial planned controls.'))
    (root/'execution_integrity.json').write_text(json.dumps(dict(check='Replay positions, attitudes, timestamps, controls, costs and clearances reconstructed from 100 Hz execution audits.',runs=records,initial_plan_direction_checks=direction_checks),indent=2,allow_nan=False))
    for r in records:print(r['file'],r['replay_frames'],'frames verified; cost',round(r['reconstructed_cost'],6),r['decision_origins'])

if __name__=='__main__':main()
