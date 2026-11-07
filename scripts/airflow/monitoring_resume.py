"""Lossless continuation of an interrupted monitoring phase; no reinitialization."""
import gzip,hashlib,json,pickle
from pathlib import Path
import numpy as np
from profile_io import load_profile

class MonitoringResume:
    def __init__(self,path):
        self.path=Path(path)
        with gzip.open(self.path,'rt') as f:self.record=json.load(f)
        r=self.record;self.id=r['id'];s=r['summary']
        if s['contact'] is not None:
            raise ValueError('Resume requires a contact-free prefix')
        self.policy_state=None
        if s['gate_time_s'] is None:
            policy_path=self.path.with_name(self.id+'-policy-checkpoint.pkl')
            if not policy_path.exists():raise ValueError('Learning continuation requires the exact saved policy state')
            with policy_path.open('rb') as f:self.policy_state=pickle.load(f)
            if self.policy_state['t']!=s['simulated_seconds']:raise ValueError('Policy checkpoint time differs')
        self.protocol=json.loads(self.path.with_name(self.id+'-protocol.json').read_text())
        if self.protocol.get('adaptive_learning'):raise ValueError('Adaptive policy continuation is not supported')
        self.state=np.load(self.path.with_name(self.id+'-execution-checkpoint.npz'))
        if abs(float(self.state['t'])-s['simulated_seconds'])>1e-8:raise ValueError('Checkpoint and recording times differ')
        for key in ('x','q','refs','accelerations','previous','gate_time'):
            if key not in self.state or not np.isfinite(self.state[key]).all():raise ValueError('Missing or nonfinite checkpoint field: '+key)
        if self.state['x'].shape!=(4,) or np.any((self.state['q']<0)|(self.state['q']>1)):
            raise ValueError('Invalid physical state in checkpoint')
        expected_gate=-1. if s['gate_time_s'] is None else s['gate_time_s']
        if float(self.state['gate_time'])!=expected_gate:raise ValueError('Checkpoint handoff differs from recording')
        self.gp=load_profile(self.path.with_name(self.id+'-final-profile.pkl'))
        o=np.asarray(r['observations'])
        np.testing.assert_array_equal(self.gp.positions,o[:,1:3])
        np.testing.assert_array_equal(self.gp.measurements,o[:,3:5])
        reference=np.load(self.path.with_name(self.id+'-posterior.npz'))
        for key,value in [('mean',self.gp.belief.mean),('covariance',self.gp.belief.covariance),
                          ('inverse',self.gp.belief.inverse),('centers',self.gp.belief.basis.centers)]:
            np.testing.assert_allclose(value,reference[key],atol=1e-10,rtol=1e-10)
        self.audit=dict(source=str(self.path),sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            resume_at_s=s['simulated_seconds'],posterior_matches_saved_arrays=True,
            observations_match_exactly=True,
            definition='Restore full-precision robot, velocity, clarity, reference and previous MBD parameters, plus complete POAM sufficient statistics, optimizer state and RNG. Preserve actual observation history and original absolute random-key schedule. No physical reset or synthetic observation.')

    def validate(self,ident,protocol):
        if ident!=self.id:raise ValueError('Continuation identity differs')
        protocol=json.loads(json.dumps(protocol))
        for key in ['comparison_method','initial_condition','settings','gp_profile','environment_spec_sha256',
                    'efficient_planning','factorized_support','exact_safety_variance','allow_validation_slack']:
            if self.protocol[key]!=protocol[key]:raise ValueError('Continuation changed '+key)
        old=dict(self.protocol['physical_settings']);new=dict(protocol['physical_settings'])
        old.pop('mission_seconds',None);new.pop('mission_seconds',None)
        if old!=new:raise ValueError('Continuation physical parameters differ')
