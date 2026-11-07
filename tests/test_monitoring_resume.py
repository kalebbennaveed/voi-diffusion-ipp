import gzip,json,sys,pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
from monitoring_resume import MonitoringResume
from profile_io import save_profile

def fixture(tmp_path,contact=None):
    ident='test';path=tmp_path/'test.json.gz'
    record=dict(id=ident,summary=dict(contact=contact,gate_time_s=0.,simulated_seconds=.2),
        observations=[[0.,1.,2.,.1,.2]])
    with gzip.open(path,'wt') as f:json.dump(record,f)
    proto={k:{} for k in ['comparison_method','initial_condition','settings','gp_profile','environment_spec_sha256',
        'efficient_planning','factorized_support','exact_safety_variance','allow_validation_slack']}
    proto['physical_settings']={'mass':.680,'mission_seconds':5};proto['gp_profile']={'lengths':[.1,.2]}
    (tmp_path/'test-protocol.json').write_text(json.dumps(proto))
    values=dict(mean=np.array([[.1,.2]]),covariance=np.eye(1),inverse=np.eye(1),centers=np.array([[1.,2.]]))
    np.savez(tmp_path/'test-posterior.npz',**values)
    belief=SimpleNamespace(**{k:v for k,v in values.items() if k!='centers'},basis=SimpleNamespace(centers=values['centers']))
    gp=SimpleNamespace(positions=np.array([[1.,2.]]),measurements=np.array([[.1,.2]]),belief=belief)
    save_profile(tmp_path/'test-final-profile.pkl',gp)
    np.savez(tmp_path/'test-execution-checkpoint.npz',t=.2,x=np.array([1.123456789012345,2.,0.,0.]),
        q=np.array([.7]),refs=np.zeros((2,4)),accelerations=np.zeros((1,2)),previous=np.zeros(2),gate_time=0.)
    return path,proto

def test_full_precision_restore_and_configuration_check(tmp_path):
    path,proto=fixture(tmp_path);resume=MonitoringResume(path)
    assert resume.state['x'][0]==1.123456789012345
    proto['gp_profile']['lengths']=(.1,.2) # JSON lists and in-memory tuples are equivalent.
    resume.validate('test',proto)
    proto['physical_settings']['mass']=1.
    with pytest.raises(ValueError,match='physical'):resume.validate('test',proto)

def test_contact_cannot_be_resumed(tmp_path):
    path,_=fixture(tmp_path,contact={'time_s':.1})
    with pytest.raises(ValueError,match='contact-free'):MonitoringResume(path)

def test_mismatched_posterior_is_rejected(tmp_path):
    path,_=fixture(tmp_path)
    np.savez(tmp_path/'test-posterior.npz',mean=np.array([[9.,9.]]))
    with pytest.raises(AssertionError):MonitoringResume(path)

def test_mismatched_checkpoint_time_is_rejected(tmp_path):
    path,_=fixture(tmp_path);np.savez(tmp_path/'test-execution-checkpoint.npz',t=.4)
    with pytest.raises(ValueError,match='times differ'):MonitoringResume(path)

def test_learning_resume_requires_persistent_policy_state(tmp_path):
    path,_=fixture(tmp_path)
    with gzip.open(path,'rt') as f:r=json.load(f)
    r['summary']['gate_time_s']=None
    with gzip.open(path,'wt') as f:json.dump(r,f)
    with pytest.raises(ValueError,match='exact saved policy'):MonitoringResume(path)
    with (tmp_path/'test-policy-checkpoint.pkl').open('wb') as f:
        pickle.dump(dict(t=.2,policy=SimpleNamespace(target=42),reacher={}),f)
    original=dict(np.load(tmp_path/'test-execution-checkpoint.npz'));original['gate_time']=-1.
    np.savez(tmp_path/'test-execution-checkpoint.npz',**original)
    assert MonitoringResume(path).policy_state['policy'].target==42
