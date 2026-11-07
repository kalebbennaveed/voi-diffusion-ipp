import gzip,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
from replay_prefix import ReplayPrefix


def prefix(tmp_path):
    path=tmp_path/'prefix.json.gz'
    data=dict(id='s2-center-pathmask',summary=dict(contact=None,gate_time_s=0.,simulated_seconds=.2),
        plans=[],task_frames=[],observations=[[0.,0.,0.,.1,.2],[.2,.2,.1,.3,.4]],
        samples=[[i*.05,i*.05,i*.025,1.,.5] for i in range(5)],
        clarity_frames=[dict(t=.2,q=[.123457])],handoff={},handoff_validation=[],
        observation_predictions=[dict(mean=[0.,0.],std=1.),dict(mean=[.2,.3],std=.5)])
    with gzip.open(path,'wt') as f:json.dump(data,f)
    return ReplayPrefix(path)


def test_replay_checks_full_precision_readings_and_rounded_display_values(tmp_path):
    r=prefix(tmp_path)
    r.verify_state(.2,[.2,.1,1.,.5],[.1234567])
    r.verify_observation(.2,[.2,.1],[.3,.4],[.2,.3],.5)
    assert r.audit['checked_observations']==1
    assert r.audit['max_position_error_m']==0
    assert r.audit['max_recorded_clarity_error']<5e-7


def test_replay_stops_on_divergent_physics_or_posterior(tmp_path):
    r=prefix(tmp_path)
    with pytest.raises(RuntimeError,match='state'):r.verify_state(.2,[.21,.1,1.,.5],[.1234567])
    with pytest.raises(RuntimeError,match='predictive_mean'):r.verify_observation(.2,[.2,.1],[.3,.4],[.21,.3],.5)
    with pytest.raises(RuntimeError,match='clarity'):r.verify_state(.2,[.2,.1,1.,.5],[np.nan])
