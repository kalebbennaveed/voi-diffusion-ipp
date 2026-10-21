"""Replay the measured contact segments through the nonlinear plant.

Fixtures contain the original MBD decisions and posterior-derived fields,
not replacement paths. Source audit hashes are recorded alongside them.
"""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from voi_diffusion_ipp.quarter.continuous import ControlSettings
from voi_diffusion_ipp.quarter.environment import environment,wind
from voi_diffusion_ipp.quarter.execution import executor


@pytest.mark.parametrize('fixture',['two_crosswind_execution','multi_crosswind_execution','multi_crosswind_late_execution'])
def test_hard_execution_filter_survives_recorded_contact_segment(fixture):
    base=Path(__file__).parent/'fixtures'/fixture
    record=json.loads(base.with_suffix('.json').read_text())
    cfg=ControlSettings(**record['config'])
    env=environment(record['layout'],cfg.environment_scale,cfg.inspection_scale,'crosswind')
    execute=executor(env,cfg,lambda p:wind(p,'crosswind',cfg.wind_strength,cfg.environment_scale))
    with np.load(base.with_suffix('.npz')) as a:
        _,logs=execute(jnp.asarray(a['initial_state'],jnp.float32),jnp.asarray(a['theta']),
                       jnp.asarray(a['mean']),jnp.asarray(a['construction']))
    states,_,motors,_,controls,slacks=map(np.asarray,logs)
    gaps=np.asarray(jax.vmap(env.clearance)(jnp.c_[states[:,:2],states[:,3:5]]))-.275
    assert len(states)==500
    assert np.isfinite(states).all() and np.isfinite(motors).all()
    assert gaps.min()>.15
    assert np.max(np.abs(controls))<=cfg.umax+1e-4
    assert (slacks>=0).all()  # Infeasibility remains visible rather than hidden.


def test_execution_period_validation():
    with pytest.raises(ValueError,match='Execution control period'):
        ControlSettings(dt=.2,execution_dt=.03)
