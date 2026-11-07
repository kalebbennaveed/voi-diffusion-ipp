"""Regressions for adapting the original optimizer to the circular setup."""
import sys
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import numpy as np
import jax
import jax.numpy as jnp
import legacy_monitoring as legacy
import monitoring_mbd as m
from voi_diffusion_ipp.quarter import continuous as old
from voi_diffusion_ipp.quarter.environment import environment,table_shape

def test_original_optimizer_default_is_identical_to_explicit_original_rollout():
    cfg=old.ControlSettings(horizon=.4,dt=.2,commit=.2,iterations=1,proposals=2,batch_size=2,worlds=1,validation_worlds=1,construction_worlds=1,geometry='compact_detour',environment_scale=.35,inspection_scale=.55,guidance_mode='off')
    env=environment('two',.35,.55,'crosswind','compact_detour');x=jnp.array([-4.6,1.,0.,0.]);q=jnp.full(len(env.cells),.65)
    field=jnp.zeros(table_shape('compact_detour'));args=(x,q,field,field[None],field[None],field[None],jnp.zeros(4),jax.random.PRNGKey(1),jnp.array(1.),jnp.array(False))
    a=old.optimizer(env,cfg,'blind_filter')(*args)
    b=old.optimizer(env,cfg,'blind_filter',rollout_fn=old.rollout)(*args)
    for aa,bb in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):np.testing.assert_array_equal(aa,bb)

def test_circle_forecast_contact_is_absorbing_with_full_horizon_decay():
    cfg=replace(legacy.Settings(),horizon=2.,commit=1.)
    env,_=m.environment(legacy.physical_settings(cfg));q=jnp.full(len(env.cells),.74)
    field=jnp.zeros(legacy.SHAPE);x=jnp.array([.01,.8,-1.,0.]);theta=jnp.zeros(2*cfg.steps)
    result=jax.jit(lambda:legacy.rollout(theta,x,q,field,field[None].repeat(2,axis=0),field,env,cfg,False))()
    assert bool(result.terminated)
    np.testing.assert_allclose(result.x[1:,:2],np.tile(np.asarray(x[:2]),(cfg.steps,1)),atol=1e-12)
    np.testing.assert_allclose(result.q[-1],q/(1+cfg.horizon*env.process_noise*q),atol=1e-12)
    assert float(result.monitor)>0

def test_initial_geometric_bank_contains_both_bypasses():
    cfg=legacy.Settings();env,_=m.environment(legacy.physical_settings(cfg));x=jnp.array([1.45,.875,0.,0.]);q=jnp.full(len(env.cells),.74)
    seeds,previous,meta=legacy.seed_bank(x,q,env,cfg)
    assert seeds.shape==(7,120) and previous.shape==(120,)
    paths=[np.asarray(p) for p in meta['coarse_paths']]
    assert any(p[:,1].max()>4.2 for p in paths)
    assert any(p[:,1].min()<.25 for p in paths)
    assert np.max(np.abs(cfg.umax*np.tanh(seeds)))<=cfg.umax

def test_zero_wind_table_matches_direct_zero_wind():
    field=jnp.zeros(legacy.SHAPE);positions=jnp.array([[.1,.1],[4.,2.],[7.9,4.5]])
    np.testing.assert_array_equal(legacy.BASIS.evaluate(positions,field),jnp.zeros((3,2)))
    grad=jax.jacfwd(lambda x:legacy.BASIS.evaluate(x,field)[0])(positions[1]);np.testing.assert_array_equal(grad,jnp.zeros((2,2)))
