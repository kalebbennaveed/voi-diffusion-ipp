"""Scientific contracts for the added frozen-belief monitoring comparison."""
import sys
from pathlib import Path
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import monitoring_mbd as m
from voi_diffusion_ipp.wind.poam import init_parameters,select_inducing


def test_every_free_cell_has_real_process_noise_and_target():
    cfg=m.Settings();env,grid=m.environment(cfg)
    assert len(env.cells)==len(grid['free_indices'])
    assert np.all(np.asarray(env.targets)>0)
    assert np.all(np.asarray(env.process_noise)>0)
    in_regions=grid['membership'].any(axis=1)
    np.testing.assert_array_equal(np.asarray(env.process_noise)[in_regions],cfg.clarity_decay)
    np.testing.assert_array_equal(np.asarray(env.process_noise)[~in_regions],cfg.background_decay)
    assert cfg.clarity_decay/cfg.background_decay>=8


def test_clarity_matches_independent_covariance_riccati_integration():
    cfg=m.Settings();env,_=m.environment(cfg)
    cells=np.array([[1.45,.875],[5.5,3.5],[4.,.4]])
    Q=np.array([.16,.16,.02]);q=np.full(3,.74);p=np.array([1.45,.875])
    env=env._replace(cells=jnp.array(cells),targets=jnp.full(3,.7),process_noise=jnp.array(Q))
    x=jnp.r_[jnp.array(p),jnp.zeros(2)]
    end=m.advance(x,jnp.array(q),jnp.zeros(2),m.StillAir(),env,cfg,True,interval=.2)
    C=np.exp(-np.sum((cells-p)**2,axis=1)/(2*cfg.sensor_sigma**2))
    P0=1/q-1
    oracle=solve_ivp(lambda t,P:Q-C*C/cfg.sensor_R*P*P,(0,.2),P0,method='DOP853',atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(end[1],1/(1+oracle.y[:,-1]),atol=2e-8)
    dead=m.advance(x,jnp.array(q),jnp.zeros(2),m.StillAir(),env,cfg,False,interval=.2)
    np.testing.assert_allclose(dead[1],q/(1+.2*Q*q),atol=1e-12)


def test_swept_contact_is_not_an_endpoint_only_check():
    env,_=m.environment(m.Settings());center=np.array(env.obstacles[2,:2]);r=float(env.obstacles[2,2])
    a=center+[-r-.1,0];b=center+[r+.1,0]
    t=float(m.contact_fraction(jnp.array(a),jnp.array(b),env))
    np.testing.assert_allclose(a+t*(b-a),center+[-r,0],atol=1e-12)


def test_baseline_forecast_uses_neither_belief_nor_filter():
    cfg=replace(m.Settings(),horizon_seconds=.4,commit_seconds=.4,mission_seconds=.4)
    env,_=m.environment(cfg);env=env._replace(cells=env.cells[:8],targets=env.targets[:8],process_noise=env.process_noise[:8])
    theta=jnp.zeros(cfg.steps*2);x=jnp.array([1.45,.875,0.,0.]);q=jnp.full(8,.74)
    class Forbidden:
        def __getattr__(self,name):raise AssertionError('Baseline consulted wind belief: '+name)
    f=m.forecast(theta,x,q,Forbidden(),Forbidden(),m.StillAir(),env,cfg,False)
    np.testing.assert_allclose(f.states,np.tile(np.array(x),(cfg.steps+1,1)),atol=1e-12)
    assert float(f.max_slack)==0 and not bool(f.collided)


def test_inducing_relocation_accepts_readonly_float64_jax_arrays():
    params=init_parameters(jax.random.PRNGKey(8));points=jnp.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]],dtype=jnp.float64)
    z=select_inducing(params,points,jnp.array([.3,.8,1.3,1.8]),3)
    assert z.shape==(3,2) and len(np.unique(np.asarray(z),axis=0))==3


def test_same_dynamics_and_original_algorithms_are_composed():
    cfg=m.Settings()
    assert cfg.umax==m.physical.CFG.umax and cfg.safety_margin==m.physical.CFG.safety_margin
    assert cfg.backup_steps==m.physical.CFG.backup_steps and cfg.spline=='full'
    assert cfg.execution_dt==m.physical.DT and cfg.physics_dt==.01
    assert m.optimize.__module__=='voi_diffusion_ipp.planning.mbd'
    assert m.relaxed_rpcbf.__module__=='voi_diffusion_ipp.quarter.filter'
    assert m.filter_control.__module__=='voi_diffusion_ipp.safety.flow_policy_value'
    source=Path(m.__file__).read_text()
    assert '.condition(' not in source and 'cache.tabulate' not in source


def test_early_rejection_objective_matches_full_rollout():
    from voi_diffusion_ipp.wind.oracle import KnownWind,KnownBasis
    cfg=replace(m.Settings(),horizon_seconds=1.,commit_seconds=1.,mission_seconds=1.)
    env,_=m.environment(cfg);env=env._replace(cells=env.cells[:8],targets=env.targets[:8],process_noise=env.process_noise[:8])
    b=KnownWind(KnownBasis(m.StillAir()));inner=b.inner_fields(jnp.zeros((cfg.inner_samples,1,2)))
    outer=b.inner_fields(jnp.zeros((cfg.worlds,1,2)))
    q=jnp.full(8,.74)
    for aware in (False,True):
        for x in (jnp.array([1.45,.875,0.,0.]),jnp.array([.01,.5,-3.,0.])):
            theta=jnp.zeros(cfg.steps*2)
            original=m.forecast(theta,x,q,b,inner,m.StillAir(),env,cfg,aware)
            expected=jnp.where(original.collided,jnp.inf,original.cost)
            actual=m.forecast_cost(theta,x,q,b,inner,outer,env,cfg,aware)
            np.testing.assert_allclose(actual,expected,atol=1e-12)


def test_all_three_transfer_seeds_reach_a_in_full_still_air_plant():
    cfg=m.Settings();env,_=m.environment(cfg)
    x=jnp.array([1.45,.875,0.,0.]);q=jnp.full(len(env.cells),cfg.initial_q)
    seeds=m.seed_bank(x,q,env,cfg)
    small=env._replace(cells=env.cells[:1],targets=env.targets[:1],process_noise=env.process_noise[:1])
    evaluate=jax.jit(lambda theta:m.forecast(theta,x,q[:1],None,None,m.StillAir(),small,cfg,False))
    for index in (3,5,7):
        result=evaluate(seeds[index])
        assert not bool(result.collided)
        assert float(result.min_clearance)>=cfg.safety_margin
        assert np.linalg.norm(np.asarray(result.states[-1,:2])-[6.5,3.525])<.05


def test_online_graph_provides_distinct_passages_without_region_destinations():
    import monitoring_routes as routes
    cfg=replace(m.Settings(),horizon_seconds=30)
    env,_=m.environment(cfg);x=np.array([1.45,.875,0.,0.]);q=np.full(len(env.cells),cfg.initial_q)
    paths,info=routes.coarse_paths(x,q,env,cfg)
    assert len(paths)==6
    assert any(np.max(p[:,1])>4.3 for p in paths) # Upper passage found online.
    assert any(np.any((p[:,0]>4.6)&(p[:,1]<.3)) for p in paths) # Lower.
    for path in paths:
        assert np.all(routes.visible(path[:-1],path[1:],np.asarray(env.obstacles),np.asarray(env.bounds),cfg.safety_margin))
    # Move the demand to a new, arbitrary free-space location. The initializer
    # must discover it from the field, rather than keep named-region goals.
    target=np.exp(-np.sum((np.asarray(env.cells)-[3.2,.65])**2,axis=1)/.25)
    changed=env._replace(targets=jnp.asarray(target))
    _,other=routes.coarse_paths(x,np.zeros_like(q),changed,cfg)
    assert np.linalg.norm(np.array(other['initial_destinations'][0])-[3.2,.65])<.4


def test_tracked_forecast_keeps_full_wind_force_and_exact_cost():
    from voi_diffusion_ipp.wind.oracle import KnownWind,KnownBasis
    cfg=replace(m.Settings(),horizon_seconds=1.,commit_seconds=1.,mission_seconds=1.,reference_tracking=True)
    env,_=m.environment(cfg);env=env._replace(cells=env.cells[:8],targets=env.targets[:8],process_noise=env.process_noise[:8])
    b=KnownWind(KnownBasis(m.StillAir()));inner=b.inner_fields(jnp.zeros((cfg.inner_samples,1,2)))
    outer=b.inner_fields(jnp.zeros((cfg.worlds,1,2)));x=jnp.array([1.45,.875,0.,0.]);q=jnp.full(8,.74)
    theta=jnp.tile(jnp.array([.12,.08]),cfg.steps)
    for aware in (False,True):
        r=m.forecast(theta,x,q,b,inner,m.StillAir(),env,cfg,aware)
        cost=m.forecast_cost(theta,x,q,b,inner,outer,env,cfg,aware)
        np.testing.assert_allclose(cost,r.cost,atol=1e-12)
        assert not bool(r.collided)


def test_online_alternatives_keep_distinct_passages_from_central_approach():
    """A later MPC state used to lose both bypasses during string pulling."""
    import monitoring_routes as routes
    cfg=replace(m.Settings(),horizon_seconds=30)
    env,_=m.environment(cfg)
    x=np.array([3.8254,.9828,0.,0.]);q=np.full(len(env.cells),.22)
    paths,info=routes.coarse_paths(x,q,env,cfg)
    assert any(np.max(p[:,1])>4.3 for p in paths)
    assert any(np.any((p[:,0]>4.6)&(p[:,1]<.3)) for p in paths)
    for path,blocked in zip(paths,info['mode_blocked_cuts']):
        assert not routes.crosses_cuts(path[:-1],path[1:],np.asarray(blocked).reshape(-1,2,2)).any()
        assert routes.visible(path[:-1],path[1:],np.asarray(env.obstacles),np.asarray(env.bounds),cfg.safety_margin).all()


def test_absorbing_forecast_failure_charges_remaining_horizon():
    cfg=replace(m.Settings(),horizon_seconds=2.,commit_seconds=2.,mission_seconds=2.,reject_collisions=False,reference_tracking=True)
    env,_=m.environment(cfg);env=env._replace(cells=env.cells[:8],targets=env.targets[:8],process_noise=env.process_noise[:8])
    x=jnp.array([.01,.5,-3.,0.]);q=jnp.full(8,.74);theta=jnp.zeros(cfg.steps*2)
    r=m.forecast(theta,x,q,None,None,m.StillAir(),env,cfg,False)
    value=m.forecast_cost(theta,x,q,None,None,None,env,cfg,False)
    assert bool(r.collided) and np.isfinite(value) and value>0
    np.testing.assert_allclose(value,r.cost,atol=1e-12)
    np.testing.assert_array_equal(r.states[-1],r.states[-2])
    expected=r.clarity[-2]/(1+cfg.dt*env.process_noise*r.clarity[-2])
    np.testing.assert_allclose(r.clarity[-1],expected,atol=1e-12)


def test_absorbing_cost_reordering_matches_full_windy_filtered_forecasts():
    from typing import NamedTuple
    from voi_diffusion_ipp.wind.oracle import KnownWind,KnownBasis
    class UniformWind(NamedTuple):
        velocity:object
        def at(self,p):return jnp.broadcast_to(self.velocity,jnp.atleast_2d(p).shape)
    cfg=replace(m.Settings(),horizon_seconds=.6,commit_seconds=.6,mission_seconds=.6,reference_tracking=True,reject_collisions=False)
    env,_=m.environment(cfg);env=env._replace(cells=env.cells[:12],targets=env.targets[:12],process_noise=env.process_noise[:12])
    wind=UniformWind(jnp.array([-.8,.3]));b=KnownWind(KnownBasis(wind))
    inner=b.inner_fields(jnp.zeros((cfg.inner_samples,1,2)));outer=b.inner_fields(jnp.zeros((cfg.worlds,1,2)))
    q=jnp.full(12,.74);theta=jnp.tile(jnp.array([.1,.04]),cfg.steps)
    evaluate=jax.jit(lambda x:m.forecast_cost(theta,x,q,b,inner,outer,env,cfg,True))
    for x in (jnp.array([1.45,.875,.2,.1]),jnp.array([.01,.5,-3.,0.])):
        full=m.forecast(theta,x,q,b,inner,wind,env,cfg,True)
        np.testing.assert_allclose(evaluate(x),full.cost,atol=1e-12)


def test_old_safety_costs_match_full_rollout_and_keep_monitoring_separate():
    cfg=replace(m.Settings(),horizon_seconds=.6,commit_seconds=.6,mission_seconds=.6,
                reference_tracking=True,reject_collisions=False,legacy_safety_costs=True)
    env,_=m.environment(cfg);env=env._replace(cells=env.cells[:8],targets=env.targets[:8],process_noise=env.process_noise[:8])
    q=jnp.full(8,.74);theta=jnp.zeros(cfg.steps*2)
    for x in (jnp.array([.25,.5,0.,0.]),jnp.array([.01,.5,-3.,0.])):
        full=m.forecast(theta,x,q,None,None,m.StillAir(),env,cfg,False)
        value=m.forecast_cost(theta,x,q,None,None,None,env,cfg,False)
        np.testing.assert_allclose(value,full.cost,atol=1e-12)
        np.testing.assert_allclose(full.cost,full.monitoring_cost+full.safety_cost,atol=1e-12)
        assert full.safety_cost>0
        legacy=m.forecast(theta,x,q,None,None,m.StillAir(),env,replace(cfg,legacy_safety_costs=False),False)
        np.testing.assert_allclose(full.states,legacy.states,atol=1e-12)
        np.testing.assert_allclose(full.monitoring_cost,legacy.cost,atol=1e-12)
