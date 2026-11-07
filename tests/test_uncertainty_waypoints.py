"""Geometry and uncertainty response of the standalone executed survey policy."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
from uncertainty_waypoints import WaypointPolicy
from monitoring_routes import visible


def environment():
    return SimpleNamespace(bounds=np.array([[0.,0.],[4.,3.]]),obstacles=np.array([[2.,1.5,.65]]))


def test_uncertainty_drives_selection_and_path_is_clear():
    policy=WaypointPolicy(environment());x=np.array([.5,1.5,0.,0.])
    variance=np.full(len(policy.nodes),.01)
    east=(policy.nodes[:,0]>3)&(np.abs(policy.nodes[:,1]-1.5)<.2)
    variance[east]=3.24
    path,decision=policy.plan(x,variance,0.)
    assert decision['target'][0]>3
    assert np.all(visible(path[:-1],path[1:],policy.ob,policy.bounds,policy.margin))
    assert len(path)>2 # The direct chord crosses the circular obstacle.
    variance[east]=.01
    variance[(policy.nodes[:,0]<1)&(policy.nodes[:,1]>2.4)]=3.24
    _,changed=policy.plan(x,variance,2.)
    assert changed['target'][0]<1 and changed['target'][1]>2.4


def test_stalled_target_is_not_permanently_excluded_or_falsely_observed():
    policy=WaypointPolicy(environment());x=np.array([.5,1.5,0.,0.])
    variance=np.ones(len(policy.nodes))*3.24;before=variance.copy()
    _,first=policy.plan(x,variance,0.)
    _,second=policy.plan(x,variance,16.)
    assert first['target']!=second['target']
    assert np.any(policy.cooldown==46.)
    np.testing.assert_array_equal(variance,before)


def test_reference_is_bounded_and_does_not_change_state():
    policy=WaypointPolicy(environment());x=np.array([.5,.5,.2,.1]);before=x.copy()
    ref=policy.reference(x,np.array([[.5,.5],[1.5,.5]]))
    assert np.linalg.norm(ref[2:])<=.4
    np.testing.assert_array_equal(x,before)
    np.testing.assert_array_equal(ref[:2],x[:2])


def test_long_recovery_segment_interpolates_a_forward_lookahead():
    policy=WaypointPolicy(environment(),reference_lookahead=.3)
    x=np.array([1.2,.5,0.,0.]);path=np.array([[1.2,.5],[.4,.5]])
    ref=policy.reference(x,path)
    np.testing.assert_allclose(ref[2:],[-.3,0.],atol=1e-12)
    np.testing.assert_array_equal(x,[1.2,.5,0.,0.])


def test_precision_required_does_not_confuse_nearby_with_learned():
    policy=WaypointPolicy(environment(),learned_variance=.02,require_target_precision=True)
    variance=np.ones(len(policy.nodes));x=np.array([.5,.5,0.,0.])
    policy.plan(x,variance,0.)
    target=policy.target
    x[:2]=policy.nodes[target]+np.array([.06,0.])
    _,decision=policy.plan(x,variance,2.)
    assert policy.target==target and decision['reason']=='persistent target'
    variance[target]=.019
    policy.plan(x,variance,3.)
    assert policy.target!=target


def test_stall_detected_after_earlier_progress():
    policy=WaypointPolicy(environment())
    variance=np.ones(len(policy.nodes));x=np.array([.5,1.5,0.,0.])
    east=policy.nodes[:,0]>3.5;priority=np.where(east,1.,0.)
    path,_=policy.plan(x,variance,0.,priority=priority)
    target=policy.target
    # The robot makes substantial early progress but then stops indefinitely.
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    x[:2]=[np.interp(1.,arc,path[:,k]) for k in (0,1)]
    policy.plan(x,variance,15.,priority=priority)
    assert policy.target==target
    policy.plan(x,variance,31.,priority=priority)
    assert policy.target!=target
    assert policy.cooldown[target]==61.


def test_stalled_route_tries_other_passage_without_changing_information_goal():
    policy=WaypointPolicy(environment(),diversify_stalls=True)
    x=np.array([.5,1.4,0.,0.]);variance=np.ones(len(policy.nodes))
    priority=np.zeros(len(policy.nodes));goal=np.argmin(np.linalg.norm(policy.nodes-[3.5,1.4],axis=1));priority[goal]=1.
    first,_=policy.plan(x,variance,0.,priority=priority)
    second,decision=policy.plan(x,variance,16.,priority=priority)
    assert policy.target==goal
    assert decision['reason']=='stalled passage: alternate geometric route'
    assert policy.blocked_cuts.any()
    assert not np.array_equal(first,second)
    assert np.all(visible(second[:-1],second[1:],policy.ob,policy.bounds,policy.margin))
    # Diversification is a reference-search device, not fictitious sensing.
    np.testing.assert_array_equal(variance,np.ones(len(policy.nodes)))


def test_learned_wind_rerouting_does_not_falsely_abandon_a_longer_bypass():
    policy=WaypointPolicy(environment(),diversify_stalls=True)
    x=np.array([.5,1.4,0.,0.]);variance=np.ones(len(policy.nodes))
    priority=np.zeros(len(policy.nodes));goal=np.argmin(np.linalg.norm(policy.nodes-[3.5,1.4],axis=1));priority[goal]=1.
    calm=np.zeros((len(policy.nodes),2));first,_=policy.plan(x,variance,0.,calm,priority)
    assert first[:,1].min()<1.
    # Actual measurements reveal a wind barrier in the previously short gap.
    mean=calm.copy();barrier=(abs(policy.nodes[:,0]-2.)<.35)&(policy.nodes[:,1]<1.5)
    mean[barrier]=[6.,0.]
    second,decision=policy.plan(x,variance,16.,mean,priority)
    assert policy.target==goal and policy.cooldown[goal]==0
    assert second[:,1].max()>2.25
    assert decision['reason']=='route changed: restart progress window'
    assert policy.progress_checked_at==16.
    # With no subsequent movement or route change, ordinary stall detection
    # still works. The fix does not disable progress checks indefinitely.
    policy.plan(x,variance,32.,mean,priority)
    assert policy.target!=goal and policy.cooldown[goal]==62.


def test_original_poam_update_and_training_are_finite_with_full_residual():
    import jax
    import jax.numpy as jnp
    from voi_diffusion_ipp.wind.poam import POAM
    from observed_poam import _double
    from waypoint_poam import condition,train
    with jax.experimental.enable_x64():
        z=jnp.array([[.2,.2],[.8,.2],[.2,.8],[.8,.8]])
        b=_double(POAM(z,jax.random.PRNGKey(731),prior_std=1.8,spectral_count=8,prior_support=z).planning_belief())
        y=jnp.array([[1.,-.2],[-.3,.4]])
        updated,_=condition(b,z[:2],y,jnp.zeros(4,bool))
        # Fixed-point-set update agrees with the corresponding full GP posterior.
        K=np.asarray(b.kernel(z,z));Kf=K[:,:2]
        expected=Kf@np.linalg.solve(K[:2,:2]+.01*np.eye(2),np.asarray(y))
        np.testing.assert_allclose(updated.mean_at(z),expected,atol=1e-5)
        zero=jax.tree.map(jnp.zeros_like,b.basis.parameters)
        trained,_,_,step,loss=train(updated,z[:2],y,jnp.array(2.,dtype=jnp.float64),zero,zero,jnp.array(0))
        assert int(step)==4 and np.isfinite(np.asarray(loss)).all()
        query=jnp.array([[.5,.5],[2.,2.]])
        mu,cov=trained.predict(query)
        assert np.isfinite(mu).all() and np.isfinite(cov).all()
        assert float(cov[1,0,0])>0 # Unobserved-space residual has not been removed.
        assert np.linalg.eigvalsh(np.asarray(trained.covariance)).min()>-1e-8
        assert not np.allclose(trained.basis.parameters['w1'],b.basis.parameters['w1'])
        # Training also refreshes the support kernel cache used by field sampling.
        KS=np.asarray(trained.kernel(z,z))
        np.testing.assert_allclose(np.asarray(trained.basis.support_cholesky)@np.asarray(trained.basis.support_cholesky).T,KS+1e-4*np.eye(4),atol=1e-7)
