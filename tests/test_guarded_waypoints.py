import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
from types import SimpleNamespace
import numpy as np
from guarded_waypoints import GuardedWaypointPolicy,holding_force_bound


def test_route_diversification_preserves_goal_and_progress_clock_across_recovery():
    from types import SimpleNamespace
    env=SimpleNamespace(bounds=np.array([[0.,0.],[4.,3.]]),obstacles=np.array([[2.,1.5,.65]]))
    policy=GuardedWaypointPolicy(env,diversify_stalls=True)
    var=np.ones(len(policy.nodes));mu=np.zeros((len(var),2));priority=np.zeros(len(var))
    goal=np.argmin(np.linalg.norm(policy.nodes-[3.5,1.4],axis=1));priority[goal]=1.
    policy.plan(np.array([.5,.5,0.,0.]),var,0.,mu,np.zeros(2),.05,priority=priority)
    x=np.array([.6,1.3,0.,0.])
    _,decision=policy.plan(x,var,5.,mu,np.array([3.,0.]),.05,priority=priority)
    assert decision['reason'].startswith('retreat') and policy.target==goal
    policy.plan(x,var,6.,mu,np.zeros(2),.05,priority=priority)
    assert policy.retreat is None and policy.target==goal
    assert policy.selected_at==0. and policy.progress_checked_at==0.

def test_holding_bound_encloses_quadratic_wind_force():
    mu=np.array([1.2,-1.7]);sigma=.25;R=3*np.sqrt(sigma*sigma+.01);b=holding_force_bound(mu,sigma)
    angles=np.linspace(0,2*np.pi,1000)
    w=mu+R*np.c_[np.cos(angles),np.sin(angles)];d=(.1/.680)*np.linalg.norm(w,axis=1)[:,None]*w
    assert np.max(np.abs(d))<=b

def test_recovery_returns_toward_actual_low_wind_history():
    env=SimpleNamespace(obstacles=np.array([[3.,3.,.1]]),bounds=np.array([[0.,0.],[4.,4.]]))
    policy=GuardedWaypointPolicy(env);variance=np.ones(len(policy.nodes));means=np.zeros((len(policy.nodes),2))
    for i in range(7):policy.plan(np.array([1+i*.1,1,0,0]),variance,i*.2,means,np.zeros(2),.05)
    path,decision=policy.plan(np.array([1.7,1.,.1,0]),variance,1.4,means,np.array([3.,0]),.05)
    assert decision['reason'].startswith('retreat:')
    assert path[1,0]<1.7
    assert path[-1,0]<=1.2
    assert policy.target is None


def test_recovery_does_not_retrace_a_loop_toward_rejected_area():
    env=SimpleNamespace(obstacles=np.array([[3.,3.,.1]]),bounds=np.array([[0.,0.],[4.,4.]]))
    policy=GuardedWaypointPolicy(env)
    policy.retreat=np.array([[.75,.35],[.50,.325],[.854,.31],[.317,.33],[.317,.634]])
    variance=np.ones(len(policy.nodes));means=np.zeros((len(policy.nodes),2))
    path,decision=policy.plan(np.array([.56,.334,0,0]),variance,20.,means,np.array([2.,0]),.05)
    assert decision['reason'].startswith('retreat:')
    assert path[1,0]<.56


def test_recovery_finishes_after_wind_carries_robot_to_another_safe_place():
    env=SimpleNamespace(obstacles=np.array([[2.,2.,.7]]),bounds=np.array([[0.,0.],[4.,4.]]))
    policy=GuardedWaypointPolicy(env);policy.retreat=np.array([[2.,1.],[2.,3.]])
    policy.trigger=np.array([3.,3.])
    variance=np.ones(len(policy.nodes));means=np.zeros((len(policy.nodes),2))
    path,decision=policy.plan(np.array([2.,1.,0.,0.]),variance,20.,means,np.zeros(2),.05)
    assert not decision['reason'].startswith('retreat:')
    assert policy.retreat is None


def test_recovery_never_uses_blocked_old_goal_as_a_fallback_segment():
    from monitoring_routes import visible
    env=SimpleNamespace(obstacles=np.array([[2.,2.,.7]]),bounds=np.array([[0.,0.],[4.,4.]]))
    policy=GuardedWaypointPolicy(env);policy.retreat=np.array([[2.,1.],[2.,3.]])
    policy.trigger=np.array([3.,3.]);variance=np.ones(len(policy.nodes));means=np.zeros((len(policy.nodes),2))
    path,decision=policy.plan(np.array([2.,1.,0.,0.]),variance,20.,means,np.array([4.,0.]),.05)
    assert decision['reason'].startswith('retreat:')
    assert np.all(visible(path[:-1],path[1:],policy.ob,policy.bounds,policy.margin))


def test_recovery_does_not_hold_short_of_nearby_safe_goal():
    env=SimpleNamespace(obstacles=np.array([[3.,3.,.1]]),bounds=np.array([[0.,0.],[4.,4.]]))
    policy=GuardedWaypointPolicy(env);policy.retreat=np.array([[.44,2.86],[.44,2.76]])
    policy.trigger=np.array([.44,3.4]);variance=np.ones(len(policy.nodes));means=np.zeros((len(policy.nodes),2))
    path,decision=policy.plan(np.array([.44,2.86,0.,0.]),variance,20.,means,np.array([2.,0.]),.05)
    assert decision['reason'].startswith('retreat:')
    assert path[1,1]<2.86


def test_low_force_ends_recovery_even_close_to_trigger():
    env=SimpleNamespace(obstacles=np.array([[3.,3.,.1]]),bounds=np.array([[0.,0.],[4.,4.]]))
    policy=GuardedWaypointPolicy(env);policy.retreat=np.array([[2.,1.],[1.5,1.]])
    policy.trigger=np.array([2.,1.]);variance=np.ones(len(policy.nodes));means=np.zeros((len(policy.nodes),2))
    path,decision=policy.plan(np.array([2.,.8,0.,0.]),variance,20.,means,np.zeros(2),.05)
    assert policy.retreat is None
    assert not decision['reason'].startswith('retreat:')


def test_wind_displacement_outside_history_tube_can_rejoin_instead_of_holding():
    from monitoring_routes import visible
    env=SimpleNamespace(obstacles=np.array([[6.,5.,.4]]),bounds=np.array([[0.,0.],[8.,6.]]))
    policy=GuardedWaypointPolicy(env)
    policy.retreat=np.array([[3.38,1.48],[3.32,1.48]])
    # Actual failed pilot state was 0.47 m from its remaining safe history.
    x=np.array([3.728,1.254,0.,0.]);variance=np.ones(len(policy.nodes));means=np.zeros((len(policy.nodes),2))
    path,decision=policy.plan(x,variance,70.,means,np.array([-2.2,-2.6]),.01)
    assert decision['rejoining_measured_path']
    assert np.linalg.norm(path[-1]-x[:2])>.3
    assert np.linalg.norm(path[-1]-policy.retreat,axis=1).min()<1e-12
    assert np.all(visible(path[:-1],path[1:],policy.ob,policy.bounds,policy.margin))


def test_rejoin_does_not_connect_through_an_obstacle():
    from monitoring_routes import visible
    env=SimpleNamespace(obstacles=np.array([[2.,2.,.6]]),bounds=np.array([[0.,0.],[4.,4.]]))
    policy=GuardedWaypointPolicy(env);policy.retreat=np.array([[2.,2.8],[2.,3.2]])
    x=np.array([2.,1.,0.,0.]);variance=np.ones(len(policy.nodes));means=np.zeros((len(policy.nodes),2))
    path,decision=policy.plan(x,variance,20.,means,np.array([3.,0.]),.01)
    np.testing.assert_allclose(path[-1],x[:2])
    assert np.all(visible(path[:-1],path[1:],policy.ob,policy.bounds,policy.margin))


def test_stalled_recovery_can_go_around_wind_instead_of_requesting_impossible_upwind_motion():
    from monitoring_routes import visible
    env=SimpleNamespace(obstacles=np.array([[3.8,3.8,.1]]),bounds=np.array([[0.,0.],[4.,4.]]))
    policy=GuardedWaypointPolicy(env,spacing=.05,reference_lookahead=.3)
    wind=np.zeros((len(policy.nodes),2))
    jet=(policy.nodes[:,0]>1.5)&(policy.nodes[:,0]<2.5)&(policy.nodes[:,1]<2.5)
    wind[jet,1]=-3.2
    policy.retreat=np.array([[2.,1.],[2.,2.]])
    x=np.array([2.,1.,0.,0.]);variance=np.ones(len(policy.nodes))
    policy.plan(x,variance,0.,wind,np.array([0.,-3.2]),.01)
    path,decision=policy.plan(x,variance,15.,wind,np.array([0.,-3.2]),.01)
    assert decision['recovery_graph'] and len(path)>4
    assert np.all(visible(path[:-1],path[1:],policy.ob,policy.bounds,policy.margin))
    velocity=policy.reference(x,path)[2:]
    # Moving straight north at .4 m/s requires 1.906 m/s² > 1.6.
    assert .1/.680*3.6**2>policy.umax
    force=.1/.680*np.linalg.norm(np.array([0.,-3.2])-velocity)*(np.array([0.,-3.2])-velocity)
    assert abs(velocity[0])>.2 and np.max(np.abs(force))<=policy.umax+1e-12
    np.testing.assert_allclose(path[-1],[2.,2.])


def test_recovery_connector_checks_actual_local_wind_not_a_nearby_optimistic_grid_node():
    env=SimpleNamespace(obstacles=np.array([[7.,5.,.2]]),bounds=np.array([[0.,0.],[8.,6.]]))
    policy=GuardedWaypointPolicy(env,spacing=.05,reference_lookahead=.3)
    x=np.array([3.94,1.367,0.,0.]);mu=np.array([-1.447,-3.13])
    path=policy.recovery_geometric_path(x[:2],[4.094,1.697],np.zeros((len(policy.nodes),2)),mu)
    assert path is not None
    velocity=policy.reference(x,path)[2:]
    force=.1/.680*np.linalg.norm(mu-velocity)*(mu-velocity)
    assert np.max(np.abs(force))<=policy.umax+1e-12
    assert np.linalg.norm(path[1]-x[:2])>=policy.reference_lookahead


def test_stalled_recovery_can_select_a_different_actually_measured_refuge():
    env=SimpleNamespace(obstacles=np.array([[3.8,3.8,.1]]),bounds=np.array([[0.,0.],[4.,4.]]))
    policy=GuardedWaypointPolicy(env,spacing=.05,reference_lookahead=.3)
    policy.retreat=np.array([[2.,1.],[2.,2.]])
    policy.history=[(np.array([.7,1.]),.1)]
    policy.recovery_anchor=np.array([2.,1.]);policy.recovery_since=0.
    x=np.array([2.,1.,0.,0.]);means=np.zeros((len(policy.nodes),2))
    path,decision=policy.plan(x,np.ones(len(policy.nodes)),15.,means,np.array([0.,-3.2]),.01)
    np.testing.assert_allclose(path[-1],[.7,1.])
    assert decision['recovery_graph']
