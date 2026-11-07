import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import full_learning_monitoring as model
from reachable_learning import ReachableLearning,observation_gain,observation_gain_and_variance,acquisition_variances,braking_feasible,measured_route_mask
from guarded_waypoints import GuardedWaypointPolicy
from cached_poam_fields import CachedFields
from voi_diffusion_ipp.wind.poam import POAM


@pytest.mark.parametrize('conditioned',[False,True])
def test_off_grid_waypoint_gain_matches_full_joint_gaussian_conditioning(conditioned):
    nodes=jnp.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    b=POAM(nodes,jax.random.PRNGKey(3),prior_support=nodes,noise_std=.1).planning_belief()
    b=jax.tree.map(lambda z:z.astype(jnp.float64) if hasattr(z,'dtype') and jnp.issubdtype(z.dtype,jnp.floating) else z,b)
    if conditioned:
        b=b.condition(jnp.array([[.2,.4],[.8,.1],[.2,.4]]),jnp.array([[1.,-.5],[.2,.8],[1.1,-.4]]))
    cache=CachedFields(nodes,(2,2,2),model.settings(),nodes);cache.map(b)
    query=jnp.array([[.23,.42],[.75,.12]]);weights=jnp.array([.1,.3,.2,.4])
    gain=observation_gain(b,nodes,cache.Kmap,cache.posterior[1],weights,query)
    _,joint=b.joint(jnp.concatenate((nodes,query)))
    expected=[]
    for i in (4,5):
        posterior=joint[:4,:4]-jnp.outer(joint[:4,i],joint[:4,i])/(joint[i,i]+.01)
        expected.append(weights@(jnp.diag(joint[:4,:4])-jnp.diag(posterior)))
    np.testing.assert_allclose(gain,expected,rtol=1e-8,atol=1e-9)
    returned_gain,variance=observation_gain_and_variance(b,nodes,cache.Kmap,cache.posterior[1],weights,query)
    np.testing.assert_allclose(returned_gain,gain,rtol=1e-12)
    np.testing.assert_allclose(variance,jnp.diag(joint)[4:],rtol=1e-8,atol=1e-9)


def test_direct_query_retires_learned_goal_without_changing_safety_field():
    from types import SimpleNamespace
    from uncertainty_waypoints import WaypointPolicy
    env=SimpleNamespace(bounds=np.array([[0.,0.],[2.,2.]]),obstacles=np.array([[1.8,1.8,.1]]))
    policy=WaypointPolicy(env,spacing=.05,learned_variance=.02,require_target_precision=True)
    x=np.array([.5,.5,0.,0.]);grid_variance=np.full(len(policy.nodes),.026)
    _,first=policy.plan(x,grid_variance,0.)
    target=policy.target;goal=policy.nodes[target];at_goal=np.r_[goal,[0.,0.]]
    _,old=policy.plan(at_goal,grid_variance,1.)
    assert policy.target==target and old['reason']=='persistent target'
    # Recorded failure: direct GP variance was below .02, whereas the
    # interpolated field used for completion still exceeded that threshold.
    direct=acquisition_variances(grid_variance,[target],[.1248**2])
    _,corrected=policy.plan(at_goal,direct,1.2)
    assert policy.target!=target and corrected['reason'] in ('arrived','target learned')
    assert direct[target]<.02 and np.all(grid_variance==.026)
    np.testing.assert_array_equal(np.delete(direct,target),np.delete(grid_variance,target))


def test_guarded_waypoints_keep_new_target_in_the_admissible_component():
    from types import SimpleNamespace
    env=SimpleNamespace(bounds=np.array([[0.,0.],[2.,2.]]),obstacles=np.array([[1.8,1.8,.1]]))
    policy=GuardedWaypointPolicy(env,spacing=.05,minimum_target_distance=.05,reference_lookahead=.3)
    local=(policy.nodes[:,0]<.8)&(policy.nodes[:,1]<.8)
    island=(policy.nodes[:,0]>1.2)&(policy.nodes[:,1]>1.2)
    variance=np.ones(len(policy.nodes));priority=variance.copy();priority[island]=100.
    path,decision=policy.plan(np.array([.5,.5,0.,0.]),variance,0.,np.zeros((len(variance),2)),np.zeros(2),.05,
        priority=priority,admissible_nodes=local|island)
    assert len(path)>1 and np.all(path<.8)
    ref=policy.reference(np.array([.5,.5,0.,0.]),path)
    assert np.linalg.norm(ref[2:])<=policy.speed+1e-12


def test_control_feasibility_accepts_corrective_acceleration_near_obstacle():
    from types import SimpleNamespace
    class ConstantBasis:
        def evaluate(self,p,values):return jnp.atleast_2d(values)
    class ConstantMean:
        def mean_at(self,p):return jnp.zeros((1,2))
    env=SimpleNamespace(bounds=jnp.array([[-4.,-4.],[4.,4.]]),obstacles=jnp.array([[0.,0.,.7]]))
    cfg=SimpleNamespace(safety_margin=.06,umax=1.6)
    p=jnp.array([.78,0.]);sigma=.2
    error=(.1/.680)*9*(sigma*sigma+.01)
    assert error>1.5**2*(.78-.7-.06) # Earlier zero-correction test rejects it.
    assert bool(braking_feasible(p,ConstantMean(),jnp.array([[sigma]]),ConstantBasis(),env,cfg))
    assert not bool(braking_feasible(p,ConstantMean(),jnp.array([[4.]]),ConstantBasis(),env,cfg))


def test_route_bound_matches_recovery_and_never_invents_remote_measurements():
    from guarded_waypoints import holding_force_bound
    nodes=np.array([[0.,0.],[.1,0.],[1.,1.]])
    mean=np.array([[0.,0.],[3.,-3.],[3.,-3.]]);sigma=np.array([.05,.1,1.8])
    valid,bound,observed=measured_route_mask(nodes,mean,sigma,[[0.,0.]],1.6)
    np.testing.assert_allclose(bound,[holding_force_bound(m,s) for m,s in zip(mean,sigma)],rtol=1e-12)
    np.testing.assert_array_equal(observed,[True,True,False])
    np.testing.assert_array_equal(valid,[True,False,True])


def test_route_precision_includes_unobserved_remaining_path_and_deduplicates_visits():
    from types import SimpleNamespace
    learner=ReachableLearning.__new__(ReachableLearning)
    xx,yy=np.meshgrid(np.arange(60)*.05+.025,np.arange(20)*.05+.025)
    nodes=np.c_[xx.ravel(),yy.ravel()]
    learner.policy=SimpleNamespace(bounds=np.array([[0.,0.],[3.,1.]]),spacing=.05,nodes=nodes,
        path=np.array([[.25,.5],[2.75,.5]]),target=650)
    learner.track_route_precision=True;learner.vector_precision=.2
    learner.route_visited=np.zeros(len(nodes),bool);learner.last_route_position=None
    learner.route_handoff_ids=None;learner.route_current_ids=None;learner.route_goal=None
    sigma=np.where(nodes[:,0]<.5,.1,1.8);x=np.array([.25,.5,0.,0.])
    learner.update_route_precision(x,sigma);ids=learner.route_current_ids.copy()
    assert learner.route_handoff_ids is None
    for _ in range(20):learner.update_route_precision(x,sigma)
    np.testing.assert_array_equal(learner.route_current_ids,ids)
    assert learner.route_handoff_ids is None
    learner.update_route_precision(x,np.full(len(nodes),.1))
    np.testing.assert_array_equal(learner.route_handoff_ids,ids)
