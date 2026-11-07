import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import numpy as np
import jax.numpy as jnp
from types import SimpleNamespace
from wind_model_trust import acceleration_error_bound,trajectory_error_bound,authority_margin


def test_nonlinear_error_encloses_the_confidence_disk_without_changing_drag():
    rng=np.random.default_rng(908)
    for _ in range(20):
        mean=rng.normal(size=2)*3;v=rng.normal(size=2);sigma=rng.uniform(.01,2)
        R=3*np.sqrt(sigma**2+.01);angles=np.linspace(0,2*np.pi,101)
        points=mean+R*np.c_[np.cos(angles),np.sin(angles)]
        drag=lambda w:(.1/.680)*np.linalg.norm(w-v,axis=-1,keepdims=True)*(w-v)
        error=np.linalg.norm(drag(points)-drag(mean),axis=-1)
        assert error.max()<=float(acceleration_error_bound(jnp.asarray(mean),sigma,jnp.asarray(v)))+1e-6


def test_trust_evaluates_future_positions_and_keeps_unknown_variance():
    class Basis:
        def evaluate(self,positions,values):
            if values=='mean':return jnp.zeros((len(positions),2))
            return jnp.where(positions[:,0:1]>.5,1.8,.05)
    tables=SimpleNamespace(mean='mean',std='std')
    states=jnp.array([[[0.,0.,0.,0.],[1.,0.,0.,0.]]])
    result=np.asarray(trajectory_error_bound(states,tables,Basis()))
    assert result.shape==(1,2)
    assert result[0,0]<.4 and result[0,1]>.4


def test_error_cap_alone_does_not_establish_actuator_headroom():
    np.testing.assert_allclose(authority_margin(jnp.array([.2,.2]),jnp.array([[1.5,0.],[1.,-.5]]),1.6),[-.1,.4],atol=1e-6)


def test_finalist_selection_keeps_mbd_cost_but_rejects_unknown_route(monkeypatch):
    import jax
    from dataclasses import replace
    from typing import NamedTuple
    import rmse_monitoring as rm
    class Result(NamedTuple):
        final:object
        finalists:object
        finalist_field_costs:object
        initial_parameters:object
        initial_field_costs:object
    result=Result(jnp.array([2.]),jnp.array([[2.],[7.]]),jnp.array([[.1,.1],[1.,1.]]),
        jnp.array([[0.],[7.]]),jnp.array([[5.,5.],[1.,1.]]))
    monkeypatch.setattr(rm,'search',lambda *a,**kw:result)
    def reference(theta,*args):
        return jnp.array([[theta[0],0.,0.,0.],[theta[0],0.,0.,0.]]),jnp.zeros((1,2))
    monkeypatch.setattr(rm.old.legacy.old,'nominal_reference',reference)
    def forecast(theta,*args,**kwargs):
        states=reference(theta)[0]
        return rm.legacy.old.Rollout(states,jnp.zeros((2,1)),jnp.zeros((1,2)),states,
            theta[0],jnp.array(0.),jnp.array(0.),jnp.array(1.),jnp.array(False))
    monkeypatch.setattr(rm.core,'execution_forecast',forecast)
    class Basis:
        def evaluate(self,p,values):
            if values.shape[-1]==2:return jnp.zeros((len(p),2))
            return jnp.where(p[:,0:1]==7,.03,1.8)
    tables=rm.Tables(jnp.zeros((1,2)),jnp.zeros((1,1)),jnp.zeros(1),jnp.zeros(2),jnp.zeros(2),jnp.zeros(2))
    planner=rm.optimizer(None,replace(rm.settings(),mode_count=2),Basis(),reuse_training=True,wind_error_reserve=.4)
    theta,_,_,_,audit=planner(jnp.zeros(4),jnp.zeros(1),tables,jnp.array([[0.],[7.]]),jnp.array([0.]),jax.random.PRNGKey(4))
    assert float(theta[0])==7
    assert not bool(audit['no_feasible'])
    assert float(audit['forecast_error_bounds'][audit['selected']])<.4
    assert not bool(audit['request_targeted_learning']) # accepted forecast moves to a new footprint
    _,_,_,_,stationary=planner(jnp.array([7.,0.,0.,0.]),jnp.zeros(1),tables,jnp.array([[0.],[7.]]),jnp.array([0.]),jax.random.PRNGKey(4))
    assert bool(stationary['request_targeted_learning'])
    assert abs(stationary['uncertainty_blocked_improvement']-.9)<1e-6
    np.testing.assert_allclose(stationary['learning_reference'][:,0],2.)
    free=SimpleNamespace(clearance=lambda x:jnp.array(1.))
    revised=rm.optimizer(free,replace(rm.settings(),mode_count=2),Basis(),reuse_training=True,
        wind_error_reserve=.4,information_trigger='lower-cost',prepare_validation=True)
    rt,_,_,_,moving=revised(jnp.zeros(4),jnp.zeros(1),tables,jnp.array([[0.],[7.]]),jnp.array([0.]),jax.random.PRNGKey(4))
    np.testing.assert_array_equal(rt,theta) # Same exploitation decision and cost.
    assert bool(moving['request_targeted_learning']) # Movement alone is not enough.
    blocked=SimpleNamespace(clearance=lambda x:jnp.where(x[0]==2,-1.,1.))
    invalid=rm.optimizer(blocked,replace(rm.settings(),mode_count=2),Basis(),reuse_training=True,
        wind_error_reserve=.4,information_trigger='lower-cost')
    _,_,_,_,outside=invalid(jnp.zeros(4),jnp.zeros(1),tables,jnp.array([[0.],[7.]]),jnp.array([0.]),jax.random.PRNGKey(4))
    assert not bool(outside['request_targeted_learning'])
    class KnownBasis(Basis):
        def evaluate(self,p,values):
            return jnp.zeros((len(p),2)) if values.shape[-1]==2 else jnp.full((len(p),1),.03)
    known=rm.optimizer(free,replace(rm.settings(),mode_count=2),KnownBasis(),reuse_training=True,
        wind_error_reserve=.4,information_trigger='lower-cost')
    _,_,_,_,accurate=known(jnp.zeros(4),jnp.zeros(1),tables,jnp.array([[0.],[7.]]),jnp.array([0.]),jax.random.PRNGKey(4))
    assert not bool(accurate['request_targeted_learning'])
    class UnknownBasis(Basis):
        def evaluate(self,p,values):
            return jnp.zeros((len(p),2)) if values.shape[-1]==2 else jnp.full((len(p),1),1.8)
    unknown=rm.optimizer(free,replace(rm.settings(),mode_count=2),UnknownBasis(),reuse_training=True,
        wind_error_reserve=.4,information_trigger='lower-cost',prepare_validation=True)
    _,_,_,_,fallback=unknown(jnp.zeros(4),jnp.zeros(1),tables,jnp.array([[0.],[7.]]),jnp.array([0.]),jax.random.PRNGKey(4))
    assert bool(fallback['no_feasible']) and len(fallback['costs'])==3


def test_reference_weights_match_quadratic_drag_jacobian_and_vanish_without_coupling():
    import jax
    from wind_model_trust import reference_information_weights
    states=jnp.array([[0.,0.,.2,.1],[1.,0.,-.1,.2]])
    mean=jnp.array([[1.,.4],[.5,.8]]);variance=jnp.array([.1,.3]);k=.1/.680
    cells=jnp.array([[0.,0.],[1.,0.]]);need=jnp.array([.4,.8]);sensor=.6
    expected=[]
    for i in range(2):
        offsets=jnp.sqrt(2*variance[i])*jnp.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        derivative=jax.jacfwd(lambda w:k*jnp.linalg.norm(w-states[i,2:])*(w-states[i,2:]))
        sensitivity=jnp.mean(jax.vmap(lambda w:jnp.sum(derivative(w)**2))(mean[i]+offsets))
        sensing=jnp.exp(-jnp.sum((states[i,:2]-cells)**2,axis=-1)/sensor**2)@need
        expected.append(sensitivity*sensing)
    expected=jnp.asarray(expected);expected/=expected.sum()
    np.testing.assert_allclose(reference_information_weights(states,mean,variance,need,cells,sensor),expected,rtol=1e-6)
    np.testing.assert_array_equal(reference_information_weights(states,mean,variance,need,cells,sensor,drag=0.),jnp.zeros(2))


def test_downstream_value_includes_transit_before_sensing_but_not_irrelevant_tail():
    from wind_model_trust import reference_information_weights
    states=jnp.array([[0.,0.,.1,0.],[1.,0.,.1,0.],[2.,0.,.1,0.]])
    mean=jnp.zeros((3,2));variance=jnp.ones(3);need=jnp.ones(1);cells=jnp.array([[1.,0.]])
    local=reference_information_weights(states,mean,variance,need,cells,.01)
    downstream=reference_information_weights(states,mean,variance,need,cells,.01,downstream=True)
    np.testing.assert_allclose(local,[0.,1.,0.],atol=1e-7)
    np.testing.assert_allclose(downstream,[.5,.5,0.],atol=1e-7)
