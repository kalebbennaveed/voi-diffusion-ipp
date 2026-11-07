"""Decision-rule fixtures, separate from physical mission validation.

A two-route toy isolates whether the actual selector values information only
through its downstream task effect. These are not replacement robot models.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
from types import SimpleNamespace
import numpy as np
import pytest
import jax.numpy as jnp
import jax
import exploration_monitoring as dual
import legacy_monitoring as legacy

def fixture(monkeypatch,wind_matters=True,learning=True,exploration_cost=.2,conditional_samples=1):
    cfg=legacy.Settings();env=SimpleNamespace(process_noise=jnp.array([.02]),targets=jnp.array([.7]))
    selector=dual.BayesianSelector(env,cfg,learning=learning,roots=2,branches=1,conditional_worlds=conditional_samples)
    field=lambda values:jnp.broadcast_to(jnp.asarray(values)[:,None,None,None],(len(values),*legacy.SHAPE))
    selector.initial_fields=lambda b,k,ck:(jnp.array(0.),jnp.array(0.),field([1.]))
    def rollout(parameters,x,q,mu,inner,world,prefix=False):
        n=len(parameters);steps=selector.prefix_cfg.steps if prefix else selector.tail_cfg.steps
        flags=parameters[:,0]
        xs=jnp.zeros((n,steps+1,4)).at[:,:,0].set(flags[:,None])
        qs=jnp.full((n,steps+1,1),.5)
        if prefix:cost=exploration_cost*flags
        else:
            actual=world[:,0,0,0] if wind_matters else jnp.full(n,-1.)
            cost=(flags-actual)**2
        return legacy.old.Rollout(xs,qs,jnp.zeros((n,steps,2)),xs,cost,jnp.zeros(n),jnp.zeros(n),jnp.ones(n),jnp.zeros(n,bool))
    selector.prefix=lambda th,x,q,mu,inner,w:rollout(th,x,q,mu,inner,w,True)
    selector.tail=rollout
    def update(b,path,controls,world,noise,terminated):
        # Only the exploratory prefix reveals which wind regime holds.
        posterior=jnp.where(path[:,-1,0]>.5,world[:,0,0,0],0.)
        return posterior,jnp.zeros((len(path),25,2)),jnp.zeros((len(path),25,2))
    selector.update=update
    def children(b,k,ck):
        values=jnp.where(b[:,None]==0,jnp.array([-1.,1.])[:conditional_samples][None],b[:,None])
        worlds=jnp.broadcast_to(values[:,:,None,None,None],(len(b),conditional_samples,*legacy.SHAPE))
        return b,jnp.zeros(len(b)),worlds
    selector.child_fields=children
    controls=jnp.repeat(jnp.array([-1.,1.,-1.,1.])[:,None],2*selector.tail_cfg.steps,axis=1)
    monkeypatch.setattr(dual,'continuation_seeds',lambda x,q,env,cfg:controls)
    candidates=jnp.stack([jnp.zeros(2*cfg.steps),jnp.ones(2*cfg.steps)])
    return selector.select(jnp.array(0.),jnp.zeros(4),jnp.array([.5]),candidates,jax.random.PRNGKey(0),jax.random.PRNGKey(1))

@pytest.mark.parametrize("conditional_samples",[1,2])
def test_irrelevant_wind_never_rewards_information_detour(monkeypatch,conditional_samples):
    decision=fixture(monkeypatch,conditional_samples=conditional_samples,wind_matters=False)
    np.testing.assert_allclose(decision['scores'],[0,.2],atol=1e-6)
    assert decision['index']==0

@pytest.mark.parametrize("conditional_samples",[1,2])
def test_useful_information_can_justify_initial_task_cost(monkeypatch,conditional_samples):
    decision=fixture(monkeypatch,conditional_samples=conditional_samples,wind_matters=True)
    np.testing.assert_allclose(decision['scores'],[4,.2],atol=1e-6)
    assert decision['index']==1

@pytest.mark.parametrize("conditional_samples",[1,2])
def test_frozen_belief_does_not_receive_revealed_wind(monkeypatch,conditional_samples):
    decision=fixture(monkeypatch,conditional_samples=conditional_samples,wind_matters=True,learning=False)
    np.testing.assert_allclose(decision['scores'],[4,4.2],atol=1e-6)
    assert decision['index']==0

@pytest.mark.parametrize("conditional_samples",[1,2])
def test_learning_is_rejected_when_its_task_cost_exceeds_benefit(monkeypatch,conditional_samples):
    decision=fixture(monkeypatch,conditional_samples=conditional_samples,wind_matters=True,exploration_cost=5.)
    np.testing.assert_allclose(decision['scores'],[4,5],atol=1e-6)
    assert decision['index']==0
