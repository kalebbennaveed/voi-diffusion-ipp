"""Safety failures must affect finalist selection, not merely its annotation."""
import sys
from pathlib import Path
from typing import NamedTuple
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import jax.numpy as jnp
import pytest
import implicit_monitoring_v4 as im


@pytest.mark.parametrize('all_fail',[False,True])
def test_rejects_contact_and_labels_failed_hold(monkeypatch,all_fail):
    class Result(NamedTuple):
        final:object
        initial_parameters:object
        initial_field_costs:object
    def search(*args,**kwargs):
        return Result(jnp.array([2.]),jnp.arange(8.)[:,None],jnp.array([5.,1.,2.,3.,4.,5.,6.,7.])[:,None])
    def forecast(theta,x,q,tables,w,basis,env,cfg,aware):
        return im.legacy.old.Rollout(jnp.zeros((2,4)),jnp.zeros((2,1)),jnp.zeros((1,2)),jnp.zeros((2,4)),
            theta[0],jnp.array(0.),jnp.array(0.),jnp.array(1.),(theta[0]==2)|all_fail)
    monkeypatch.setattr(im,'search',search)
    monkeypatch.setattr(im,'execution_forecast',forecast)
    t=im.Tables(*([jnp.zeros(2)]*6))
    plan=im.optimizer(None,im.settings(),None)
    theta,cost,result,forecast,audit=plan(jnp.zeros(4),jnp.zeros(1),t,jnp.arange(7.)[:,None],jnp.zeros(1),jnp.array([0,1],dtype=jnp.uint32))
    if all_fail:
        assert bool(audit['no_feasible']) and bool(forecast.terminated.all())
        assert float(theta[0])==0 and len(audit['candidates'])==3
    else:
        assert not bool(audit['no_feasible']) and not bool(forecast.terminated.any())
        assert float(theta[0])==1 and len(audit['candidates'])==2
