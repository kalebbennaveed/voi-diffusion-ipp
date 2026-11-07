"""Policy properties, including zero incentive and cumulative loss accounting."""
from pathlib import Path
import sys
import numpy as np
import jax.numpy as jnp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
from ledger_monitoring import select_with_ledger,replenish


def test_extra_information_cannot_spend_unavailable_committed_loss_credit():
    task=jnp.array([1.,1.5,2.])
    commit=jnp.array([.4,.55,.9])
    gain=jnp.array([.01,.2,.8])
    index,monitor,debit,exploring,allowed=select_with_ledger(task,commit,gain,jnp.ones(3,bool),.2,True)
    assert int(index)==1 and int(monitor)==0 and bool(exploring)
    np.testing.assert_allclose(debit,.15,atol=1e-7)
    np.testing.assert_array_equal(allowed,[True,True,False])


def test_irrelevant_or_equal_information_selects_monitoring_even_with_credit():
    task=jnp.array([1.,2.]);commit=jnp.array([.4,.5]);safe=jnp.ones(2,bool)
    for gain,enabled in [(jnp.array([0.,.8]),False),(jnp.array([.2,.2]),True)]:
        index,_,debit,exploring,_=select_with_ledger(task,commit,gain,safe,1.,enabled)
        assert int(index)==0 and not bool(exploring) and float(debit)==0


def test_prediction_of_contact_or_invalid_information_excludes_survey():
    task=jnp.array([1.,2.,3.]);commit=jnp.array([.4,.5,.6])
    index,*_=select_with_ledger(task,commit,jnp.array([.1,.9,jnp.nan]),jnp.array([True,False,True]),1.,True)
    assert int(index)==0


def test_ledger_debits_are_bounded_by_initial_credit_and_declared_refill():
    credit=.4;spent=0.;elapsed=0.
    for debit,seconds in [(.2,5.),(.1,5.),(.15,5.),(.05,5.),(0.,100.)]:
        assert debit<=credit+1e-7
        spent+=debit;elapsed+=seconds
        credit=float(replenish(credit,debit,seconds,.01,.6))
        assert 0<=credit<=.6000001
        assert spent<=.4+.01*elapsed+1e-7
