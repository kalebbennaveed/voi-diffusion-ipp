import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import numpy as np
import jax
import jax.numpy as jnp
from types import SimpleNamespace
from rmse_gate import wind_vector_rmse,ready_for_mbd
import rmse_monitoring as rm
from voi_diffusion_ipp.safety.qp import solve


def test_gate_requires_vector_accuracy_and_excludes_solid_cells():
    truth=np.array([[3.,4.],[100.,100.]])
    assert wind_vector_rmse(np.zeros((2,2)),truth,[0])==5
    assert ready_for_mbd(.2) and not ready_for_mbd(.20001)
    assert not ready_for_mbd(np.nan) and not ready_for_mbd(np.inf)
    assert not ready_for_mbd(-1)


def test_boundary_constraint_brakes_before_contact_with_hard_authority():
    cfg=rm.settings();env=SimpleNamespace(bounds=jnp.array([[0.,0.],[8.,6.]]))
    basis=rm.ExtendedTableBasis(jnp.array([-1.,-1.]),jnp.array([.2,.2]))
    mean=rm.TableMean(basis,jnp.zeros((41,51,2)));std=jnp.zeros((41,51,1))
    x=jnp.array([4.,.5,0.,-.6]);A,b=rm.boundary_constraints(x,mean,std,basis,env,cfg)
    sol=solve(jnp.array([0.,-1.6]),A,b,cfg.umax,jnp.zeros(2))
    assert bool(sol.feasible) and sol.control[1]>0
    assert np.max(np.abs(sol.control))<=cfg.umax+1e-6
    assert np.max(np.array(A@sol.control-b))<=2e-5
    # Check physical nonlinear drag over the entire declared wind disk.
    theta=np.linspace(0,2*np.pi,1024)
    w=.3*np.c_[np.cos(theta),np.sin(theta)]
    drift=rm.m.physical.acceleration(np.array(x[2:]),w)
    h=float(x[1]-cfg.safety_margin);hdot=float(x[3]);rate=1.5
    assert np.min(float(sol.control[1])+drift[:,1]+2*rate*hdot+rate**2*h)>=-2e-5


def test_planning_antithetic_fields_keep_unattenuated_posterior_mean():
    from voi_diffusion_ipp.wind.poam import POAM
    from observed_poam import _double
    z=jnp.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    with jax.experimental.enable_x64():
        b=_double(POAM(z,jax.random.PRNGKey(1),prior_std=1.8,spectral_count=8,prior_support=z).planning_belief())
        b=b.condition(z[:1],jnp.array([[3.,-2.]]))
        table=rm.field_builder(z,(2,2,2),rm.settings())(b,jnp.array(1),jax.random.PRNGKey(2),jax.random.PRNGKey(3))
        np.testing.assert_allclose(table.planning.mean(axis=0),np.array(b.mean_at(z)).reshape(2,2,2),atol=1e-8)
        np.testing.assert_allclose(table.validation.mean(axis=0),table.mean,atol=1e-8)
        assert np.any(np.std(table.planning,axis=0)>.1)


def test_backup_cache_extension_is_finite_beyond_every_wall_in_both_precisions():
    from voi_diffusion_ipp.wind.cache import TableBasis
    for enabled in (False,True):
        with jax.experimental.enable_x64(enabled):
            dtype=jnp.float64 if enabled else jnp.float32
            b=rm.ExtendedTableBasis(jnp.array([-1.,-1.],dtype=dtype),jnp.full(2,.2,dtype=dtype))
            field=jnp.arange(41*51*2,dtype=dtype).reshape(41,51,2)/100
            outside=jnp.array([[-2.,1.],[2.,-2.],[10.,1.],[2.,8.],[-1.,-1.]],dtype=dtype)
            assert np.isfinite(np.array(b.evaluate(outside,field))).all()
            for p in outside:
                assert np.isfinite(np.array(jax.jacfwd(lambda x:b.evaluate(x,field))(p))).all()
            interior=jnp.array([[0.,0.],[4.,3.],[8.,6.]],dtype=dtype)
            np.testing.assert_array_equal(b.evaluate(interior,field),TableBasis(*b).evaluate(interior,field))


def test_mbd_validates_optimized_alternate_and_rejects_failed_global(monkeypatch):
    from typing import NamedTuple
    class Result(NamedTuple):
        final:object
        finalists:object
    def search(*args,**kwargs):
        return Result(jnp.array([2.]),jnp.array([[2.],[7.],[8.],[9.],[10.],[11.],[12.]]))
    def rollout(theta,*args):
        return rm.legacy.old.Rollout(jnp.zeros((2,4)),jnp.zeros((2,1)),jnp.zeros((1,2)),jnp.zeros((2,4)),
            theta[0],jnp.array(0.),jnp.array(0.),jnp.array(1.),jnp.array(False))
    def forecast(theta,*args):
        return rollout(theta)._replace(terminated=theta[0]==2)
    monkeypatch.setattr(rm,'search',search)
    monkeypatch.setattr(rm.old,'rollout',rollout)
    monkeypatch.setattr(rm.core,'execution_forecast',forecast)
    tables=rm.Tables(*([jnp.zeros(2)]*6))
    plan=rm.optimizer(None,rm.settings(),None)
    theta,score,result,fore,audit=plan(jnp.zeros(4),jnp.zeros(1),tables,jnp.arange(7.)[:,None],jnp.zeros(1),jax.random.PRNGKey(1))
    assert float(theta[0])==7 # optimized leader, not the old unoptimized seed 1
    assert not bool(audit['no_feasible']) and not bool(fore.terminated.any())
    assert int(audit['leader_reevaluations'])==7


def test_safety_aware_selection_logs_relaxation_but_still_rejects_contact(monkeypatch):
    from typing import NamedTuple
    class Result(NamedTuple):
        final:object
        finalists:object
    monkeypatch.setattr(rm,'search',lambda *a,**k:Result(jnp.array([2.]),jnp.arange(2.,9.)[:,None]))
    def rollout(theta,*args):
        return rm.legacy.old.Rollout(jnp.zeros((2,4)),jnp.zeros((2,1)),jnp.zeros((1,2)),jnp.zeros((2,4)),
            theta[0],jnp.array(.1),jnp.where(theta[0]==2,.8,0.),jnp.array(.3),jnp.array(False))
    monkeypatch.setattr(rm.old,'rollout',rollout)
    monkeypatch.setattr(rm.core,'execution_forecast',rollout)
    tables=rm.Tables(*([jnp.zeros(2)]*6))
    args=(jnp.zeros(4),jnp.zeros(1),tables,jnp.arange(7.)[:,None],jnp.zeros(1),jax.random.PRNGKey(1))
    strict=rm.optimizer(None,rm.settings(),None)(*args)
    relaxed=rm.optimizer(None,rm.settings(),None,allow_validation_slack=True)(*args)
    assert float(strict[0][0])==3
    assert float(relaxed[0][0])==2
    assert bool(relaxed[-1]['feasible'][0]) and not bool(relaxed[-1]['safety_feasible'][0])
    np.testing.assert_allclose(relaxed[1],2.1)
    monkeypatch.setattr(rm.core,'execution_forecast',lambda theta,*args:rollout(theta)._replace(terminated=theta[0]==2))
    contact=rm.optimizer(None,rm.settings(),None,allow_validation_slack=True)(*args)
    assert float(contact[0][0])==3
