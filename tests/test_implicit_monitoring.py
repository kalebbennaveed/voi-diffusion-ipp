"""Model separation, coherent planning distribution, and robust force bounds."""
import inspect
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import numpy as np
import jax
import jax.numpy as jnp
import implicit_monitoring as im
from voi_diffusion_ipp.wind.poam import POAM
from supported_poam import condition_supported


def test_cold_planning_is_exactly_zero_without_erasing_safety_uncertainty():
    mu=jnp.zeros((3,2));draws=jnp.ones((4,3,2))*4
    var=jnp.full(3,1.8**2)
    perf,support=im.planning_fields(mu,draws,var,1.8**2,jnp.array(0))
    np.testing.assert_array_equal(perf,0)
    np.testing.assert_array_equal(support,0)
    np.testing.assert_array_equal(draws,4)
    np.testing.assert_allclose(var,1.8**2)


def test_actual_measurements_change_local_planning_and_leave_remote_zero():
    with jax.experimental.enable_x64():
        xy=jnp.array([[0.,0.],[.3,0.],[0.,.3],[.3,.3]])
        b=im._double(POAM(xy,jax.random.PRNGKey(8),prior_std=1.8).planning_belief())
        b,mask=condition_supported(b,xy[:1],jnp.array([[4.,-1.]]),jnp.zeros(4,bool))
        locations=jnp.array([[0.,0.],[.1,0.],[20.,20.]])
        mu,cov=b.predict(locations)
        eps=jax.random.normal(jax.random.PRNGKey(9),(2,*b.sample_shape))
        co=b.sample_coefficients(jnp.concatenate((eps,-eps)))
        raw=jax.vmap(lambda c:b.basis.evaluate(locations,c))(co)
        perf,r=im.planning_fields(mu,raw,cov[:,0,0],jnp.exp(b.basis.parameters['log_amplitude']),1)
        assert float(r[0])>.99 and float(r[1])>.5
        assert float(r[2])<1e-12
        np.testing.assert_allclose(perf[:,2],0,atol=1e-12)
        np.testing.assert_allclose(perf.mean(axis=0),mu,atol=1e-10)
        assert float(cov[2,0,0])>3.
        # Diagonal congruence preserves full spatial covariance and PSD.
        _,joint=b.joint(locations)
        covariance=jnp.sqrt(r[:,None]*r[None,:])*joint
        assert np.linalg.eigvalsh(covariance).min()>-1e-10
        assert abs(float(covariance[0,1]))>1e-5


def test_force_bound_contains_nonlinear_drag_over_confidence_ball():
    rng=np.random.default_rng(54)
    for _ in range(12):
        v=rng.normal(size=2);mu=rng.normal(size=2)*3;radius=rng.uniform(.1,6.)
        gradients=rng.normal(size=(9,2));angles=np.linspace(0,2*np.pi,501)
        wind=mu+radius*np.c_[np.cos(angles),np.sin(angles)]
        actual=gradients@np.asarray(im.m.physical.acceleration(v,wind)).T
        bound=np.asarray(im.force_support(jnp.asarray(gradients),jnp.asarray(v),jnp.asarray(mu),radius))
        assert np.all(actual<=bound[:,None]+1e-8)


def test_planning_and_controller_interfaces_have_no_physical_truth():
    for fn in [im.initial_belief,im.field_builder,im.optimizer,im.controller,im.constraints]:
        assert not any('truth' in p or 'hidden' in p for p in inspect.signature(fn).parameters)
    source=inspect.getsource(im.execution_block)
    assert source.count('hidden_physics_field')==2  # signature and plant advance only
    assert 'command(x,reference,a,tables)' in source


def test_table_padding_covers_revised_tall_environment_and_derivatives():
    basis,points,shape=im.table_grid(np.array([[0.,0.],[8.,6.]]))
    values=jnp.stack((points[:,0]+2*points[:,1],points[:,1]),axis=1).reshape(shape)
    p=jnp.array([7.9,5.9]);got=basis.evaluate(p,values)[0]
    np.testing.assert_allclose(got,[19.7,5.9],atol=1e-10)
    grad=jax.jacfwd(lambda p:basis.evaluate(p,values)[0])(p)
    np.testing.assert_allclose(grad,[[1,2],[0,1]],atol=1e-10)
