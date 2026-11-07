import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import numpy as np
import jax
import jax.numpy as jnp
import rmse_monitoring as model
from voi_diffusion_ipp.wind.poam import POAM
from cached_poam_fields import CachedFields
from task_wind_acquisition import deficit_weights,expected_variance_reduction,covariance_gains,route_weights


def test_mask_uses_exact_no_sensing_clarity_and_preserves_zero_need():
    q=np.array([.74,.74]);decay=np.array([.16,0.]);target=np.array([.7,.35])
    need=deficit_weights(q,target,decay,15)
    assert need[0]>0 and need[1]==0
    expected=np.maximum(.7-.74/(1+np.array([3.75,7.5,15])*.16*.74),0).mean()
    np.testing.assert_allclose(need[0],expected)


def test_correlated_reading_reduces_uncertainty_inside_mask_even_if_outside():
    cov=np.array([[1.,.8],[.8,1.]])
    weights=np.array([1.,0.]);gain=np.asarray(expected_variance_reduction(cov,np.diag(cov),weights,.01))
    assert gain[1]>0
    for i in range(2):
        posterior=cov-np.outer(cov[:,i],cov[:,i])/(cov[i,i]+.01)
        np.testing.assert_allclose(gain[i],weights@(np.diag(cov)-np.diag(posterior)))


def test_acquisition_covariance_includes_poam_conditional_residual():
    xx,yy=np.meshgrid(np.linspace(0,2,5),np.linspace(0,2,5));points=jnp.asarray(np.c_[xx.ravel(),yy.ravel()])
    gp=POAM(points[::4],jax.random.PRNGKey(13),lengths=(.2,.5,1.,2.),prior_std=1.8,
        noise_std=.1,attention_bounds=jnp.array([[0.,0.],[2.,2.]]),prior_support=points)
    gp.update(np.array([[.1,.2],[1.3,1.4]]),np.array([[.7,-.3],[.2,.6]]),jax.random.PRNGKey(14),train_steps=0)
    b=jax.tree.map(lambda a:a.astype(jnp.float64) if hasattr(a,'dtype') and jnp.issubdtype(a.dtype,jnp.floating) else a,gp.planning_belief())
    cache=CachedFields(points,(5,5,2),model.settings(),points);_,cov=cache.map(b)
    ids=jnp.array([1,6,17]);weights=jnp.arange(1,26,dtype=jnp.float64);weights/=weights.sum()
    gain=covariance_gains(b,points,cache.Kmap,cache.posterior[1],ids,weights,cov[:,0,0])
    K=b.kernel(points,points);A=b.kernel(points,b.basis.centers)@b.inverse
    joint=K-A@b.kernel(b.basis.centers,points)+A@b.covariance@A.T
    expected=expected_variance_reduction(joint[:,ids],jnp.diag(joint)[ids],weights,b.noise_std**2)
    np.testing.assert_allclose(gain,expected,rtol=1e-6,atol=1e-7)


def test_route_mask_covers_low_deficit_transit_and_vanishes_without_wind_coupling():
    nodes=np.array([[0.,0.],[1.,0.],[2.,0.],[1.,2.]])
    paths=[np.array([[0.,0.],[2.,0.]])];need=np.array([0.,0.,1.,0.])
    mean=np.zeros((4,2));variance=np.ones(4)
    mask=route_weights(nodes,paths,need,nodes,mean,variance)
    assert mask[1]>0 and mask[1]>100*mask[3]
    np.testing.assert_array_equal(route_weights(nodes,paths,need,nodes,mean,variance,drag=0),np.zeros(4))
