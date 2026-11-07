"""Unmeasured planning mean and identity transport regression cases."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import numpy as np
import jax
import jax.numpy as jnp
import implicit_monitoring_v5 as im
from observed_poam import _double
from voi_diffusion_ipp.wind.poam import POAM


def test_unsupported_nonzero_gp_mean_does_not_leak_into_performance():
    mu=jnp.array([[14.,-8.],[2.,1.]])
    draws=jnp.stack((mu+1,mu-1));variance=jnp.array([3.24,.0324])
    fields,r=im.planning_fields(mu,draws,variance,3.24,15)
    np.testing.assert_array_equal(fields[:,0],0)
    np.testing.assert_allclose(fields.mean(0),r[:,None]*mu,atol=1e-12)
    np.testing.assert_array_equal(mu[0],[14.,-8.])


def test_unchanged_basis_does_not_repeatedly_transport_old_statistics():
    from supported_poam_identity import condition_supported
    x=jnp.array([[0.,0.],[.15,0.],[0.,.15],[.15,.15]])
    b=_double(POAM(x,jax.random.PRNGKey(31),prior_std=1.8).planning_belief())
    # All points are already inducing locations; no allocation occurs.
    mask=jnp.ones(4,bool);xx=jnp.tile(x,(8,1));yy=jnp.stack((jnp.sin(xx[:,0]),jnp.cos(xx[:,1])),axis=1)
    batch,_=condition_supported(b,xx,yy,mask)
    online=b
    for i in range(len(xx)):online,_=condition_supported(online,xx[i:i+1],yy[i:i+1],mask)
    np.testing.assert_allclose(online.a,batch.a,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(online.B,batch.B,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(online.mean,batch.mean,atol=1e-8,rtol=1e-8)
    np.testing.assert_allclose(online.covariance,batch.covariance,atol=1e-8,rtol=1e-8)
