"""Independent information identities for the active-sensing objective."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import numpy as np
import jax.numpy as jnp
from active_sensing_monitoring import information_gain

class GaussianFixture:
    def __init__(self,cov,noise=.1):self.cov=jnp.asarray(cov);self.noise_std=noise
    def joint(self,positions):return jnp.zeros((len(positions),2)),self.cov

def test_independent_components_and_repeated_samples():
    n=25;variance=.4;noise=.1
    repeated=GaussianFixture(np.full((n,n),variance),noise)
    got=float(information_gain(repeated,jnp.zeros((n,2))))
    np.testing.assert_allclose(got,np.log1p(n*variance/noise**2),rtol=2e-5)
    independent=GaussianFixture(np.eye(n)*variance,noise)
    np.testing.assert_allclose(float(information_gain(independent,jnp.zeros((n,2)))),n*np.log1p(variance/noise**2),rtol=2e-5)
    assert got<float(information_gain(independent,jnp.zeros((n,2))))

def test_known_wind_and_noise_limits():
    positions=jnp.zeros((4,2))
    assert float(information_gain(GaussianFixture(np.zeros((4,4))),positions))==0
    low=float(information_gain(GaussianFixture(np.eye(4),.1),positions))
    high=float(information_gain(GaussianFixture(np.eye(4),10.),positions))
    assert 0<high<low

from active_sensing_monitoring import inducing_information_gain
from types import SimpleNamespace

def test_inducing_information_retains_correlated_residual():
    # One latent inducing variable; independent uncertain residual at each point.
    n=5;latent=.3;residual=.8;noise=.1
    A=np.ones((n,1));K=A@A.T+residual*np.eye(n)
    b=SimpleNamespace(basis=SimpleNamespace(kernel_centers=lambda x:jnp.asarray(A)),inverse=jnp.eye(1),covariance=jnp.asarray([[latent]]),kernel=lambda x,y:jnp.asarray(K),noise_std=noise)
    positions=jnp.zeros((n,2))
    got=float(inducing_information_gain(b,positions))
    np.testing.assert_allclose(got,np.log1p(n*latent/(residual+noise**2)),rtol=2e-5)
    # Knowing the latent variable leaves residual wind uncertainty, but no
    # information about that known variable can be gained by another visit.
    b.covariance=jnp.zeros((1,1))
    np.testing.assert_allclose(float(inducing_information_gain(b,positions)),0,atol=1e-6)
