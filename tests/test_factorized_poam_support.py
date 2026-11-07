import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import numpy as np
import jax
import jax.numpy as jnp
import rmse_monitoring as model
from voi_diffusion_ipp.wind.poam import POAM
from waypoint_poam import condition,train
from cached_poam_fields import CachedFields


def test_cholesky_support_representation_preserves_full_fields_after_training():
    xx,yy=np.meshgrid(np.linspace(0,2,8),np.linspace(0,2,8));points=jnp.asarray(np.c_[xx.ravel(),yy.ravel()])
    b=POAM(points[::5],jax.random.PRNGKey(13),lengths=(.2,.5,1.,2.),prior_std=1.8,
        noise_std=.1,attention_bounds=jnp.array([[0.,0.],[2.,2.]]),prior_support=points).planning_belief()
    b=jax.tree.map(lambda z:z.astype(jnp.float64) if hasattr(z,'dtype') and jnp.issubdtype(z.dtype,jnp.floating) else z,b)
    fast=b._replace(basis=b.basis._replace(support_inverse=jnp.zeros((0,0),dtype=jnp.float64)))
    mask=jnp.zeros(len(b.mean),bool);x=jnp.array([[.3,.8],[1.4,1.3]]);y=jnp.array([[2.,-1.],[-.4,.6]])
    moment=jax.tree.map(jnp.zeros_like,b.basis.parameters)
    eps=jax.random.normal(jax.random.PRNGKey(99),(4,*b.sample_shape),dtype=jnp.float64)
    for stage in range(3):
        if stage==1:b,_=condition(b,x,y,mask);fast,_=condition(fast,x,y,mask)
        if stage==2:
            b,*_=train(b,x,y,jnp.array(2.),moment,moment,jnp.array(0),.01)
            fast,*_=train(fast,x,y,jnp.array(2.),moment,moment,jnp.array(0),.01)
        assert fast.basis.support_inverse.size==0
        for left,right in zip(fast.predict(points),b.predict(points)):np.testing.assert_allclose(left,right,rtol=1e-9,atol=1e-9)
        c1=fast.sample_coefficients(eps);c2=b.sample_coefficients(eps)
        f1=jax.vmap(lambda c:fast.basis.evaluate(points,c))(c1);f2=jax.vmap(lambda c:b.basis.evaluate(points,c))(c2)
        np.testing.assert_allclose(f1,f2,rtol=1e-8,atol=1e-8)
        for left,right in zip(fast.sample_moments(points[::7]),b.sample_moments(points[::7])):
            np.testing.assert_allclose(left,right,rtol=1e-8,atol=1e-8)
