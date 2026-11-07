import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import jax
import jax.numpy as jnp
import numpy as np

from spatial_poam import SpatialPOAM,SpatialPOAMSettings
from waypoint_poam import train
from voi_diffusion_ipp.wind.poam import POAM,attention_weights,attentive_kernel


def test_geometry_normalization_preserves_physical_distance_and_translation():
    with jax.experimental.enable_x64():
        bounds=jnp.array([[0.,0.],[8.,6.]])
        x=jnp.array([[.2,.5],[4.,3.],[7.5,5.8]])
        model=POAM(x,jax.random.PRNGKey(731),attention_bounds=bounds,prior_std=1.8)
        w=attention_weights(model.parameters,x)
        original={k:v for k,v in model.parameters.items() if not k.startswith('attention_')}
        normalized=(x-bounds.mean(axis=0))/((bounds[1]-bounds[0])/2)
        np.testing.assert_allclose(w,attention_weights(original,normalized),atol=1e-14)
        # Distances remain meters; normalization is not applied to RBF lengths.
        distance=np.sum((np.array(x)[:,None]-np.array(x)[None])**2,axis=2)
        expected=np.zeros_like(distance)
        for i,length in enumerate(model.lengths):
            expected+=np.outer(w[:,i],w[:,i])*np.exp(-distance/(2*float(length)**2))
        expected*=np.exp(float(model.parameters['log_amplitude']))*np.array(w@w.T)
        np.testing.assert_allclose(model.kernel(x,x),expected,atol=1e-12)
        shift=jnp.array([100.,-31.])
        shifted=POAM(x+shift,jax.random.PRNGKey(731),attention_bounds=bounds+shift,prior_std=1.8)
        np.testing.assert_allclose(model.kernel(x,x),shifted.kernel(x+shift,x+shift),atol=1e-12)


def test_spatial_profile_learns_from_measurements_and_never_trains_geometry():
    with jax.experimental.enable_x64():
        yy,xx=np.mgrid[0:5,0:5];positions=np.c_[xx.ravel(),yy.ravel()]/4
        cfg=SpatialPOAMSettings(inducing_count=12,training_batch_size=16)
        gp=SpatialPOAM(positions,[[0.,0.],[1.,1.]],cfg)
        original_offset=np.array(gp.belief.basis.parameters['attention_offset'])
        original_scale=np.array(gp.belief.basis.parameters['attention_scale'])
        y=np.c_[positions[:,0],.3*np.sin(positions[:,1])]
        before=np.asarray(gp.predict(positions)[0])
        gp.update(positions,y)
        after,cov=gp.predict(positions)
        assert np.linalg.norm(np.array(after)-y)<np.linalg.norm(before-y)/5
        assert np.isfinite(cov).all() and np.min(np.asarray(cov)[:,0,0])>=0
        assert len(gp.positions)==25 and int(gp.adam_step)==4
        np.testing.assert_array_equal(gp.belief.basis.parameters['attention_offset'],original_offset)
        np.testing.assert_array_equal(gp.belief.basis.parameters['attention_scale'],original_scale)
        grad=jax.grad(lambda p:attentive_kernel(p,jnp.asarray(positions),jnp.asarray(positions),gp.belief.basis.lengths).sum())(gp.belief.basis.parameters)
        np.testing.assert_array_equal(grad['attention_offset'],0)
        np.testing.assert_array_equal(grad['attention_scale'],0)


def test_legacy_repeated_minibatch_training_is_unchanged():
    with jax.experimental.enable_x64():
        z=jnp.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
        b=POAM(z,jax.random.PRNGKey(3),prior_std=1.8).planning_belief()
        b=jax.tree.map(lambda v:v.astype(jnp.float64) if hasattr(v,'dtype') and jnp.issubdtype(v.dtype,jnp.floating) else v,b)
        m=jax.tree.map(jnp.zeros_like,b.basis.parameters)
        y=jnp.array([[.1,.3],[.5,.2],[.2,.4],[.7,.1]])
        b=b.condition(z,y)
        old=train(b,z,y,jnp.array(4.),m,m,jnp.array(0))
        stacked=train(b,jnp.tile(z[None],(4,1,1)),jnp.tile(y[None],(4,1,1)),jnp.array(4.),m,m,jnp.array(0))
        for one,two in zip(jax.tree.leaves(old),jax.tree.leaves(stacked)):
            np.testing.assert_allclose(one,two,atol=1e-12,rtol=1e-12)
