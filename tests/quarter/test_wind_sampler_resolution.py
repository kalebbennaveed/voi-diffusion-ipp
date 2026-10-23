"""The field sampler must represent the declared GP, not change its inference."""
from dataclasses import replace
import numpy as np
import jax,jax.numpy as jnp
import pytest
from voi_diffusion_ipp.quarter.continuous import ControlSettings
from voi_diffusion_ipp.quarter.environment import initial_belief

@pytest.mark.parametrize('geometry,lower,upper',[
    ('quarter',[-12.,-8.],[12.,11.]),
    ('compact',[-6.7,-5.],[6.7,7.]),
    ('compact_detour',[-6.7,-5.],[6.7,9.]),
])
def test_covariance_anchors_preserve_inference_and_resolve_continuous_sampling(geometry,lower,upper):
    cfg=ControlSettings(environment_scale=.35,inspection_scale=.55,prior_std=1.2,geometry=geometry)
    ordinary=initial_belief(cfg);resolved=initial_belief(replace(cfg,wind_support_nodes=25))
    locations=jnp.array(np.random.default_rng(72).uniform(lower,upper,(60,2)),jnp.float32)
    observations=locations[:7];measurements=jnp.c_[jnp.sin(observations[:,0]),jnp.cos(observations[:,1])]
    condition=jax.jit(lambda b:b.condition(observations,measurements))
    ordinary=condition(ordinary);resolved=condition(resolved)
    for before,after in zip(ordinary.predict(locations),resolved.predict(locations)):np.testing.assert_array_equal(before,after)
    np.testing.assert_array_equal(ordinary.mean,resolved.mean)
    np.testing.assert_array_equal(ordinary.covariance,resolved.covariance)
    exact=np.array(resolved.joint(locations)[1]);sampled=np.array(resolved.sample_moments(locations)[1]);coarse=np.array(ordinary.sample_moments(locations)[1])
    fine_error=np.linalg.norm(sampled-exact)/np.linalg.norm(exact);coarse_error=np.linalg.norm(coarse-exact)/np.linalg.norm(exact)
    assert fine_error<.005
    assert fine_error<coarse_error/20
