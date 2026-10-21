"""Regression checks for failures identified in the recorded mission audit."""
from dataclasses import replace
import numpy as np
import jax.numpy as jnp
from voi_diffusion_ipp.quarter.continuous import ControlSettings,aggregate_costs
from voi_diffusion_ipp.quarter.diffusion import lower_tail
from voi_diffusion_ipp.quarter.environment import environment,wind
from voi_diffusion_ipp.quarter.rrt import demonstrations,segment_free,winding_signature

def test_optimism_cannot_discard_unfavorable_constraint_penalties():
    monitor=jnp.array([3.,4.,6.,8.]);penalty=jnp.array([0.,0.,100.,200.])
    combined=aggregate_costs(monitor,penalty)
    assert float(lower_tail(combined,.5))==float(lower_tail(monitor,.5)+penalty.mean())
    np.testing.assert_allclose(combined.mean(),(monitor+penalty).mean())
    assert float(lower_tail(combined,.5))>float(lower_tail(aggregate_costs(monitor,penalty,'tail'),.5))+70

def test_geometry_only_rrt_initializations_discover_all_three_path_classes():
    cfg=ControlSettings(environment_scale=.35,inspection_scale=.55,horizon=45,dt=.2,commit=5,rrt_paths=3,rrt_visits=1,rrt_diversity=True)
    env=environment('two',.35,.55)
    controls,record=demonstrations([-9.8,2.8,0.,0.],env,cfg,8101)
    paths=[np.array(r['path']) for r in record['records'] if r.get('goal_index')==1 and r['cruise']==1.8]
    assert len(set(winding_signature(p,np.array(env.rectangles)) for p in paths))==3
    for path in paths:
        assert all(segment_free(a,b,np.array(env.rectangles),np.array(env.bounds),.65) for a,b in zip(path[:-1],path[1:]))
    assert controls.shape==(12,450)

def test_calm_a_is_an_explicit_new_physical_case_and_preserves_old_case():
    points=jnp.array([[0.,8.],[0.,-23.],[0.,27.]])*.35
    calm=np.array(wind(points,'calm_a',scale=.35))
    all_windy=np.array(wind(points,'commitment',scale=.35))
    np.testing.assert_array_equal(calm[0],0.)
    np.testing.assert_array_equal(calm[1:],all_windy[1:])
    np.testing.assert_allclose(all_windy[:,0],3.1,atol=1e-6)
    np.testing.assert_array_equal(environment(scale=.35,inspection_scale=.55,case='calm_a').cells,environment(scale=.35,inspection_scale=.55,case='commitment').cells)
