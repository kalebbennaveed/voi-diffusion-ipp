"""Scenario changes must be physical inputs, not scripted planner behavior."""
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.quarter.environment import environment, metadata, wind


def test_crosswind_preserves_geometry_and_leaves_b_calm():
    for a,b in zip(environment(scale=.35,inspection_scale=.55),
                   environment(scale=.35,inspection_scale=.55,case='crosswind')):
        np.testing.assert_array_equal(a,b)
    x=np.linspace(-18,18,37)
    for y in [8,27]:
        field=np.asarray(wind(jnp.asarray(np.c_[x,np.full_like(x,y)])*.35,'crosswind',scale=.35))
        assert np.all(field[:,0]<-2)
        assert field[:,1].min()<-.8 and field[:,1].max()>.8
    sheltered=np.asarray(wind(jnp.asarray(np.c_[x,np.full_like(x,-23)])*.35,'crosswind',scale=.35))
    np.testing.assert_array_equal(sheltered,0)


def test_larger_eastern_region_has_more_uniform_task_cells():
    regular=environment(scale=.35,inspection_scale=.55)
    enlarged=environment(scale=.35,inspection_scale=.55,case='commitment')
    np.testing.assert_array_equal(regular.rectangles,enlarged.rectangles)
    np.testing.assert_array_equal(regular.cells[:42],enlarged.cells[:42])
    assert len(enlarged.cells)==120
    assert metadata(scale=.35,inspection_scale=.55,case='commitment')['region_cell_counts']==[42,78]
    np.testing.assert_array_equal(enlarged.targets,.75)
    for r in np.asarray(enlarged.regions):
        for p in (r[:2],r[2:],np.array([r[0],r[3]]),np.array([r[2],r[1]])):
            assert float(enlarged.clearance(jnp.r_[p,0.,0.]))>.275
    width=regular.regions[1,3]-regular.regions[1,1]
    np.testing.assert_allclose(enlarged.regions[1,3]-enlarged.regions[1,1],2*width)


def test_commitment_field_loads_all_three_passages():
    points=jnp.array([[0.,8.],[0.,-23.],[0.,27.]])*.35
    field=np.asarray(wind(points,'commitment',scale=.35))
    np.testing.assert_allclose(field[:,0],3.1,atol=1e-6)
    np.testing.assert_allclose(field[:,1],0,atol=1e-6)
