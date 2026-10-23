"""The compact case preserves physical aircraft size and usable geometry."""
import numpy as np
import jax.numpy as jnp
from voi_diffusion_ipp.quarter.environment import environment,wind,table_basis,initial_belief,planning_sampler,table_sampler
from voi_diffusion_ipp.quarter.continuous import ControlSettings


def test_compact_task_geometry_and_reversal():
    env=environment('two',geometry='compact')
    assert len(env.cells)==84 and len(env.regions)==2
    centers=np.asarray(env.regions[:,:2]+env.regions[:,2:])/2
    assert np.linalg.norm(centers[1]-centers[0])>2*3.2
    assert centers[1,1]>centers[0,1]
    assert float(env.rectangles[1,1]-env.rectangles[0,3])>4*.275
    for cell in np.asarray(env.cells):assert float(env.clearance(jnp.r_[cell,jnp.zeros(2)]))>.275
    assert float(jnp.min(env.rectangles[:,4]))>7
    points=jnp.array([[0.,1.],[0.,-3.6],[0.,5.6]])
    f=np.asarray(wind(points,geometry='compact'));r=np.asarray(wind(points,case='reverse',geometry='compact'))
    np.testing.assert_allclose(f[:,0],-r[:,0]);np.testing.assert_allclose(f[:,1],r[:,1])
    assert f[0,0]>3 and np.max(np.abs(f[1:]))==0


def test_compact_table_domain_and_initial_inducing_positions():
    cfg=ControlSettings(geometry='compact',wind_support_nodes=25)
    b=initial_belief(cfg);env=environment(geometry='compact')
    basis=table_basis(cfg.environment_scale,cfg.geometry)
    upper=np.asarray(basis.lower+basis.spacing*jnp.array([90.,79.]))
    assert np.all(np.asarray(env.bounds[0])>np.asarray(basis.lower))
    assert np.all(np.asarray(env.bounds[1])<upper)
    assert b.basis.centers.shape==(36,2)
    assert np.max(np.abs(np.asarray(b.basis.centers[:,0])))<6
    np.testing.assert_allclose(b.basis.lengths,[1.2,2.4])


def test_combined_sampler_uses_same_controller_base_draws():
    import jax
    cfg=ControlSettings(geometry='compact',wind_support_nodes=9,worlds=2,validation_worlds=3)
    b=initial_belief(cfg);pk,ck=jax.random.split(jax.random.PRNGKey(9))
    mean,bank=planning_sampler(cfg)(b,pk,ck)
    controller_mean,controller=table_sampler(cfg.construction_worlds,cfg.environment_scale,cfg.geometry)(b,ck)
    np.testing.assert_allclose(mean,controller_mean,atol=1e-6)
    np.testing.assert_allclose(bank[:cfg.construction_worlds],controller,atol=1e-6)
    assert bank.shape[0]==cfg.construction_worlds+cfg.worlds+cfg.validation_worlds


def test_hard_passages_have_adverse_a_c_and_exactly_calm_b():
    from voi_diffusion_ipp.quarter.environment import metadata
    env=environment(geometry='compact_passages',case='crosswind')
    assert float(env.rectangles[1,1]-env.rectangles[0,3])==np.float32(1.7)
    assert float(env.rectangles[0,1]-env.bounds[0,1])>1.79
    # Every physically usable cross-section has wind: no calm center stripe.
    ys=np.r_[np.linspace(.25+.275,1.95-.275,15),np.linspace(-4.2+.275,-2.4-.275,15)]
    field=np.asarray(wind(jnp.array(np.c_[np.zeros(len(ys)),ys]),'crosswind',3.75,geometry='compact_passages'))
    assert np.min(np.linalg.norm(field,axis=1))>3.74
    assert np.all(field[:15,1]>0) and np.all(field[15:,1]<0)
    upper=jnp.array([[x,y] for x in np.linspace(-5,5,15) for y in np.linspace(4.7,6.7,5)])
    np.testing.assert_array_equal(wind(upper,'crosswind',3.75,geometry='compact_passages'),np.zeros((len(upper),2)))
    assert metadata(geometry='compact_passages')['passage_labels']=={'A':1.1,'B':5.6,'C':-3.3}


def test_hard_passages_share_compact_belief_domain():
    cfg=ControlSettings(geometry='compact_passages',wind_support_nodes=9)
    b=initial_belief(cfg);env=environment(geometry=cfg.geometry)
    np.testing.assert_array_equal(b.basis.centers,initial_belief(ControlSettings(geometry='compact',wind_support_nodes=9)).basis.centers)
    basis=table_basis(cfg.environment_scale,cfg.geometry)
    assert np.all(np.asarray(env.bounds[0])>np.asarray(basis.lower))
    assert np.all(np.asarray(env.bounds[1])<np.asarray(basis.lower)+np.asarray(basis.spacing)*[90,79])


def test_farther_b_extends_building_and_wind_domain_without_coarsening():
    from voi_diffusion_ipp.quarter.environment import table_shape,metadata
    near=environment(geometry='compact_detour',detour_depth=0)
    far=environment(geometry='compact_detour',detour_depth=2)
    np.testing.assert_array_equal(near.regions,far.regions)
    assert float(far.rectangles[1,3]-near.rectangles[1,3])==2.
    assert float(far.bounds[1,1]-near.bounds[1,1])==2.
    cfg=ControlSettings(geometry='compact_detour',detour_depth=2,wind_support_nodes=9)
    basis=table_basis(cfg.environment_scale,cfg.geometry);shape=table_shape(cfg.geometry)
    np.testing.assert_allclose(basis.spacing,[.2,.2])
    assert shape==(90,91,2)
    assert np.all(np.asarray(far.bounds[1])<np.asarray(basis.lower)+np.asarray(basis.spacing)*np.array([shape[1]-2,shape[0]-2]))
    np.testing.assert_array_equal(wind(jnp.array([[0.,7.6]]),'crosswind',3.65,geometry='compact_detour'),[[0.,0.]])
    assert float(wind(jnp.array([[0.,1.]]),'crosswind',3.65,geometry='compact_detour')[0,0])< -3.64
    assert metadata(geometry='compact_detour',detour_depth=2)['passage_labels']['B']==7.6


def test_declared_initial_measurements_inform_a_but_do_not_reveal_b():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    observations=json.loads((root/'configs/scenarios/detour_a_wind_stations.json').read_text())['observations']
    cfg=ControlSettings(geometry='compact_detour',wind_support_nodes=9,initial_wind_observations=tuple(tuple(row) for row in observations))
    observed=initial_belief(cfg)
    mean,cov=observed.predict(jnp.array([[0.,1.],[0.,7.6]]))
    assert float(mean[0,0])< -3.
    assert float(cov[0,0,0])<.12
    assert float(cov[1,0,0])>10*float(cov[0,0,0])
    # Exposing the physical start does not itself reveal a wind observation.
    from dataclasses import replace
    prior=initial_belief(replace(cfg,initial_wind_observations=()))
    same=initial_belief(replace(cfg,initial_wind_observations=(),wind_at_start=True))
    np.testing.assert_array_equal(prior.mean,same.mean)
    np.testing.assert_array_equal(prior.covariance,same.covariance)
