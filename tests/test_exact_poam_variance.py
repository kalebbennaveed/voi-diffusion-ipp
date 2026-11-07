import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import numpy as np
import jax
import jax.numpy as jnp
import full_learning_monitoring as model
from voi_diffusion_ipp.wind.poam import POAM
from waypoint_poam import condition
from cached_poam_fields import CachedFields
from exact_poam_variance import PosteriorStandardDeviation,ResolvedVarianceBasis


def belief():
    z=jnp.array([[0.,0.],[.4,0.],[.8,0.],[0.,.4],[.4,.4],[.8,.4],[0.,.8],[.4,.8],[.8,.8]])
    b=POAM(z,jax.random.PRNGKey(731),lengths=(.1,.2,.5,1.),prior_std=1.8,noise_std=.1).planning_belief()
    return jax.tree.map(lambda x:x.astype(jnp.float64) if hasattr(x,'dtype') and jnp.issubdtype(x.dtype,jnp.floating) else x,b)


def test_current_actual_measurement_wins_equal_first_pivot_and_is_conditioned():
    b=belief();p=jnp.array([[.23,.31]]);y=jnp.array([[2.,-1.]])
    updated,mask=condition(b,p,y,jnp.zeros(len(b.mean),bool),observation_first=True)
    np.testing.assert_array_equal(updated.basis.centers[0],p[0])
    mean,cov=updated.predict(p)
    assert cov[0,0,0] <= .01001
    assert np.linalg.norm(np.asarray(mean-y)) < .02
    assert np.linalg.eigvalsh(np.asarray(updated.covariance)).min()>=-1e-10
    # Repeated real readings reduce uncertainty, rather than retaining an
    # unobserved conditional-residual floor at the measured location.
    before=float(cov[0,0,0])
    for _ in range(4):updated,mask=condition(updated,p,y,mask,observation_first=True)
    assert float(updated.predict(p)[1][0,0,0]) < before/3


def test_direct_variance_matches_full_posterior_between_cache_nodes():
    b=belief();p=jnp.array([[.23,.31]]);y=jnp.array([[2.,-1.]])
    b,_=condition(b,p,y,jnp.zeros(len(b.mean),bool),observation_first=True)
    basis,points,shape=model.table_grid(jnp.array([[0.,0.],[1.,1.]]),.1)
    cache=CachedFields(points,shape,model.settings(),points,exact_variance=True)
    tables=cache.build(b,jnp.array(1),jax.random.PRNGKey(9),jax.random.PRNGKey(10))
    exact=ResolvedVarianceBasis(basis.lower,basis.spacing)
    query=jnp.array([[.23,.31],[.241,.319],[.63,.79]])
    wanted=np.sqrt(np.asarray(b.predict(query)[1])[:,0,0])
    np.testing.assert_allclose(exact.evaluate(query,tables.std)[:,0],wanted,atol=1e-10)
    # The change resolves only variance. Existing mean/field interpolation is
    # byte-for-byte the same function for ordinary array fields.
    np.testing.assert_array_equal(exact.evaluate(query,tables.mean),basis.evaluate(query,tables.mean))
    old=CachedFields(points,shape,model.settings(),points)
    old_tables=old.build(b,jnp.array(1),jax.random.PRNGKey(9),jax.random.PRNGKey(10))
    np.testing.assert_array_equal(tables.planning,old_tables.planning)
    np.testing.assert_array_equal(tables.safety,old_tables.safety)
    np.testing.assert_array_equal(tables.std.grid,old_tables.std)


def test_exact_variance_tracks_each_actual_update_without_changing_other_positions_to_zero():
    b=belief();basis,points,shape=model.table_grid(jnp.array([[0.,0.],[1.,1.]]),.1)
    cache=CachedFields(points,shape,model.settings(),points,exact_variance=True)
    tables=cache.build(b,jnp.array(0),jax.random.PRNGKey(1),jax.random.PRNGKey(2))
    p=jnp.array([[.23,.31]]);b,_=condition(b,p,jnp.zeros((1,2)),jnp.zeros(len(b.mean),bool),observation_first=True)
    refreshed=cache.safety(b,tables,jax.random.PRNGKey(2))
    assert float(refreshed.std.evaluate(p)[0,0]) < float(tables.std.evaluate(p)[0,0])/5
    assert float(refreshed.std.evaluate(jnp.array([[.9,.9]]))[0,0]) > .5


def test_route_handoff_uses_full_posterior_and_requires_unmeasured_part_to_be_learned():
    from types import SimpleNamespace
    from reachable_learning import ReachableLearning
    from exact_poam_variance import route_standard_deviation
    # Use represented sites to isolate the stopping-rule check from the
    # conditional residual of a deliberately tiny nine-inducing-point model.
    b=belief();query=jnp.array([[0.,0.],[.8,.8]])
    basis,points,shape=model.table_grid(jnp.array([[0.,0.],[1.,1.]]),.1)
    cache=CachedFields(points,shape,model.settings(),points,exact_variance=True)
    learner=ReachableLearning.__new__(ReachableLearning)
    learner.policy=SimpleNamespace(nodes=np.asarray(query),target=1)
    learner.route_current_ids=np.array([0,1]);learner.route_handoff_ids=None
    learner.route_goal=None;learner.vector_precision=.2
    # An old coarse marginal table does not define the direct-query gate.
    learner.moments=lambda mean,std:(basis.evaluate(query,mean),basis.evaluate(query,std.grid)[:,0])
    mask=jnp.zeros(len(b.mean),bool)
    b,mask=condition(b,query[:1],jnp.zeros((1,2)),mask,observation_first=True)
    tables=cache.build(b,jnp.array(1),jax.random.PRNGKey(1),jax.random.PRNGKey(2))
    metric=learner.route_metric(tables)
    assert not metric['handoff_route_selected']
    for _ in range(3):b,mask=condition(b,query,jnp.zeros((2,2)),mask,observation_first=True)
    tables=cache.safety(b,tables,jax.random.PRNGKey(2))
    metric=learner.route_metric(tables)
    expected=np.sqrt(np.asarray(b.predict(query)[1])[:,0,0])
    np.testing.assert_allclose(metric['std'],expected,atol=1e-10)
    assert metric['handoff_route_selected'] and metric['vector_std_m_s']<=.2
    repeated=np.tile(np.asarray(query),(70,1))
    np.testing.assert_allclose(route_standard_deviation(tables.std,repeated),np.tile(expected,70),atol=1e-10)
