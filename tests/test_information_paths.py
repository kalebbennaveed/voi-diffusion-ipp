import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import full_learning_monitoring as m
from information_paths import batch_variance_reduction,prefix_points,usable_prefix
from voi_diffusion_ipp.environment import Environment
from monitoring_routes import coarse_paths


def test_batch_information_gain_matches_independent_gaussian_conditioning():
    rng=np.random.default_rng(82);A=rng.normal(size=(9,9));C=A@A.T+.01*np.eye(9)
    cross=C[:6,6:];obs=C[6:,6:];weights=np.arange(1,7,dtype=float);weights/=weights.sum()
    posterior=C[:6,:6]-cross@np.linalg.solve(obs+(.01+1e-9)*np.eye(3),cross.T)
    expected=weights@(np.diag(C[:6,:6])-np.diag(posterior))
    np.testing.assert_allclose(batch_variance_reduction(cross,obs,weights,.01),expected,rtol=1e-10)


def test_repeated_observations_have_diminishing_information_and_zero_weights_have_none():
    weights=np.array([1.]);one=batch_variance_reduction(jnp.ones((1,1)),jnp.ones((1,1)),weights,.01)
    two=batch_variance_reduction(jnp.ones((1,2)),jnp.ones((2,2)),weights,.01)
    assert one<two<2*one
    assert float(batch_variance_reduction(jnp.ones((1,2)),jnp.ones((2,2)),np.zeros(1),.01))==0


def test_frontier_stops_before_first_unsupported_segment_and_retains_corners():
    p,padded=prefix_points(np.array([[0.,0.],[1.,0.],[1.,1.]]),spacing=.1,maximum=30)
    limit=np.zeros(30);limit[15:]=2.
    out=usable_prefix(p,limit,1.2)
    np.testing.assert_allclose(out,[[0.,0.],[1.,0.],[1.,.4]],atol=1e-12)
    np.testing.assert_allclose(padded[-1],[1.,1.])
    np.testing.assert_allclose(usable_prefix(p,np.full(30,2.),1.2),[[0.,0.]])


def test_admissible_graph_cannot_jump_to_a_disconnected_island():
    import cluster_environment as ce
    cfg=m.settings();env,grid,_=ce.environment(m.legacy.physical_settings(cfg),
        m.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
    nodes=np.asarray(grid['all_cells']);x=np.array([1.45,.875,0.,0.]);q=np.full(len(env.cells),.74)
    local=(nodes[:,0]>1)&(nodes[:,0]<2)&(nodes[:,1]>.5)&(nodes[:,1]<1.5)
    island=(nodes[:,0]>6)&(nodes[:,0]<7)&(nodes[:,1]>3)&(nodes[:,1]<4)
    paths,_=coarse_paths(x,q,env,cfg,admissible_nodes=local|island)
    assert len(paths)==6
    for path in paths:
        assert np.all((path[:,0]>=1)&(path[:,0]<=2)&(path[:,1]>=.5)&(path[:,1]<=1.5))
    paths,_=coarse_paths(x,q,env,cfg,admissible_nodes=island)
    for path in paths:np.testing.assert_allclose(path,x[None,:2])


def test_motion_only_forecast_preserves_existing_execution_resolution():
    env=Environment(jnp.array([[1.,1.],[2.,1.]]),jnp.full(2,.7),jnp.array([[3.,3.,.3]]),
        jnp.array([[0.,0.],[4.,4.]]),jnp.zeros(2),jnp.zeros(2),jnp.zeros(2),jnp.full(2,.02))
    cfg=replace(m.settings(),horizon=.5,commit=.5);basis,points,shape=m.table_grid(env.bounds,.2)
    mu=jnp.stack((.2+.01*points[:,0],-.1+.01*points[:,1]),axis=1).reshape(shape)
    fields=jnp.stack((mu+.05,mu-.05));tables=m.Tables(mu,jnp.full((*shape[:2],1),.03),jnp.ones((*shape[:2],1)),fields,fields,fields)
    x=jnp.array([1.4,1.1,.1,.05]);q=jnp.full(2,.6);theta=jnp.zeros(2*cfg.steps)
    full=jax.jit(lambda: m.core.execution_forecast(theta,x,q,tables,mu,basis,env,cfg,True,control_fn=m.control))()
    motion=jax.jit(lambda: m.core.execution_forecast(theta,x,q,tables,mu,basis,env,cfg,True,control_fn=m.control,motion_only=True))()
    dense=jax.jit(lambda: m.core.execution_forecast(theta,x,q,tables,mu,basis,env,cfg,True,control_fn=m.control,return_dense=True))()
    np.testing.assert_allclose(dense.x,motion[0],atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(dense.q[::5],full.q,atol=1e-10,rtol=1e-10)
    for key in ['monitor','penalty','max_slack','min_clearance']:
        np.testing.assert_allclose(getattr(dense,key),getattr(full,key),atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(motion[0][::5],full.x,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(motion[1],full.u,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(motion[2],full.max_slack,atol=1e-10)
    np.testing.assert_allclose(motion[3],full.min_clearance,atol=1e-10)
    assert bool(motion[4])==bool(full.terminated)
