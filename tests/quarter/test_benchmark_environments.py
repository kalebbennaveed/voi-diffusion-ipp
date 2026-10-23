"""Check benchmark task requirements, without asserting favorable outcomes."""
import numpy as np
from voi_diffusion_ipp.quarter.benchmark_environments import SCENARIOS,make_environment,flow,flow_parameters
from voi_diffusion_ipp.quarter.rrt import segment_free
from voi_diffusion_ipp.quarter.modes import visibility_paths
from voi_diffusion_ipp.quarter.environment import environment,wind


def test_original_pair_preserves_geometry_and_opposite_disturbance():
    a,b=map(make_environment,['original','reversed'])
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    old=environment(geometry='compact',case='forward')
    np.testing.assert_array_equal(a.rectangles[:2],old.rectangles)
    # Duplicating equally weighted cells leaves the original task unchanged.
    np.testing.assert_allclose(np.unique(a.cells,axis=0),np.unique(old.cells,axis=0),atol=1e-6)
    p=np.random.default_rng(7).uniform([-6,-4],[6,6],(500,2)).astype(np.float32)
    f=flow(p,flow_parameters(SCENARIOS[0]))
    np.testing.assert_allclose(f,wind(p,'forward',3.65,geometry='compact'),atol=1e-6)
    np.testing.assert_array_equal(f,-flow(p,flow_parameters(SCENARIOS[1])))


def test_new_tasks_are_structurally_distinct_and_complex():
    assert len(SCENARIOS)==10
    assert all(len(s.rectangles)>=6 for s in SCENARIOS[2:])
    assert all(len(s.patches)>=4 for s in SCENARIOS[2:])
    assert len({s.rectangles for s in SCENARIOS[2:]})==8
    assert len({s.patches for s in SCENARIOS[2:]})==8
    assert len({s.topology for s in SCENARIOS[2:]})==8


def test_targets_and_initializer_goals_are_free_and_reachable():
    for s in SCENARIOS:
        env=make_environment(s.name);r=np.asarray(env.rectangles)[:,:4];b=np.asarray(env.bounds)
        assert env.cells.shape==(168,2) and env.rectangles.shape==(8,5)
        assert all(segment_free(p,p,r,b,0.) for p in np.asarray(env.cells)),s.name
        goals=np.array([(np.array(p[:2])+p[2:])/2 for p in s.patches])
        assert all(segment_free(g,g,r,b,.6) for g in goals),s.name
        for g in goals[1:]:assert visibility_paths(goals[0],g,r,b,.6,1),s.name


def test_disturbance_intersects_required_geometric_transits():
    for s in SCENARIOS:
        e=make_environment(s.name);r=np.asarray(e.rectangles)[:,:4];b=np.asarray(e.bounds)
        goals=np.array([(np.array(p[:2])+p[2:])/2 for p in s.patches]);points=[]
        for goal in goals[1:]:
            for path in visibility_paths(goals[0],goal,r,b,.6,1):
                for x,y in zip(path[:-1],path[1:]):points.extend(np.linspace(x,y,20))
        f=np.asarray(flow(np.array(points),flow_parameters(s)))
        assert np.mean(np.linalg.norm(f,axis=1)>1.5)>.03,s.name


def test_multisite_exact_reversal_stops_without_nonfinite_parameters():
    from voi_diffusion_ipp.quarter.continuous import ControlSettings,nominal_reference
    from voi_diffusion_ipp.quarter.modes import parameterize
    cfg=ControlSettings(dt=.25,horizon=16,commit=5,geometry='suite',initialization_effort=.75)
    state=np.array([-5.,1.,0.,0.])
    theta,info=parameterize(np.array([[-5.,1.],[-3.,1.],[-5.,1.]]),state,cfg,2.8,
        np.empty((0,4)),np.array([[-8.,-8.],[8.,8.]]),.6)
    assert np.isfinite(theta).all()
    reference=np.asarray(nominal_reference(theta,state,cfg)[0])
    np.testing.assert_allclose(reference[-1,:2],state[:2],atol=.01)
    np.testing.assert_allclose(reference[-1,2:],0,atol=.01)
    assert info['reached_goal']
