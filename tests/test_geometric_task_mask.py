import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import numpy as np
from geometric_task_mask import candidate_path_weights
from task_wind_acquisition import expected_variance_reduction


def test_transit_has_value_but_unrelated_uncertainty_does_not():
    # Only the destination needs monitoring; the connecting passage still
    # matters to the task. An equally uncertain unrelated corner must not.
    nodes=np.array([[0.,0.],[1.,0.],[2.,0.],[1.,1.]])
    path=np.array([[0.,0.],[2.,0.]])
    weights,_=candidate_path_weights(nodes,[path],np.array([[2.,0.]]),np.array([1.]),np.arange(4))
    assert weights[1]>0 and weights[-1]==0
    gains=np.asarray(expected_variance_reduction(np.eye(4),np.ones(4),weights,.01))
    assert gains[1]>0 and gains[-1]==0
    # A remote observation can still be useful through actual GP correlation.
    correlated=np.eye(4);correlated[1,-1]=.8
    gain=np.asarray(expected_variance_reduction(correlated,np.ones(4),weights,.01))
    assert gain[-1]>0


def test_path_mask_is_independent_of_polyline_sampling_and_duplicate_seeds():
    xx,yy=np.meshgrid(np.linspace(0,2,21),np.linspace(-.5,.5,11))
    nodes=np.c_[xx.ravel(),yy.ravel()];free=np.arange(len(nodes))
    args=(np.array([[2.,0.]]),np.array([1.]),free)
    a=np.array([[0.,0.],[2.,0.]])
    b=np.array([[0.,0.],[.4,0.],[.4,0.],[1.4,0.],[2.,0.]])
    wa,_=candidate_path_weights(nodes,[a],*args)
    wb,_=candidate_path_weights(nodes,[b,b.copy()],*args)
    np.testing.assert_allclose(wa,wb,atol=1e-15)


def test_deficit_selects_routes_without_erasing_their_transit_or_obstacles():
    nodes=np.array([[0.,0.],[1.,0.],[2.,0.],[0.,1.],[0.,2.],[9.,9.]])
    paths=[np.array([[0.,0.],[2.,0.]]),np.array([[0.,0.],[0.,2.]])]
    cells=np.array([[2.,0.],[0.,2.]])
    w,_=candidate_path_weights(nodes,paths,cells,np.array([1.,.01]),np.arange(5))
    assert w[1]>50*w[3] and w[-1]==0
    assert np.isclose(w.sum(),1)
    zero,_=candidate_path_weights(nodes,paths,cells,np.zeros(2),np.arange(5))
    assert not zero.any()
