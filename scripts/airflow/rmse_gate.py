"""Simulation-only phase decision. Truth is not returned to planning/control."""
import numpy as np


def wind_vector_rmse(estimate,true_wind,free_indices=None):
    estimate=np.asarray(estimate);true_wind=np.asarray(true_wind)
    if estimate.shape!=true_wind.shape or estimate.shape[-1]!=2:
        raise ValueError('Matching 2D vector maps are required')
    error=estimate-true_wind
    if free_indices is not None:error=error[free_indices]
    return float(np.sqrt(np.mean(np.sum(error**2,axis=-1))))


def ready_for_mbd(rmse,threshold=.2):
    return bool(np.isfinite(rmse) and 0<=rmse<=threshold)
