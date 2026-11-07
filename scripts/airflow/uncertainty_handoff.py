"""Deployment-available stopping rule; this API deliberately accepts no truth."""
import numpy as np


def posterior_vector_rms(variance, weights):
    variance=np.asarray(variance,float);weights=np.asarray(weights,float)
    if variance.shape!=weights.shape or not np.isfinite(variance).all() or not np.isfinite(weights).all():
        raise ValueError('Finite matching marginal-variance and weight arrays required')
    if np.any(weights<0) or np.any(variance < -1e-8):
        raise ValueError('Nonnegative variance and weights required')
    if weights.sum()<=0:return None
    return float(np.sqrt(2*np.dot(weights,np.maximum(variance,0))/weights.sum()))


def ready(variance,weights,threshold):
    if not np.isfinite(threshold) or threshold<=0:raise ValueError('Positive uncertainty threshold required')
    value=posterior_vector_rms(variance,weights)
    return value is not None and value<=threshold
