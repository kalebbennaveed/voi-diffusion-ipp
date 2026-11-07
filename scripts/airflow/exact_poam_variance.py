"""Query the existing full POAM marginal variance at the requested position.

No uncertainty attenuation or posterior update occurs here. Mean and coherent
field interpolation are unchanged. The variance query avoids interpolating
standard deviations, which discards covariance between adjacent grid values.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from voi_diffusion_ipp.wind.poam import attentive_kernel
from voi_diffusion_ipp.quarter.environment import ExtendedTableBasis

class PosteriorStandardDeviation(NamedTuple):
    grid: object
    parameters: object
    centers: object
    lengths: object
    correction: object
    amplitude: object

    def evaluate(self,positions):
        p=jnp.atleast_2d(positions)
        K=attentive_kernel(self.parameters,p,self.centers,self.lengths)
        variance=self.amplitude+jnp.sum((K@self.correction)*K,axis=1)
        return jnp.sqrt(jnp.maximum(variance,0.))[:,None]

class ResolvedVarianceBasis(NamedTuple):
    lower: object
    spacing: object

    def evaluate(self,positions,field):
        if isinstance(field,PosteriorStandardDeviation):return field.evaluate(positions)
        return ExtendedTableBasis(self.lower,self.spacing).evaluate(positions,field)

def grid_standard_deviation(field):
    """The coarse initializer may use the old grid; execution never does."""
    return field.grid if isinstance(field,PosteriorStandardDeviation) else field

@jax.jit
def _query_standard_deviation(field,points):
    return field.evaluate(points)[:,0]

def route_standard_deviation(field,points):
    """Exact route queries in fixed batches, avoiding shape-specific recompiles."""
    points=np.asarray(points);parts=[];size=128
    for start in range(0,len(points),size):
        chunk=points[start:start+size];padded=np.pad(chunk,((0,size-len(chunk)),(0,0)),mode='edge')
        parts.append(np.asarray(_query_standard_deviation(field,jnp.asarray(padded)))[:len(chunk)])
    return np.concatenate(parts) if parts else np.empty(0)
