"""Declared model-support test for a planned trajectory, separate from RPCBF.

Bound the error in the unchanged quadratic wind acceleration throughout the
same local 3-sigma confidence disk used by execution. A finite trajectory test
is conditional on that disk's calibration; it is not a new safety guarantee.
"""
import jax.numpy as jnp


def acceleration_error_bound(mean,std,velocity):
    radius=3*jnp.sqrt(jnp.maximum(std,0.)**2+.1**2)
    relative=jnp.linalg.norm(mean-velocity,axis=-1)
    return (.1/.680)*(2*relative*radius+radius**2)


def trajectory_error_bound(states,tables,basis):
    shape=states.shape[:-1];flat=states.reshape(-1,4)
    mean=basis.evaluate(flat[:,:2],tables.mean)
    std=basis.evaluate(flat[:,:2],tables.std)[:,0]
    return acceleration_error_bound(mean,std,flat[:,2:]).reshape(shape)


def authority_margin(error,controls,umax):
    """Remaining component-wise authority after the vector error enclosure."""
    return umax-jnp.max(jnp.abs(controls),axis=-1)-error


def reference_support(theta,x,tables,basis,cfg,reference_fn):
    states,accelerations=reference_fn(theta,x,cfg)
    error=trajectory_error_bound(states,tables,basis)
    mean=basis.evaluate(states[:-1,:2],tables.mean);relative=mean-states[:-1,2:]
    force=(.1/.680)*jnp.linalg.norm(relative,axis=-1,keepdims=True)*relative
    nominal=accelerations-force
    return (jnp.max(error),jnp.min(authority_margin(error[:-1],nominal,cfg.umax)),
            cfg.umax-jnp.max(jnp.abs(nominal)))


def reference_information_weights(states,mean,variance,need,cells,sensor_sigma,drag=.1/.680,downstream=False):
    """Declared acquisition proxy, not the MBD monitoring objective.

    Sensing value multiplies E||d acceleration / d wind||_F². For quadratic
    drag in two dimensions, this sensitivity is exactly 5 k² E||w-v||².
    Variance reduction is then evaluated with the full GP at these reference
    positions. No expected measurements are assimilated.
    """
    sensing=jnp.exp(-jnp.sum((states[:,None,:2]-cells[None])**2,axis=-1)/sensor_sigma**2)
    value=sensing@need
    if downstream:
        # A transit location can matter because it precedes useful sensing,
        # even when that location itself has no clarity deficit. This is a
        # discrete downstream sensing-value proxy, not an exact cost adjoint.
        value=jnp.cumsum(value[::-1])[::-1]
    sensitivity=5*drag**2*(jnp.sum((mean-states[:,2:])**2,axis=-1)+2*variance)
    weights=jnp.maximum(value*sensitivity,0.)
    return weights/jnp.maximum(jnp.sum(weights),1e-30)
