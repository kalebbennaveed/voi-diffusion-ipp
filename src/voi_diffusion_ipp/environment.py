from typing import NamedTuple
import jax
import jax.numpy as jnp


class Environment(NamedTuple):
    cells: jax.Array
    targets: jax.Array
    obstacles: jax.Array  # x,y,r
    bounds: jax.Array  # lower xy, upper xy
    roi_a: jax.Array
    roi_b: jax.Array
    probe: jax.Array

    def violations(self, x, margin=0.):
        p = x[:2]
        # Smooth squared circle constraints, positive in failure.
        obs = (self.obstacles[:, 2]+margin)**2 - jnp.sum((p-self.obstacles[:,:2])**2, axis=1)
        wall = jnp.concatenate((self.bounds[0]+margin-p, p-self.bounds[1]+margin))
        return jnp.concatenate((obs, wall))

    def clearance(self, x):
        p = x[:2]
        d = jnp.sqrt(jnp.sum((p-self.obstacles[:,:2])**2, axis=1)) - self.obstacles[:,2]
        return jnp.min(jnp.concatenate((d, p-self.bounds[0], self.bounds[1]-p)))
