"""Original reset/replay candidate-selection heuristic, with MBD-generated candidates.

The source draft uses Stein candidate generation. Here passive MBD supplies the
library so the simulator and optimizer dependencies stay common; the original
positive-clipped, noncausal reset/replay selection rule is retained explicitly.
"""

import jax
import jax.numpy as jnp

from ..planning.two_stage import objective


def select_candidates(result, x, q, belief, buffers, env, cfg):
    library = jnp.concatenate((result.iterates, result.vector[None]))
    costs = jax.lax.map(
        lambda y: objective(y, x, q, belief, buffers, env, cfg, "heuristic"), library
    )
    i = jnp.argmin(jnp.where(jnp.isfinite(costs), costs, jnp.inf))
    return result._replace(
        vector=library[i], cost=costs[i], evaluations=result.evaluations + len(library)
    )
