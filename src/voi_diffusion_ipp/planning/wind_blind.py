"""Explicit baseline: wind-blind nominal MBD, wind-aware safety at execution.

This is NOT the current-belief baseline. The objective predicts still-air,
unfiltered trajectories; known obstacles reject geometric collisions. Actual
execution uses the same full online POAM and continuous sampled spline RPCBF
as the other arms. No actual or hypothetical wind is accepted by this planner.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .budgeted import future_reference, reference_matrices, segment, split_vector
from .mbd import optimize


class StillAir(NamedTuple):
    """Deliberately zero disturbance assumption, never a wind estimator."""

    def at(self, positions):
        return jnp.zeros_like(jnp.atleast_2d(positions))

    def mean_at(self, positions):
        return self.at(positions)

    def predict(self, positions):
        mean = self.at(positions)
        return mean, jnp.zeros((len(mean), 2, 2))


def evaluate_nominal(vector, x, q, env, cfg):
    """Same task, controls, drag, horizon and obstacle geometry; no wind/filter.

    Return a single-world batch to share the saved forecast schema. The six
    posterior-feedback gains are inactive because the baseline has no posterior
    features. Four genuine MBD levels and all proposal evaluations are retained.
    """
    still = StillAir()
    controls, _, _ = split_vector(vector, cfg)
    first = segment(
        x,
        q,
        controls,
        still,
        None,
        still,
        jnp.zeros((cfg.stage_steps, 2)),
        env,
        cfg,
        False,
        nominal_wind_aware=False,
        apply_safety=False,
    )
    matrices = reference_matrices(cfg.future_steps * cfg.dt, cfg.future_steps)
    reference = future_reference(vector, first.x[-1], still, env, matrices, cfg)
    future = segment(
        first.x[-1],
        first.q[-1],
        reference,
        still,
        None,
        still,
        jnp.zeros((cfg.future_steps, 2)),
        env,
        cfg,
        True,
        nominal_wind_aware=False,
        apply_safety=False,
    )
    cost = first.cost + future.cost
    valid = jnp.isfinite(cost)
    if cfg.reject_collisions:
        valid &= (first.clearance.min() >= 0) & (future.clearance.min() >= 0)
    parts = jax.tree.map(lambda value: value[None], (first, future))
    return jnp.where(valid, cost, jnp.inf), parts


def make_nominal_planner(cfg, objective_cells=None):
    @jax.jit
    def plan(x, q, env, initials, key):
        if objective_cells is None:
            qe, ee, scale = q, env, 1.0
        else:
            idx = jnp.asarray(objective_cells)
            qe = q[idx]
            ee = env._replace(
                cells=env.cells[idx], targets=env.targets[idx], process_noise=env.process_noise[idx]
            )
            scale = len(objective_cells) / len(q)
        objective = lambda y: evaluate_nominal(y, x, qe, ee, cfg)[0] * scale
        costs = jax.lax.map(objective, initials)
        result = optimize(objective, initials[jnp.argmin(costs)], key, cfg)
        return result._replace(evaluations=result.evaluations + len(initials))

    return plan
