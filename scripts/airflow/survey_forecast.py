"""Execution-resolution forecasting with an optional original coarse tail.

The optimized reference still has one independent acceleration pair per .25 s.
Within ``fine_seconds``, tracking and hard RPCBF run every .05 s, and the
unchanged plant, contact and clarity updates run every .01 s. Final validation
uses the full horizon; the default committed prefix is retained for diagnostics.
Any optional remaining tail uses the original .25 s forecast discretization.
The belief stays frozen: no hypothetical observation updates occur here.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

import legacy_monitoring as legacy
import monitoring_mbd as m
from voi_diffusion_ipp.safety.flow_policy_value import filter_control
from voi_diffusion_ipp.wind.cache import TableMean
from voi_diffusion_ipp.wind.gp import InnerFields, WorldField


class FineForecast(NamedTuple):
    rollout: object
    prefix_max_slack: jax.Array
    prefix_infeasible_steps: jax.Array
    observation_positions: jax.Array
    prefix_states: jax.Array
    commit_monitor: jax.Array
    commit_penalty: jax.Array
    full_observation_positions: jax.Array | None

    @property
    def commit_cost(self):
        """Exactly accumulated monitoring and penalty cost before commitment ends."""
        return self.commit_monitor+self.commit_penalty


def sensing_coefficients(p0, p1, cells, sigma, sensor_R):
    """The same three spatial RK4 factors, evaluated together along a path."""
    positions = jnp.stack((p0, (p0+p1)/2, p1), axis=-2)
    return jax.vmap(jax.vmap(lambda p: m.clarity.sensing(p, cells, sigma)**2/sensor_R))(positions)


def clarity_step_from_coefficients(q, coefficients, process_noise, dt):
    """Algebraically identical to clarity.step; only sensing is precomputed."""
    rate = lambda qq, factor: factor*(1-qq)**2-process_noise*qq**2
    c0, cm, c1 = coefficients
    k1 = rate(q, c0)
    k2 = rate(q+dt*k1/2, cm)
    k3 = rate(q+dt*k2/2, cm)
    k4 = rate(q+dt*k3, c1)
    return jnp.clip(q+dt*(k1+2*k2+2*k3+k4)/6, 0, 1)


def forecast(theta, x0, q0, mean_array, construction, physical, env, cfg, *, fine_seconds=None):
    """Full-model forecast refined through the requested fine interval.

    ``rollout.x/q`` retain the original reference-grid shape for diagnostics;
    ``rollout.u`` samples the first applied control of each reference interval.
    It must not be treated as a held control over the refined interval. The
    information objective uses the separately returned actual .2 s sensor
    positions, rather than interpolating those subsampled diagnostic arrays.
    ``observation_positions`` always covers the committed interval. Exact
    ``full_observation_positions`` is available only when the full horizon is
    refined; mixed prefix/tail calls return None for that field.
    """
    execution_dt, physics_dt, observation_dt = .05, .01, .2
    fine_seconds = cfg.commit if fine_seconds is None else fine_seconds
    if not cfg.commit <= fine_seconds <= cfg.horizon:
        raise ValueError('Fine integration must cover the commit interval and stay within the horizon')
    prefix_steps = round(fine_seconds / execution_dt)
    committed_steps = round(cfg.commit / execution_dt)
    fine_reference_steps = round(fine_seconds/cfg.dt)
    reference_stride = round(cfg.dt / execution_dt)
    observation_stride = round(observation_dt / execution_dt)
    if (abs(fine_seconds / cfg.dt-fine_reference_steps) > 1e-7 or
            abs(cfg.commit / cfg.dt-round(cfg.commit/cfg.dt)) > 1e-7 or
            abs(cfg.commit / observation_dt-round(cfg.commit/observation_dt)) > 1e-7 or
            abs(cfg.dt/execution_dt-reference_stride) > 1e-7):
        raise ValueError('Commit/reference/observation intervals must align with the .05 s execution grid')
    mean = TableMean(legacy.BASIS, mean_array)
    inner = InnerFields(legacy.BASIS, construction)
    world = WorldField(legacy.BASIS, physical)
    reference, aa = legacy.old.nominal_reference(theta, x0, cfg)
    physics_steps = round(execution_dt/physics_dt)

    def penalty_at(state):
        value = cfg.violation_weight*jax.nn.relu(cfg.planning_clearance-env.clearance(state))**2
        return value+cfg.speed_penalty*jax.nn.relu(jnp.linalg.norm(state[2:])-cfg.speed_max)**2

    def prefix_step(carry, k):
        x, penalty, max_slack, gap, alive, infeasible = carry
        index = k // reference_stride
        offset = (k % reference_stride)*execution_dt
        ref, a = reference[index], aa[index]
        target = jnp.r_[ref[:2]+offset*ref[2:]+.5*offset**2*a, ref[2:]+offset*a]
        nominal = m.tracking_control(x, target, a, mean, cfg, True)
        u, info = filter_control(x, nominal, mean, inner, env, cfg)
        # Motion is independent of clarity. Retain every .01 s segment and
        # exact pre-contact sensing fraction from the executed advance model.
        def motion_tick(state, _):
            z, living, gap = state
            zn = m.physical.rk4(z, u, world, physics_dt)
            fraction = m.contact_fraction(z[:2], zn[:2], env)
            hit = living & (fraction <= 1)
            f = jnp.minimum(fraction, 1.)
            end = z+f*(zn-z)
            end = jnp.where(hit, jnp.r_[end[:2], jnp.zeros(2, dtype=z.dtype)], end)
            end = jnp.where(living, end, z)
            gap = jnp.minimum(gap, jnp.where(hit, 0., env.clearance(end)))
            return (end, living & ~hit, gap), (end[:2], jnp.where(living, f, 0.))
        (xn, survived, new_gap), (positions, fractions) = jax.lax.scan(
            motion_tick, (x, alive, env.clearance(x)), None, length=physics_steps)
        updated = (xn, penalty+execution_dt*penalty_at(xn)*alive,
                   jnp.maximum(max_slack, jnp.where(alive, info.slack, 0.)),
                   jnp.minimum(gap, new_gap), survived,
                   infeasible+(alive & ~info.feasible).astype(jnp.int32))
        return updated, (xn, jnp.where(alive, u, jnp.zeros_like(u)), positions, fractions,
                         updated[2], updated[5], updated[1])

    zero = jnp.array(0., dtype=x0.dtype)
    initial = (x0, zero, zero, env.clearance(x0),
               jnp.array(True), jnp.array(0, jnp.int32))
    motion_end, (prefix_x, prefix_u, segments, fractions, slacks, infeasibilities, penalties) = jax.lax.scan(
        prefix_step, initial, jnp.arange(prefix_steps))
    path = jnp.concatenate((x0[None, :2], segments.reshape((-1, 2))))
    coefficients = sensing_coefficients(path[:-1], path[1:], env.cells,
                                        cfg.sensor_sigma, cfg.sensor_R)
    coefficients = coefficients.reshape((prefix_steps, physics_steps, 3, len(env.cells)))

    def clarity_period(carry, inputs):
        q, total = carry
        factors, fractions = inputs
        def tick(carry, inputs):
            q, cost = carry
            coeff, f = inputs
            sensed = clarity_step_from_coefficients(q, coeff, env.process_noise, physics_dt*f)
            qn = sensed/(1+physics_dt*(1-f)*env.process_noise*sensed)
            incurred = .5*physics_dt*(m.clarity.deficit(q, env.targets)+m.clarity.deficit(qn, env.targets))
            return (qn, cost+incurred), None
        (qn, incurred), _ = jax.lax.scan(tick, (q, zero), (factors, fractions))
        return (qn, total+incurred), (qn, total+incurred)
    (prefix_q_end, prefix_monitor), (prefix_q, monitoring_costs) = jax.lax.scan(
        clarity_period, (q0, zero), (coefficients, fractions))
    prefix_end = (motion_end[0], prefix_q_end, prefix_monitor, motion_end[1],
                  motion_end[2], motion_end[3], motion_end[4], motion_end[5])

    def tail_step(carry, inputs):
        x, q, monitor, penalty, max_slack, gap, alive, infeasible = carry
        ref, a = inputs
        nominal = m.tracking_control(x, ref, a, mean, cfg, True)
        u, info = filter_control(x, nominal, mean, inner, env, cfg)
        xn = m.physical.rk4(x, u, world, cfg.dt)
        hit = m.contact_fraction(x[:2], xn[:2], env) <= 1
        observing = alive & ~hit
        xn = jnp.where(observing, xn, jnp.r_[x[:2], jnp.zeros(2, dtype=x.dtype)])
        qn = m.clarity.step(q, x[:2], xn[:2], env.cells, cfg.sensor_sigma,
                            cfg.sensor_R, env.process_noise, cfg.dt)
        qn = jnp.where(observing, qn, q/(1+cfg.dt*env.process_noise*q))
        incurred = cfg.dt*.5*(m.clarity.deficit(q, env.targets)+m.clarity.deficit(qn, env.targets))
        new_gap = jnp.where(alive, jnp.minimum(gap, jnp.where(hit, 0., env.clearance(xn))), gap)
        updated = (xn, qn, monitor+incurred, penalty+cfg.dt*penalty_at(xn)*alive,
                   jnp.maximum(max_slack, jnp.where(alive, info.slack, 0.)), new_gap, observing, infeasible)
        return updated, (xn, qn, jnp.where(alive, u, jnp.zeros_like(u)))

    end, (tail_x, tail_q, tail_u) = jax.lax.scan(
        tail_step, prefix_end, (reference[fine_reference_steps:-1], aa[fine_reference_steps:]))
    xs = jnp.concatenate((x0[None], prefix_x[reference_stride-1::reference_stride], tail_x))
    qs = jnp.concatenate((q0[None], prefix_q[reference_stride-1::reference_stride], tail_q))
    us = jnp.concatenate((prefix_u[::reference_stride], tail_u))
    rollout = legacy.old.Rollout(xs, qs, us, reference, end[2], end[3], end[4], end[5], ~end[6])
    full_positions = (prefix_x[observation_stride-1::observation_stride, :2]
                      if fine_seconds == cfg.horizon else None)
    return FineForecast(rollout, slacks[committed_steps-1], infeasibilities[committed_steps-1],
                        prefix_x[observation_stride-1:committed_steps:observation_stride, :2],
                        jnp.concatenate((x0[None], prefix_x[:committed_steps])),
                        monitoring_costs[committed_steps-1], penalties[committed_steps-1], full_positions)
