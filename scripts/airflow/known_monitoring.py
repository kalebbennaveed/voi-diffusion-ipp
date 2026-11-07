"""Deterministic known-wind specialization of the original full-control MBD.

The physical field is queried directly: no posterior, sampling, or cached GP
interpolant. One exact field replaces identical copies, not uncertain worlds.
The original multimode reverse diffusion and full RPCBF are retained.
"""
from dataclasses import replace
import jax
import jax.numpy as jnp
import legacy_monitoring as legacy
import monitoring_mbd as m
from voi_diffusion_ipp.quarter import continuous as old
from voi_diffusion_ipp.quarter.multimode import search
from voi_diffusion_ipp.quarter.filter import control as relaxed_control
from voi_diffusion_ipp.safety.flow_policy_value import filter_control
from voi_diffusion_ipp.wind.oracle import KnownWind, KnownBasis


def settings():
    return replace(legacy.Settings(), iterations=2, proposals=14, batch_size=7,
                   worlds=1, validation_worlds=1, construction_worlds=1)


def rollout(theta, x0, q0, field, env, cfg, aware):
    world = field if aware else m.StillAir()
    known = KnownWind(KnownBasis(world))
    inner = known.inner_fields(jnp.zeros((1, 1, 2)))
    reference, aa = old.nominal_reference(theta, x0, cfg)

    def step(carry, inputs):
        x, q, monitor, penalty, max_slack, gap, alive = carry
        ref, a = inputs
        nominal = m.tracking_control(x, ref, a, known, cfg, True)
        if aware:
            u, slack = relaxed_control(x, nominal, known, inner, env, cfg)
        else:
            u, slack = jnp.clip(nominal, -cfg.umax, cfg.umax), jnp.array(0.)
        xn = m.physical.rk4(x, u, world, cfg.dt)
        hit = m.contact_fraction(x[:2], xn[:2], env) <= 1
        observing = alive & ~hit
        xn = jnp.where(observing, xn, jnp.r_[x[:2], jnp.zeros(2)])
        qn = m.clarity.step(q, x[:2], xn[:2], env.cells, cfg.sensor_sigma,
                            cfg.sensor_R, env.process_noise, cfg.dt)
        qn = jnp.where(observing, qn, q/(1+cfg.dt*env.process_noise*q))
        cost = cfg.dt*.5*(m.clarity.deficit(q, env.targets)+m.clarity.deficit(qn, env.targets))
        clearance = env.clearance(xn)
        soft = cfg.violation_weight*jax.nn.relu(cfg.planning_clearance-clearance)**2
        soft += cfg.speed_penalty*jax.nn.relu(jnp.linalg.norm(xn[2:])-cfg.speed_max)**2
        new_gap = jnp.where(alive, jnp.minimum(gap, jnp.where(hit, 0., clearance)), gap)
        return (xn, qn, monitor+cost, penalty+cfg.dt*soft*alive,
                jnp.maximum(max_slack, slack*alive), new_gap, observing), (xn, qn, u)

    initial = (x0, q0, jnp.array(0.), jnp.array(0.), jnp.array(0.), env.clearance(x0), jnp.array(True))
    end, (xs, qs, us) = jax.lax.scan(step, initial, (reference[:-1], aa))
    return old.Rollout(jnp.r_[x0[None], xs], jnp.r_[q0[None], qs], us,
                       reference, end[2], end[3], end[4], end[5], ~end[6])


def optimizer(env, cfg, aware):
    @jax.jit
    def plan(x, q, field, seeds, previous, key):
        def evaluate(theta):
            r = rollout(theta, x, q, field, env, cfg, aware)
            return jnp.atleast_1d(r.monitor+r.penalty)
        result = search(evaluate, seeds, previous, key, cfg,
                        jnp.array(1.), jnp.array(0.), jnp.array(False))
        # With a single known deterministic field, validation is the identical
        # objective. Search.final already minimizes it over all evaluated
        # candidates (including previous and final reverse iterates).
        # Re-evaluate only the selected trajectory to record its forecast.
        forecast = rollout(result.final, x, q, field, env, cfg, aware)
        return result.final, forecast.monitor+forecast.penalty, result, forecast
    return plan


def execution_block(env, cfg, pcfg):
    """Compile an entire commit interval; retain every .05 s control sample.

    This only removes Python dispatch. Full hard RPCBF, .01 s physical/clarity
    integration and swept contact detection are the existing implementations.
    """
    @jax.jit
    def execute(x, q, theta, field, alive):
        known = KnownWind(KnownBasis(field))
        inner = known.inner_fields(jnp.zeros((1, 1, 2)))
        refs, aa = old.nominal_reference(theta, x, cfg)
        def tick(carry, k):
            x, q, alive = carry
            local = k*pcfg.execution_dt
            i = jnp.minimum(jnp.floor((local+1e-8)/cfg.dt).astype(int), cfg.steps-1)
            off = local-i*cfg.dt
            ref, a = refs[i], aa[i]
            reference = jnp.r_[ref[:2]+off*ref[2:]+.5*off*off*a, ref[2:]+off*a]
            nominal = m.tracking_control(x, reference, a, known, pcfg, True)
            u, info = filter_control(x, nominal, known, inner, env, pcfg)
            xn, qn, survived, incurred, gap, hit = m.advance(
                x, q, u, field, env, pcfg, alive, interval=pcfg.execution_dt)
            return (xn, qn, survived), (xn, qn, jnp.where(alive,u,0), nominal,
                jnp.where(alive,info.slack,0), jnp.where(alive,info.correction,0),
                jnp.where(alive,info.feasible,True), gap, incurred, survived,
                jnp.where(alive,hit,jnp.inf))
        return jax.lax.scan(tick, (x, q, alive), jnp.arange(round(cfg.commit/pcfg.execution_dt)))
    return execute
