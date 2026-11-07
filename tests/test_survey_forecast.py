"""Refined prefix agrees with the unchanged executed plant and clarity model."""
from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import legacy_monitoring as legacy
import monitoring_mbd as m
from survey_forecast import forecast, sensing_coefficients, clarity_step_from_coefficients
from voi_diffusion_ipp.wind.cache import TableMean
from voi_diffusion_ipp.wind.gp import InnerFields, WorldField


def compare_executed_prefix(x, wind, horizon=1.):
    cfg = replace(legacy.Settings(), horizon=horizon, commit=1., construction_worlds=8,
                  safety_margin=.12, prediction_filter='hard')
    pcfg = replace(legacy.physical_settings(cfg), safety_margin=.12)
    env, _ = m.environment(pcfg)
    q = jnp.full(len(env.cells), .74)
    theta = jnp.linspace(-.12, .17, cfg.steps*2)
    table = jnp.broadcast_to(jnp.array(wind), legacy.SHAPE)
    construction = jnp.broadcast_to(table, (8, *legacy.SHAPE))
    refined = jax.jit(lambda th, xx, qq: forecast(th, xx, qq, table, construction,
                                               table, env, cfg, fine_seconds=horizon))(theta, x, q)
    mean = TableMean(legacy.BASIS, table)
    inner = InnerFields(legacy.BASIS, construction)
    world = WorldField(legacy.BASIS, table)
    execute = m.execution_kernel(env, pcfg)
    refs, aa = legacy.old.nominal_reference(theta, x, cfg)
    states, clarities, cost, max_slack, infeasible = [np.asarray(x)], [np.asarray(q)], 0., 0., 0
    commit_monitor, commit_penalty = 0., 0.
    alive = True
    for k in range(round(horizon/.05)):
        index, off = k//5, (k % 5)*.05
        ref, a = refs[index], aa[index]
        reference = jnp.r_[ref[:2]+off*ref[2:]+.5*off*off*a, ref[2:]+off*a]
        nominal = m.tracking_control(x, reference, a, mean, pcfg, True)
        end, _, info = execute(x, q, nominal, mean, inner, world, jnp.asarray(alive))
        # Preserve a non-finite diagnostic from an already infeasible active
        # state rather than silently discarding it through Python max(0,nan).
        if alive and k < 20:
            max_slack = np.maximum(max_slack, float(info.slack))
        infeasible += int(alive and k < 20 and not bool(info.feasible))
        was_alive = bool(alive)
        x, q, alive, incurred, _, _ = end
        cost += float(incurred)
        if k < 20:
            commit_monitor += float(incurred)
            penalty = cfg.violation_weight*max(cfg.planning_clearance-float(env.clearance(x)), 0.)**2
            penalty += cfg.speed_penalty*max(float(jnp.linalg.norm(x[2:]))-cfg.speed_max, 0.)**2
            commit_penalty += .05*penalty*was_alive
        states.append(np.asarray(x)); clarities.append(np.asarray(q))
    np.testing.assert_allclose(refined.prefix_states, states[:21], atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(refined.rollout.x, np.asarray(states)[::5], atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(refined.rollout.q, np.asarray(clarities)[::5], atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(refined.observation_positions, np.asarray(states)[4:21:4, :2], atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(refined.full_observation_positions, np.asarray(states)[4::4, :2], atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(refined.commit_monitor, commit_monitor, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(refined.commit_penalty, commit_penalty, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(refined.commit_cost, commit_monitor+commit_penalty, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(refined.rollout.monitor, cost, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(refined.prefix_max_slack, max_slack, atol=1e-10, rtol=1e-10)
    assert int(refined.prefix_infeasible_steps) == infeasible
    assert bool(refined.rollout.terminated) == (not bool(alive))
    np.testing.assert_array_equal(refined.rollout.nominal, refs)
    return refined


def test_prefix_matches_execution_state_clarity_cost_and_sensor_positions():
    result = compare_executed_prefix(jnp.array([1.45, .875, .1, -.05]), [.7, -.3])
    assert not bool(result.rollout.terminated)
    assert result.rollout.x.shape == (5, 4)
    assert result.observation_positions.shape == (5, 2)
    assert result.full_observation_positions.shape == (5, 2)


def test_fractional_contact_and_subsequent_clarity_decay_match_execution():
    result = compare_executed_prefix(jnp.array([.002, .8, -2., 0.]), [-1., 0.])
    assert bool(result.rollout.terminated)
    np.testing.assert_array_equal(result.prefix_states[-1, 2:], [0., 0.])
    assert np.all(np.asarray(result.rollout.q[-1]) <= np.asarray(result.rollout.q[1]))


def test_full_horizon_refinement_retains_only_committed_sensor_locations():
    result = compare_executed_prefix(jnp.array([1.45, .875, .1, -.05]), [.7, -.3], horizon=2.)
    assert result.rollout.x.shape == (9, 4)
    assert result.prefix_states.shape == (21, 4)
    assert result.observation_positions.shape == (5, 2)
    assert result.full_observation_positions.shape == (10, 2)
    assert not np.isclose(float(result.commit_monitor), float(result.rollout.monitor)/2, atol=1e-5)


def test_fifteen_second_forecast_reports_exact_five_second_cost_and_75_sensor_locations():
    cfg = replace(legacy.Settings(), construction_worlds=8, safety_margin=.12, prediction_filter='hard')
    env, _ = m.environment(legacy.physical_settings(cfg))
    x = jnp.array([1.45, .875, .1, -.05])
    q = jnp.full(len(env.cells), .74)
    theta = jnp.zeros(2*cfg.steps)
    table = jnp.broadcast_to(jnp.array([.7, -.3]), legacy.SHAPE)
    construction = jnp.broadcast_to(table, (8, *legacy.SHAPE))
    full = jax.jit(lambda th: forecast(th, x, q, table, construction, table, env, cfg,
                                       fine_seconds=cfg.horizon))(theta)
    prefix_cfg = replace(cfg, horizon=cfg.commit)
    prefix = jax.jit(lambda th: forecast(th, x, q, table, construction, table, env, prefix_cfg,
                                         fine_seconds=prefix_cfg.horizon))(theta[:2*prefix_cfg.steps])
    assert full.full_observation_positions.shape == (75, 2)
    assert full.observation_positions.shape == (25, 2)
    np.testing.assert_allclose(full.observation_positions, full.full_observation_positions[:25], atol=0, rtol=0)
    np.testing.assert_allclose(full.commit_monitor, prefix.rollout.monitor, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(full.commit_penalty, prefix.rollout.penalty, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(full.commit_cost, prefix.rollout.monitor+prefix.rollout.penalty, atol=2e-13, rtol=2e-13)
    assert not np.isclose(float(full.commit_monitor), float(full.rollout.monitor)/3, atol=1e-5)


def test_precomputed_sensing_preserves_every_physics_step_clarity_and_cost():
    # Includes a fractional sensing step followed by absorbing pure decay.
    # The physical sampling interval and every clarity cell are retained.
    env, _ = m.environment(m.Settings())
    path = jnp.c_[jnp.linspace(1., 1.25, 51), .7+.05*jnp.sin(jnp.linspace(0., 2., 51))]
    fractions = jnp.r_[jnp.ones(31), .37, jnp.zeros(18)]
    factors = sensing_coefficients(path[:-1], path[1:], env.cells, .5, .1)
    def compare(precomputed):
        def tick(carry, inputs):
            q, cost = carry
            p0, p1, f, coefficients = inputs
            if precomputed:
                sensed = clarity_step_from_coefficients(q, coefficients, env.process_noise, .01*f)
            else:
                sensed = m.clarity.step(q, p0, p1, env.cells, .5, .1, env.process_noise, .01*f)
            qn = sensed/(1+.01*(1-f)*env.process_noise*sensed)
            cost += .005*(m.clarity.deficit(q, env.targets)+m.clarity.deficit(qn, env.targets))
            return (qn, cost), (qn, cost)
        return jax.lax.scan(tick, (jnp.full(len(env.cells), .74), jnp.array(0.)),
                            (path[:-1], path[1:], fractions, factors))[1]
    exact = jax.jit(lambda: compare(False))()
    reordered = jax.jit(lambda: compare(True))()
    for actual, expected in zip(reordered, exact):
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
