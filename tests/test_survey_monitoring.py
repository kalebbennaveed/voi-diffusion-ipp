"""Controlled invariants for route task-relevance gating."""
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/airflow'))
from survey_monitoring import (route_tangents, force_relevance, task_information_benefits,
                               route_task_weights, sampled_route_values, stationary_learning_bonus,
                               validation_candidates)


def route_points():
    first = jnp.c_[jnp.arange(8), jnp.zeros(8)]
    second = jnp.c_[jnp.full(8, 20.), jnp.arange(8)]
    return jnp.concatenate((first, second))


def test_route_endpoint_tangents_do_not_jump_to_another_path():
    tangent = route_tangents(route_points())
    np.testing.assert_allclose(tangent[:8], np.tile([.7, 0.], (8, 1)), atol=1e-7)
    np.testing.assert_allclose(tangent[8:], np.tile([0., .7], (8, 1)), atol=1e-7)


def test_zero_wind_force_coupling_has_exactly_zero_survey_relevance():
    points = route_points()
    winds = jax.random.normal(jax.random.PRNGKey(3), (8, len(points), 2)) * 10
    weights = jnp.ones(len(points)) / len(points)
    zero_coupling = lambda velocity, wind, xp: xp.zeros_like(wind)
    gate, relevance = force_relevance(winds, points, weights, force_fn=zero_coupling)
    np.testing.assert_array_equal(gate, np.zeros(len(points)))
    assert float(relevance) == 0
    actual_gate, actual_relevance = force_relevance(winds, points, weights)
    assert float(actual_relevance) > .1
    assert float(actual_gate.sum()) > 0


def test_identical_known_wind_and_zero_task_weights_disable_survey_relevance():
    points = route_points()
    weights = jnp.ones(len(points)) / len(points)
    winds = jnp.ones((8, len(points), 2))
    gate, relevance = force_relevance(winds, points, weights)
    np.testing.assert_array_equal(gate, np.zeros(len(points)))
    assert float(relevance) < 1e-12  # Floating-point mean of identical forces.
    uncertain = jax.random.normal(jax.random.PRNGKey(4), winds.shape) * 10
    gate, relevance = force_relevance(uncertain, points, jnp.zeros(len(points)))
    np.testing.assert_array_equal(gate, np.zeros(len(points)))
    assert float(relevance) == 0


def test_unchanged_task_paths_disable_information_gate_despite_possible_wind_variance():
    # No exploration when perfect knowledge makes no material difference to
    # any of the three task paths. Wind variation is tested independently.
    current = jnp.array([[3., .2, 0.], [2., .3, 0.], [4., .1, 0.]])
    benefits, useful = task_information_benefits(current, current)
    np.testing.assert_array_equal(benefits, np.zeros((3, 3)))
    assert not bool(useful)
    tiny_improvement = current.at[:, 0].add(-.005).at[:, 1].add(-.05)
    _, useful = task_information_benefits(current, tiny_improvement)
    assert not bool(useful)


def test_any_task_path_benefit_can_enable_gate_but_unsafe_counterfactual_cannot():
    current = jnp.array([[3., .2, 0.], [2., .3, 0.], [4., .1, 0.]])
    for row, column, reduction in [(0, 0, .02), (1, 1, .11)]:
        known = current.at[row, column].add(-reduction)
        benefits, useful = task_information_benefits(current, known)
        assert bool(useful)
        assert float(benefits[row, column]) > 0
    contacted = current.at[2, 2].set(1.)
    benefits, useful = task_information_benefits(contacted, current)
    assert bool(useful) and float(benefits[2, 2]) == 1.
    unsafe_known = current.at[:, 0].set(0.).at[:, 1].set(0.).at[:, 2].set(1.)
    benefits, useful = task_information_benefits(current, unsafe_known)
    assert not bool(useful)
    np.testing.assert_array_equal(benefits, np.zeros((3, 3)))


def test_only_the_geometric_route_that_benefits_keeps_information_targets():
    weights = jnp.arange(1., 49.) / jnp.arange(1., 49.).sum()
    # Probe order is direct, lower, upper; geometric target storage uses the
    # original six route groups, so the direct route can be group2, notgroup0.
    route_indices = jnp.array([2, 4, 5])
    benefits = jnp.array([[0., 0., 0.], [0., 0., 0.], [0., .15, 0.]])
    selected = route_task_weights(weights, route_indices, benefits)
    np.testing.assert_array_equal(selected[:40], np.zeros(40))
    np.testing.assert_array_equal(selected[40:], weights[40:])
    none = route_task_weights(weights, route_indices, jnp.zeros((3, 3)))
    np.testing.assert_array_equal(none, np.zeros(48))
    # Adding a benefit for the direct path preserves exactly its own group.
    direct = route_task_weights(weights, route_indices, benefits.at[0, 0].set(.03))
    np.testing.assert_array_equal(direct[16:24], weights[16:24])
    np.testing.assert_array_equal(direct[:16], np.zeros(16))


def test_tracking_only_or_inferior_alternative_gives_no_cost_unit_learning_value():
    uncertain = jnp.array([[5., .8, 0.], [6., 1., 0.], [7., 2., 0.]])
    known = jnp.array([[5., .0, 0.], [5.2, .0, 0.], [5.5, .0, 0.]])
    # Every tracking error improves and individual alternatives improve, but
    # none beats the current best monitoring cost. Survey value must be zero.
    np.testing.assert_array_equal(sampled_route_values(uncertain, known), np.zeros(3))
    known = known.at[1, 0].set(4.8).at[2, 0].set(3.).at[2, 2].set(1.)
    np.testing.assert_allclose(sampled_route_values(uncertain, known), [0., .2, 0.], atol=1e-6)


def test_expensive_learning_loses_for_one_use_but_can_repay_repeated_stationary_uses():
    costs = jnp.array([1., 1.25])  # Monitoring choice and a costly survey detour.
    values = jnp.array([1., .4, 0.])
    fractions = jnp.array([[0., 0., 0.], [.1, .1, 0.]])
    short_bonus = stationary_learning_bonus(values, fractions, 15., 15.)
    assert int(jnp.argmin(costs-short_bonus)) == 0
    long_bonus = stationary_learning_bonus(values, fractions, 75., 15.)
    assert int(jnp.argmin(costs-long_bonus)) == 1
    # Mutually alternative route values are not added together.
    np.testing.assert_allclose(short_bonus, [0., .1], atol=1e-7)
    # Repeated-use credit is explicitly bounded and irrelevant information
    # remains worthless, however long the mission.
    np.testing.assert_allclose(stationary_learning_bonus(values, fractions, 3000., 15.), [0., 2.], atol=1e-6)
    np.testing.assert_array_equal(stationary_learning_bonus(jnp.zeros(3), fractions, 300., 15.), np.zeros(2))


def test_finite_validation_budget_retains_winner_original_hold_and_each_task_passage():
    finalists=jnp.arange(15*120).reshape((15,120))*.01
    seeds=jnp.ones((7,120))*(-.1)
    trained=SimpleNamespace(finalists=finalists,final=finalists[10])
    indices,candidates=validation_candidates(trained,seeds)
    np.testing.assert_array_equal(indices,[10,-1,1,2,3])
    assert candidates.shape==(5,120)
    np.testing.assert_array_equal(candidates[0],trained.final)
    np.testing.assert_array_equal(candidates[1],seeds[0])
    np.testing.assert_array_equal(candidates[2:],finalists[1:4])
