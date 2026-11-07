"""Geometric feasibility and contract checks for embedded survey starts."""
from pathlib import Path
import sys
from dataclasses import replace
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/airflow'))
import legacy_monitoring as legacy
import monitoring_mbd as monitoring
import survey_routes


def test_fine_grid_represents_upper_passage_and_preserves_original_robot_reference():
    cfg = replace(legacy.Settings(), safety_margin=.12)
    env, _ = monitoring.environment(legacy.physical_settings(cfg))
    state = np.array([1.45, .875, 0., 0.])
    clarity = np.full(len(env.cells), .74)
    belief = legacy.matched_prior(env)
    seeds, shifted, metadata, targets, weights = survey_routes.seed_bank(state, clarity, env, cfg, belief)
    assert seeds.shape == (7, 2 * cfg.steps)
    assert shifted.shape == (2 * cfg.steps,)
    assert targets.shape == (48, 2)
    assert weights.shape == (48,)
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-6)
    assert metadata['grid'] == [128, 74]
    assert len(set(f['path_index'] for f in metadata['survey_frontiers'])) == 2
    assert metadata['clarity_route_index'] == 0
    np.testing.assert_array_equal(metadata['seed_paths'][3], metadata['coarse_paths'][0])
    paths = [np.asarray(path) for path in metadata['coarse_paths']]
    assert max(path[:, 1].max() for path in paths) > 4.44
    for path in paths:
        assert np.all(survey_routes.visible(path[:-1], path[1:], np.asarray(env.obstacles), np.asarray(env.bounds), .125))
    # The optimizer receives actual accelerations of the same full nominal
    # double integrator, not waypoints masquerading as dynamic trajectories.
    for seed in seeds:
        states, acceleration = legacy.old.nominal_reference(seed, jnp.asarray(state), cfg)
        states, acceleration = np.asarray(states), np.asarray(acceleration)
        np.testing.assert_allclose(states[1:, 2:], states[:-1, 2:] + cfg.dt * acceleration, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(states[1:, :2], states[:-1, :2] + cfg.dt * states[:-1, 2:] + .5 * cfg.dt**2 * acceleration, rtol=2e-6, atol=2e-6)
        assert np.max(np.abs(acceleration)) <= cfg.umax


def test_relevance_weights_align_frontier_and_persistent_passage(monkeypatch):
    cfg = replace(legacy.Settings(), safety_margin=.12)
    env, _ = monitoring.environment(legacy.physical_settings(cfg))
    state = np.array([1.45, .875, 0., 0.])
    clarity = np.full(len(env.cells), .74)
    belief = legacy.matched_prior(env)
    calls = []
    def upper_only(points, weights, references, route_indices):
        calls.append((np.asarray(references), np.asarray(route_indices)))
        result = np.zeros(48, np.float32)
        result[int(route_indices[2])*8:(int(route_indices[2])+1)*8] = 3.5
        return result
    _, _, first, _, weights = survey_routes.seed_bank(
        state, clarity, env, cfg, belief, relevance_weights=upper_only)
    assert len(calls) == 1
    assert calls[0][0].shape == (3, cfg.steps*2)
    assert set(f['path_index'] for f in first['survey_frontiers']) == {first['task_route_indices'][2]}
    np.testing.assert_array_equal(weights.reshape(6, 8)[first['task_route_indices'][2]], 3.5)
    assert float(weights.sum()) == 28.  # Exact callback weights, not renormalized.
    retained = first['seed_frontier_states'][3]
    assert retained['passage_slot'] == 3

    # A fresh geometric search changes the distant goal; a selected unfinished
    # upper survey must still keep its former full route and frontier waypoint.
    fresh = [np.asarray(p) for p in first['coarse_paths']]
    fresh[5] = fresh[5][:-1]
    def changed_destinations(*args, **kwargs):
        return [p.copy() for p in fresh], dict(first)
    monkeypatch.setattr(survey_routes, 'task_paths', changed_destinations)
    seeds, _, second, targets, returned_weights = survey_routes.seed_bank(
        state, clarity, env, cfg, belief, relevance_weights=upper_only,
        frontier_state=retained)
    assert len(calls) == 2
    assert second['persistent_frontier_status'] == 'retained'
    np.testing.assert_allclose(second['coarse_paths'][5][-1], retained['task_destination'])
    np.testing.assert_allclose(second['seed_paths'][2][-1], retained['frontier_point'])
    np.testing.assert_allclose(second['seed_frontier_states'][3]['frontier_point'], retained['frontier_point'])
    np.testing.assert_allclose(targets[47], retained['task_destination'])
    assert not np.allclose(calls[-1][0][2], seeds[3])
    np.testing.assert_array_equal(returned_weights, weights)


def test_zero_relevance_has_no_survey_state_or_frontier():
    cfg = replace(legacy.Settings(), safety_margin=.12)
    env, _ = monitoring.environment(legacy.physical_settings(cfg))
    state = np.array([1.45, .875, 0., 0.])
    clarity = np.full(len(env.cells), .74)
    belief = legacy.matched_prior(env)
    _, _, metadata, _, weights = survey_routes.seed_bank(
        state, clarity, env, cfg, belief,
        relevance_weights=lambda points, w, seeds, routes: np.zeros(48))
    assert not np.any(weights)
    assert metadata['survey_frontiers'] == []
    assert metadata['seed_frontier_states'] == [None]*7


def test_remaining_route_preserves_visible_obstacle_passage():
    obstacles = np.array([[2., 2., .75]])
    bounds = np.array([[0., 0.], [5., 5.]])
    route = np.array([[.5, 1.], [.5, 3.], [3.5, 3.], [4., 1.]])
    position = np.array([.6, 2.])
    remaining = survey_routes.remaining_route(route, position, obstacles, bounds)
    np.testing.assert_array_equal(remaining[0], position)
    np.testing.assert_array_equal(remaining[-1], route[-1])
    assert np.all(survey_routes.visible(remaining[:-1], remaining[1:], obstacles, bounds, .125))
    assert remaining[:, 1].max() == 3.
