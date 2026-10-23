"""Known-geometry mode seeds remain free full MBD acceleration sequences."""

import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.quarter.continuous import ControlSettings, nominal_reference
from voi_diffusion_ipp.quarter.environment import environment
from voi_diffusion_ipp.quarter.modes import (
    AIRCRAFT_RADIUS,
    _roadmap,
    demonstrations,
    parameterize,
    visibility_paths,
)
from voi_diffusion_ipp.quarter.rrt import winding_signature


def _compact():
    # A small physical map, with the full-size aircraft and meaningful passages.
    return SimpleNamespace(
        rectangles=np.array([[-2.0, -2.4, 2.0, -0.2, 12.0], [-2.0, 2.2, 2.0, 4.4, 14.0]]),
        bounds=np.array([[-6.7, -5.0], [6.7, 7.0]]),
        regions=np.array([[-5.65, -0.2, -3.55, 2.2], [3.55, 2.5, 5.65, 4.9]]),
    )


def _cfg(**changes):
    return ControlSettings(
        horizon=10.0,
        commit=2.0,
        environment_scale=0.35,
        inspection_scale=0.55,
        rrt_paths=3,
        **changes,
    )


def _dense_free(reference, acceleration, env, cfg, clearance):
    # Independent exact within-interval sampling: constant acceleration yields
    # parabolic segments, so checking only the geometric waypoint chord fails.
    times = np.linspace(0.0, cfg.dt, 21)[None, :, None]
    points = (
        reference[:-1, None, :2]
        + times * reference[:-1, None, 2:]
        + 0.5 * times**2 * acceleration[:, None]
    ).reshape((-1, 2))
    assert np.all(points >= env.bounds[0] + clearance - 2e-5)
    assert np.all(points <= env.bounds[1] - clearance + 2e-5)
    for rectangle in env.rectangles:
        inside = (
            (points >= rectangle[:2] - clearance) & (points <= rectangle[2:4] + clearance)
        ).all(axis=1)
        assert not inside.any()


def test_visibility_graph_has_both_sides_and_reuses_static_geometry():
    rectangles = np.array([[-1.0, -2.0, 1.0, 2.0, 12.0]])
    bounds = np.array([[-5.0, -5.0], [5.0, 5.0]])
    start, goal = np.array([-4.0, 0.0]), np.array([4.0, 0.0])
    _roadmap.cache_clear()
    paths = visibility_paths(start, goal, rectangles, bounds, margin=0.55, count=3)
    assert len(paths) == 2
    assert len({winding_signature(path, rectangles) for path in paths}) == 2
    assert any(np.max(path[:, 1]) > 2.5 for path in paths)
    assert any(np.min(path[:, 1]) < -2.5 for path in paths)
    before = _roadmap.cache_info()
    visibility_paths(start + [-0.1, 0.0], goal, rectangles, bounds, margin=0.55)
    assert _roadmap.cache_info().hits > before.hits


def test_compact_modes_are_fast_full_dynamics_and_reach_other_sensing_region():
    env, cfg = _compact(), _cfg()
    state = np.array([-4.6, 1.0, 0.0, 0.0])
    parameters, metadata = demonstrations(state, env, cfg, 17)
    assert parameters.shape == (12, 2 * cfg.steps)
    assert parameters.dtype == np.float32
    assert metadata["found"] == 12
    json.dumps(metadata)  # Includes only plain JSON-compatible diagnostics.
    fast = [
        (theta, item)
        for theta, item in zip(parameters, metadata["records"])
        if item.get("goal_index") == 1 and item["cruise"] > 2.0
    ]
    assert len(fast) == 3
    assert len({tuple(item["signature"]) for _, item in fast}) == 3
    goal = np.array([4.6, 3.7])
    for theta, item in fast:
        reference, acceleration = map(
            np.asarray, nominal_reference(jnp.array(theta), jnp.array(state), cfg)
        )
        assert np.max(np.abs(acceleration)) < cfg.umax
        assert np.max(np.linalg.norm(reference[:, 2:], axis=1)) <= cfg.speed_max + 1e-5
        _dense_free(reference, acceleration, env, cfg, AIRCRAFT_RADIUS)
        # All 3 routes enter the opposite physical inspection region within
        # the ten-second horizon; no clipped or drawn trajectory substitutes.
        in_region = (
            (reference[:, :2] >= env.regions[1, :2]) & (reference[:, :2] <= env.regions[1, 2:])
        ).all(axis=1)
        assert in_region.any(), item
        assert np.linalg.norm(reference[-1, :2] - goal) < 0.04
        # While transiting past the obstacles, the reference remains moving.
        transit = (reference[:, 0] > -2.0) & (reference[:, 0] < 2.0)
        assert np.min(np.linalg.norm(reference[transit, 2:], axis=1)) > 0.5
    theta, _ = fast[0]
    reference = np.asarray(nominal_reference(jnp.array(theta), jnp.array(state), cfg)[0])
    changed = theta.copy()
    changed[121] += 0.05
    altered = np.asarray(nominal_reference(jnp.array(changed), jnp.array(state), cfg)[0])
    assert not np.array_equal(altered[-1], reference[-1])


def test_bank_only_reads_geometry_and_seed_reproduces_visit_orders():
    env, cfg = _compact(), _cfg(rrt_visits=2)
    state = np.array([-4.6, 1.0, 0.0, 0.0])
    first, a = demonstrations(state, env, cfg, 31)
    second, b = demonstrations(state, env, cfg, 31)
    np.testing.assert_array_equal(first, second)
    assert a == b
    assert first.shape == (24, 2 * cfg.steps)
    assert any(item.get("visit_order") == [1, 0] for item in a["records"])
    # No targets, cells, wind, region names, or scenario labels were supplied.
    assert set(vars(env)) == {"rectangles", "bounds", "regions"}


def test_existing_quarter_geometry_preserves_narrow_passage_and_local_hold():
    env, cfg = environment("two", 0.35, 0.55), _cfg()
    state = np.array([-9.8, 2.8, 0.0, 0.0])
    parameters, metadata = demonstrations(state, env, cfg, 8)
    assert metadata["found"] == 12
    for theta, item in zip(parameters, metadata["records"]):
        reference, acceleration = map(
            np.asarray, nominal_reference(jnp.array(theta), jnp.array(state), cfg)
        )
        _dense_free(reference, acceleration, env, cfg, AIRCRAFT_RADIUS)
        if item["goal_index"] == 0:
            # Float32 site centers can differ from a float64 current state by
            # micrometres. They must not generate enormous near-zero journeys.
            assert item["arrival_time"] == 0.0
            np.testing.assert_allclose(
                reference[:, :2], np.tile(state[:2], (cfg.steps + 1, 1)), atol=2e-6
            )
        if item["goal_index"] == 1 and item["signature"] == [0, 0]:
            middle = (reference[:, 0] > -5.0) & (reference[:, 0] < 5.0)
            assert middle.any()
            assert np.all(reference[middle, 1] > 1.925 + AIRCRAFT_RADIUS)
            assert np.all(reference[middle, 1] < 3.675 - AIRCRAFT_RADIUS)


def test_empty_geometry_and_nonzero_current_velocity_use_exact_safe_braking():
    cfg = _cfg(rrt_visits=1)
    rectangles, bounds = np.empty((0, 4)), np.array([[-8.0, -5.0], [8.0, 5.0]])
    state = np.array([-4.0, 0.0, 0.6, -0.2])
    theta, item = parameterize(
        np.array([state[:2], [4.0, 1.0]]), state, cfg, 2.5, rectangles, bounds
    )
    reference, acceleration = map(
        np.asarray, nominal_reference(jnp.array(theta), jnp.array(state), cfg)
    )
    assert item["reached_goal"]
    np.testing.assert_allclose(reference[-1], [4.0, 1.0, 0.0, 0.0], atol=2e-5)
    assert np.max(np.abs(acceleration)) < cfg.umax
    assert len(visibility_paths([-4.0, 0.0], [4.0, 1.0], [], bounds)) == 1


def test_geometrically_unreachable_goals_are_explicit_not_colliding_routes():
    env = SimpleNamespace(
        rectangles=np.array([[-1.0, -5.0, 1.0, 5.0, 10.0]]),
        bounds=np.array([[-5.0, -5.0], [5.0, 5.0]]),
        regions=np.array([[3.0, -0.5, 4.0, 0.5]]),
    )
    parameters, metadata = demonstrations(np.array([-3.5, 0.0, 0.0, 0.0]), env, _cfg(), 2)
    assert metadata["found"] == 0
    assert parameters.shape == (6, 200)
    assert all(item.get("unavailable") for item in metadata["records"])


def test_fixed_mode_selection_gives_opposite_site_all_three_classes():
    from voi_diffusion_ipp.quarter.modes import select_initial_modes

    env, cfg = _compact(), _cfg()
    state = np.array([-4.6, 1.0, 0.0, 0.0])
    bank, record = demonstrations(state, env, cfg, 11)
    modes, selected = select_initial_modes(bank, record, state, env, cfg)
    assert modes.shape == (4, 2 * cfg.steps)
    assert selected["chosen_indices"][0] == -1
    assert selected["records"][0]["kind"] == "hold"
    assert selected["records"][0]["safe_reference"]
    np.testing.assert_array_equal(modes[0], 0.0)
    geometry = selected["records"][1:]
    assert {item["goal_index"] for item in geometry} == {1}
    assert len({tuple(item["signature"]) for item in geometry}) == 3
    assert not any(item["duplicate"] for item in geometry)
    for item in geometry:
        same_class = [
            candidate["arrival_time"]
            for candidate in record["records"]
            if candidate.get("goal_index") == item["goal_index"]
            and candidate.get("signature") == item["signature"]
        ]
        assert item["arrival_time"] == min(same_class)
    json.dumps(selected)


def test_between_site_selection_covers_each_goal_before_more_classes():
    from voi_diffusion_ipp.quarter.modes import select_initial_modes

    env, cfg = _compact(), _cfg()
    state = np.array([-0.8, 1.0, 0.2, 0.0])
    bank, record = demonstrations(state, env, cfg, 11)
    modes, selected = select_initial_modes(bank, record, state, env, cfg)
    assert modes.shape == (4, 2 * cfg.steps)
    assert selected["records"][0]["kind"] == "braking"
    assert selected["records"][0]["safe_reference"]
    assert {item["goal_index"] for item in selected["records"][1:3]} == {0, 1}
    reference, _ = map(np.asarray, nominal_reference(jnp.array(modes[0]), jnp.array(state), cfg))
    np.testing.assert_allclose(reference[-1, 2:], 0.0, atol=1e-6)


def test_fixed_budget_marks_duplicate_modes_when_classes_are_exhausted():
    from voi_diffusion_ipp.quarter.modes import select_initial_modes

    env, cfg = _compact(), _cfg()
    state = np.array([-4.6, 1.0, 0.0, 0.0])
    bank, record = demonstrations(state, env, cfg, 11)
    # Simple wrapper lets this helper remain compatible with the old settings
    # dataclass as well as the new mode_count configuration.
    selection_cfg = SimpleNamespace(
        **{
            name: getattr(cfg, name)
            for name in ("steps", "dt", "umax", "planning_clearance", "sensor_sigma")
        },
        mode_count=7,
    )
    modes, selected = select_initial_modes(bank, record, state, env, selection_cfg)
    assert modes.shape == (7, 2 * cfg.steps)
    assert selected["distinct_geometry_modes"] == 3
    assert [item["duplicate"] for item in selected["records"]] == [False] * 4 + [True] * 3
    np.testing.assert_array_equal(modes[1:4], modes[4:7])


def test_short_nonzero_route_has_an_interior_speed_sample():
    cfg = _cfg()
    state = np.zeros(4)
    theta, item = parameterize(
        np.array([[0.0, 0.0], [0.01, 0.0]]),
        state,
        cfg,
        2.0,
        [],
        np.array([[-2.0, -2.0], [2.0, 2.0]]),
    )
    assert item["arrival_time"] < 1.0
    reference, _ = map(np.asarray, nominal_reference(jnp.array(theta), jnp.array(state), cfg))
    np.testing.assert_allclose(reference[-1], [0.01, 0.0, 0.0, 0.0], atol=1e-6)


def test_reached_states_inside_preferred_clearance_recover_all_geometric_modes():
    from voi_diffusion_ipp.quarter.modes import select_initial_modes

    env = _compact()
    cfg = _cfg(dt=0.2, planning_clearance=0.6)
    # Actual plan-start states from the forward 300 s runs, before recovery.
    # The first needs an obstacle-side transition, the next a boundary escape,
    # and the last already exceeds preferred speed by 0.0434 m/s.
    reached = (
        [-2.16222, 0.42878, -0.76039, -0.56395],  # proposed, t=15 s
        [6.13, 2.70894, 0.01819, 0.24582],  # proposed, t=50 s
        [-6.17648, 0.21819, -0.0374, -0.5053],  # aware, t=120 s
        [1.55569, 1.60489, 3.1402, 0.14155],  # proposed, t=45 s
    )
    for values in reached:
        state = np.asarray(values)
        bank, metadata = demonstrations(state, env, cfg, 8)
        assert metadata["found"] == 12
        modes, selection = select_initial_modes(bank, metadata, state, env, cfg)
        assert selection["distinct_geometry_modes"] == 6
        assert selection["records"][0]["safe_reference"]
        recovery = metadata["recovery"]
        transition_steps = round((recovery["braking_time"] + recovery["recovery_time"]) / cfg.dt)
        braking_steps = round(recovery["braking_time"] / cfg.dt)
        for theta, item in zip(modes, selection["records"]):
            reference, acceleration = map(
                np.asarray, nominal_reference(jnp.array(theta), jnp.array(state), cfg)
            )
            assert np.max(np.abs(acceleration)) < cfg.umax
            _dense_free(reference, acceleration, env, cfg, AIRCRAFT_RADIUS)
            if item["kind"] == "geometry":
                # The already-reached prefix may start within the preferred
                # buffer. Every subsequent reference interval restores it.
                _dense_free(
                    reference[transition_steps:],
                    acceleration[transition_steps:],
                    env,
                    cfg,
                    0.6 - 1e-5,
                )
                assert (
                    np.max(np.linalg.norm(reference[transition_steps:, 2:], axis=1))
                    <= cfg.speed_max + 1e-5
                )
                speed = np.linalg.norm(reference[: braking_steps + 1, 2:], axis=1)
                assert np.all(np.diff(speed) <= 1e-5)
        if recovery["recovery_required"]:
            assert recovery["recovery_distance"] < 0.25
            assert recovery["recovery_time"] <= 1.0
        if np.linalg.norm(state[2:]) > cfg.speed_max:
            assert recovery["initial_speed_excess"] > 0.0


def test_recovery_never_shrinks_aircraft_to_accept_a_colliding_initial_state():
    env, cfg = _compact(), _cfg(dt=0.2, planning_clearance=0.6)
    # Center only 0.20 m above the lower building: the physical 0.275 m
    # aircraft is already intersecting it, even though center is outside.
    bank, metadata = demonstrations(np.array([0.0, 0.0, 0.0, 0.0]), env, cfg, 8)
    assert metadata["found"] == 0
    assert metadata["recovery"]["physical_clearance"] >= AIRCRAFT_RADIUS
    assert bank.shape == (12, 2 * cfg.steps)


def test_none_effort_is_bitwise_compatible_with_legacy_configuration():
    env, cfg = _compact(), _cfg(dt=0.2, planning_clearance=0.6, initialization_effort=None)
    legacy_fields = dict(vars(cfg))
    legacy_fields.pop("initialization_effort")
    legacy = SimpleNamespace(**legacy_fields, steps=cfg.steps)
    for state in (
        np.array([-4.6, 1.0, 0.0, 0.0]),
        np.array([-2.16222, 0.42878, -0.76039, -0.56395]),
    ):
        current, current_record = demonstrations(state, env, cfg, 11)
        previous, previous_record = demonstrations(state, env, legacy, 11)
        np.testing.assert_array_equal(current, previous)
        assert current_record == previous_record
        assert current_record["initialization_effort"] is None


def test_effort75_keeps_dense_physical_constraints_and_all_initial_routes_sense():
    from voi_diffusion_ipp.quarter.modes import select_initial_modes

    env = _compact()
    cfg = _cfg(dt=0.2, planning_clearance=0.6, initialization_effort=0.75, mode_count=7)
    states = (
        [-4.6, 1.0, 0.0, 0.0],
        [-2.16222, 0.42878, -0.76039, -0.56395],
        [6.13, 2.70894, 0.01819, 0.24582],
        [-6.17648, 0.21819, -0.0374, -0.5053],
        [1.55569, 1.60489, 3.1402, 0.14155],
    )
    for case, values in enumerate(states):
        state = np.asarray(values)
        bank, metadata = demonstrations(state, env, cfg, 11)
        assert metadata["initialization_effort"] == 0.75
        assert metadata["found"] == 12
        assert bank.shape == (12, 2 * cfg.steps)
        recovery = metadata["recovery"]
        prefix_steps = round((recovery["braking_time"] + recovery["recovery_time"]) / cfg.dt)
        for theta, item in zip(bank, metadata["records"]):
            reference, acceleration = map(
                np.asarray, nominal_reference(jnp.array(theta), jnp.array(state), cfg)
            )
            assert np.max(np.abs(acceleration)) <= 0.96 * cfg.umax + 1e-6
            _dense_free(reference, acceleration, env, cfg, AIRCRAFT_RADIUS)
            _dense_free(reference[prefix_steps:], acceleration[prefix_steps:], env, cfg, 0.6 - 1e-5)
            assert (
                np.max(np.linalg.norm(reference[prefix_steps:, 2:], axis=1)) <= cfg.speed_max + 1e-5
            )
            if case == 0 and item["goal_index"] == 1 and item["cruise"] > 2.0:
                in_region = (
                    (reference[:, :2] >= env.regions[1, :2])
                    & (reference[:, :2] <= env.regions[1, 2:])
                ).all(axis=1)
                assert in_region.any(), item
        modes, selected = select_initial_modes(bank, metadata, state, env, cfg)
        assert modes.shape == (7, 2 * cfg.steps)
        assert selected["distinct_geometry_modes"] == (3 if case == 0 else 6)
        altered = modes[1].copy()
        altered[45] += 0.05
        original = np.asarray(nominal_reference(jnp.array(modes[1]), jnp.array(state), cfg)[0])
        changed = np.asarray(nominal_reference(jnp.array(altered), jnp.array(state), cfg)[0])
        assert not np.array_equal(original[-1], changed[-1])
