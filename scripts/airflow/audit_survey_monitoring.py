"""Audit the added online-survey runs without changing any experiment.

Every actual wind observation, POAM update, protected-center mask, displayed
posterior, physical execution step, saved MBD reverse step, and final candidate
selection is checked. Posterior-derived planning fields are regenerated.
By default the first/last complete optimizer calls are also replayed, including
the full physical survey objective and proposal RNG. ``--planner-replay all``
checks every objective evaluation and proposal; ``none`` skips only that costly
optimizer replay. Held-control plant replay is not a global safety proof.
"""
import argparse
import ast
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import legacy_monitoring as legacy
import monitoring_mbd as monitoring
import survey_monitoring
from supported_poam import condition_supported
from run_monitoring_mbd import write
from voi_diffusion_ipp.quarter.diffusion import cosine_schedule
from voi_diffusion_ipp.quarter.multimode import mode_score_and_update
from voi_diffusion_ipp.wind.gp import WorldField


def same(actual, expected, *, atol=1e-8, rtol=1e-8):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=atol, rtol=rtol)


def source_checks(directory, protocol):
    histories = [protocol['source_sha256'], *protocol.get('source_history', [])]
    current_mismatches = []
    for history in histories:
        for relative, digest in history.items():
            current = monitoring.physical.ROOT / relative
            paths = [current, *directory.glob('source_snapshot*/' + relative)]
            assert any(path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
                       for path in paths), ('Source digest unavailable', relative, digest)
            if history is histories[0] and hashlib.sha256(current.read_bytes()).hexdigest() != digest:
                current_mismatches.append(relative)
    # Verify the actual implementation used for replay, not merely an archived
    # digest for code that differs from the interpreter's imported modules.
    assert not current_mismatches, ('Replay requires the recorded source version', current_mismatches)
    planning_files = ['survey_monitoring.py', 'survey_routes.py', 'survey_information.py',
                      'supported_poam.py', 'observed_poam.py', 'survey_forecast.py']
    for name in planning_files:
        tree = ast.parse(Path(__file__).with_name(name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert not (isinstance(node.func, ast.Attribute) and node.func.attr == 'load_wind'), name
                assert not (isinstance(node.func, ast.Attribute) and
                            isinstance(node.func.value, ast.Name) and node.func.value.id == 'truth'), name
    dependencies = ['scripts/airflow/observed_poam.py', 'scripts/airflow/active_sensing_monitoring.py',
                    'scripts/airflow/survey_forecast.py']
    return dict(recorded_files_verified=len(histories[0]), source_histories_verified=len(histories)-1,
                unrecorded_transitive_dependencies=[p for p in dependencies if p not in histories[0]],
                planning_true_field_calls_absent=True)


def allocation_numpy(centers, mask, positions, minimum_distance=.15):
    """Independent small-array allocation check; no covariance update shortcut."""
    centers, mask = centers.copy(), mask.copy()
    for point in positions:
        distance = np.linalg.norm(centers-point, axis=1)
        if np.min(np.where(mask, distance, np.inf)) > minimum_distance:
            index = np.argmin(distance if mask.all() else np.where(mask, np.inf, distance))
            centers[index] = point
            mask[index] = True
    return centers, mask


def physical_replay(raw, cfg, truth, env, check_time):
    samples, states, clarity = raw['samples'], raw['states'], raw['clarity']
    assert len(samples) == len(states) == len(clarity)
    assert np.isfinite(samples).all() and np.isfinite(clarity).all()
    assert clarity.min() >= 0 and clarity.max() <= 1
    same(samples[:, 1:5], states, atol=0, rtol=0)
    same(np.diff(samples[:, 0]), .05, atol=1e-10, rtol=0)
    assert np.max(np.abs(samples[:, 5:7])) <= cfg.umax+1e-9
    execute = jax.jit(jax.vmap(lambda x, q, u, alive:
                     monitoring.advance(x, q, u, truth, env, cfg, alive, interval=.05)))
    errors = dict(state=0., clarity=0., cost=0.)
    increments = []
    for start in range(0, len(states)-1, 600):
        stop = min(start+600, len(states)-1)
        end = execute(jnp.asarray(states[start:stop]), jnp.asarray(clarity[start:stop]),
                      jnp.asarray(samples[start+1:stop+1, 5:7]), jnp.asarray(samples[start:stop, 18] > 0))
        errors['state'] = max(errors['state'], float(np.max(np.abs(np.asarray(end[0])-states[start+1:stop+1]))))
        errors['clarity'] = max(errors['clarity'], float(np.max(np.abs(np.asarray(end[1])-clarity[start+1:stop+1]))))
        np.testing.assert_array_equal(np.asarray(end[2]), samples[start+1:stop+1, 18] > 0)
        increments.append(np.asarray(end[3]))
        check_time()
    if increments:
        errors['cost'] = float(np.max(np.abs(np.cumsum(np.concatenate(increments))-samples[1:, 14])))
    assert max(errors.values()) < 1e-8, errors
    return errors


def observations_and_beliefs(directory, ident, record, protocol, raw, cfg, env, grid, truth, check_time):
    assert 'initial_survey' not in protocol, 'This audit requires the declared no-survey experiment'
    assert protocol['prior_configuration'].startswith('survey-matched 128')
    assert protocol['execution_belief']['posterior_update_seconds'] == .2
    assert protocol['numerics']['actual_inference_dtype'] == 'float64'
    b = legacy.matched_prior(env)
    mean, covariance = b.predict(jnp.asarray(env.cells))
    same(mean, 0, atol=0, rtol=0)
    same(np.sqrt(covariance[:, 0, 0]), 1.8, atol=1e-6, rtol=3e-6)
    mask = np.zeros(len(b.mean), dtype=bool)
    assert len(mask) == 128
    rows = raw['observation_rows']
    samples = raw['samples']
    noise = np.random.default_rng(protocol['seed']).normal(
        size=(round(record['summary']['planned_seconds']/.2)+1, 2))
    indices = np.rint(rows[:, 0]/.05).astype(int)
    same(rows[:, 1:3], raw['states'][indices, :2], atol=0, rtol=0)
    assert np.all(samples[indices, 18] > 0), 'Measurements after physical contact'
    expected_times = samples[(np.abs(samples[:, 0]/.2-np.rint(samples[:, 0]/.2)) < 1e-7)
                             & (samples[:, 18] > 0), 0]
    same(rows[:, 0], expected_times, atol=1e-8, rtol=0)
    expected_wind = np.asarray(truth.at(jnp.asarray(rows[:, 1:3])))
    expected_wind += cfg.observation_std*noise[np.rint(rows[:, 0]/.2).astype(int)]
    same(rows[:, 3:5], expected_wind, atol=1e-12, rtol=0)

    frame_by_time = {round(frame['t'], 8): frame for frame in record['belief_frames']}
    plan_by_time = {round(plan['t'], 8): i for i, plan in enumerate(record['plans'])}
    checkpoints = {}
    max_map_error = 0.
    max_saved_belief_error = 0.
    saturated_at = None
    initial_parameters = jax.tree.leaves(b.basis.parameters)
    for index, row in enumerate(rows):
        # Reproduce the runner's observation-storage contract exactly: physical
        # values/positions are recorded in float64, then assimilated as float32
        # inputs by the float64 POAM update. The evolving belief is never cast.
        positions = np.asarray(row[None, 1:3], dtype=np.float32)
        values = np.asarray(row[None, 3:5], dtype=np.float32)
        old_centers = np.asarray(b.basis.centers)
        expected_centers, expected_mask = allocation_numpy(old_centers, mask, positions)
        b, got_mask = condition_supported(b, positions, values, mask)
        jax.block_until_ready(b)
        mask = np.asarray(got_mask)
        same(b.basis.centers, expected_centers, atol=0, rtol=0)
        np.testing.assert_array_equal(mask, expected_mask)
        assert all(np.isfinite(a).all() for a in jax.tree.leaves(b))
        assert np.asarray(b.mean).dtype == np.float64
        for actual, expected in zip(jax.tree.leaves(b.basis.parameters), initial_parameters):
            same(actual, expected, atol=0, rtol=0)
        if mask.all() and saturated_at is None:
            saturated_at = float(row[0])
        instant = round(float(row[0]), 8)
        if instant in frame_by_time:
            frame = frame_by_time[instant]
            mu, cov = b.predict(jnp.asarray(grid['all_cells']))
            sigma = np.sqrt(np.asarray(cov)[:, 0, 0])
            max_map_error = max(max_map_error, float(np.max(np.abs(np.asarray(mu)-frame['mean']))),
                                float(np.max(np.abs(sigma-frame['std']))))
            assert frame['observations'] == index+1
            assert np.linalg.eigvalsh(np.asarray(b.covariance)).min() > -1e-9
        if instant in plan_by_time:
            plan_index = plan_by_time[instant]
            saved = np.load(directory/f'{ident}-plan-{plan_index:03d}.npz')
            expected = [saved[key] for key in sorted(saved.files) if key.startswith('belief_leaf_')]
            for a, c in zip(jax.tree.leaves(b), expected):
                same(a, c, atol=2e-11, rtol=2e-11)
                if np.asarray(a).size:
                    max_saved_belief_error = max(max_saved_belief_error, float(np.max(np.abs(np.asarray(a)-c))))
            np.testing.assert_array_equal(mask, saved['support_mask'])
            checkpoints[plan_index] = b
        check_time()
    assert max_map_error <= 5.01e-6, max_map_error
    assert len(checkpoints) == len(record['plans'])
    checkpoints['final'] = b
    np.testing.assert_array_equal(mask, raw['support_mask'])
    assert int(mask.sum()) == record['summary']['supported_inducing_points']
    assert bool(mask.all()) == record['summary']['support_capacity_saturated']
    changes = sum(not np.array_equal(a['mean'], b['mean'])
                  for a, b in zip(record['belief_frames'], record['belief_frames'][1:]))
    return checkpoints, dict(actual_observations=len(rows), online_updates=len(rows),
                            displayed_frames=len(frame_by_time), changed_mean_frames=changes,
                            max_display_rounding_error=max_map_error,
                            max_saved_belief_error=max_saved_belief_error,
                            supported_inducing_points=int(mask.sum()),
                            support_capacity_saturated_at_s=saturated_at,
                            initial_survey_observations=0,
                            all_center_allocations_from_actual_observations=True)


def saved_planner_checks(directory, ident, record, protocol, beliefs, cfg, env, check_time,
                         planner_replay='edges'):
    assert cfg.optimism == 1 and cfg.mixture == 0 and cfg.guidance_mode == 'off'
    assert cfg.prediction_filter == 'hard' and cfg.construction_worlds == 8
    assert cfg.safety_margin == .12
    assert cfg.validation_worlds == 1
    assert protocol['validation_safety']['full_horizon_seconds'] == cfg.horizon
    assert protocol['validation_safety']['control_dt'] == .05
    assert protocol['validation_safety']['physics_clarity_dt'] == .01
    assert protocol['no_anticipated_updates'] and protocol['no_optimistic_objective']
    assert protocol['exploration_mode'] in ('survey', 'passive')
    sample = survey_monitoring.field_sampler(cfg)
    with jax.experimental.disable_x64():
        replay = survey_monitoring.optimizer(legacy.float32(env), cfg,
                                             protocol['exploration_mode'], protocol['cost_allowance'])
    maximum_equation_error = 0.
    maximum_field_error = 0.
    replay_indices = []
    maximum_optimizer_error = 0.
    achieved_changes = []
    plans = record['plans']
    for index, plan in enumerate(plans):
        saved = np.load(directory/f'{ident}-plan-{index:03d}.npz')
        same(saved['initial_state'], np.asarray(record['samples'])[round(plan['t']/.05), 1:5], atol=5.01e-8, rtol=0)
        b = beliefs[index]
        bp = legacy.float32(b)
        if protocol['numerics'].get('posterior_field_generation_dtype') == 'float64':
            with jax.experimental.enable_x64():
                regenerated = legacy.float32(sample(b, jax.random.PRNGKey(protocol['seed']+1000+index),
                                                     jax.random.PRNGKey(protocol['seed']+271828)))
        else:
            with jax.experimental.disable_x64():
                regenerated = sample(bp, jax.random.PRNGKey(protocol['seed']+1000+index),
                                     jax.random.PRNGKey(protocol['seed']+271828))
        with jax.experimental.disable_x64():
            for got, key in zip(regenerated, ('mean', 'construction', 'worlds', 'validation')):
                same(got, saved[key], atol=1e-6, rtol=1e-6)
                maximum_field_error = max(maximum_field_error, float(np.max(np.abs(np.asarray(got)-saved[key]))))
            alpha, abar = cosine_schedule(cfg.iterations, cfg.cosine_end)
            for level in range(cfg.iterations):
                j = cfg.iterations-1-level
                scores = jnp.asarray(saved['search_sample_costs'][level])
                log_weights = -scores/cfg.temperature+jnp.asarray(saved['search_log_importance_ratios'][level])
                updated, _, weights, ok = mode_score_and_update(
                    jnp.asarray(saved['search_before'][level]), jnp.asarray(saved['search_samples'][level]),
                    log_weights, jnp.asarray(saved['search_proposal_modes']), alpha[j], abar[j])
                valid = np.asarray(ok)
                if valid.any():
                    maximum_equation_error = max(maximum_equation_error, float(np.max(np.abs(
                        np.asarray(updated)[valid]-saved['search_after'][level][valid]))))
                same(weights, saved['search_weights'][level], atol=3e-5, rtol=3e-5)
        selected = int(saved['selected_index'])
        cost, gain, contact = (saved[key] for key in
                               ('validation_task_costs', 'validation_information', 'validation_contacts'))
        bonus = saved['validation_bonuses']
        route_values = saved['route_values']
        enabled = bool(saved['information_enabled'])
        safe = np.all(contact == 0, axis=1) & np.isfinite(cost) & np.isfinite(gain) & np.isfinite(bonus)
        assert len(cost) == 5 and contact.shape == (5, cfg.validation_worlds)
        winner = int(np.argmax(np.all(saved['search_finalists'] == saved['search_final'], axis=1)))
        candidate_indices = np.array([winner, -1, 1, 2, 3])
        candidates = np.stack((saved['search_finalists'][winner], saved['search_initial_parameters'][0],
                               *saved['search_finalists'][1:4]))
        np.testing.assert_array_equal(saved['candidate_indices'], candidate_indices)
        np.testing.assert_array_equal(saved['candidates'], candidates)
        remaining = record['summary']['planned_seconds']-plan['t']
        amortization = np.clip(remaining/cfg.horizon, 1., 20.)
        same(saved['amortization_factor'], amortization, atol=2e-6, rtol=2e-6)
        assert np.all(route_values >= 0) and np.all(bonus >= 0)
        assert bonus.max() <= amortization*route_values.max()+1e-5
        budget = cost[safe].min()+protocol['cost_allowance']
        same(saved['cost_budget'], budget, atol=2e-6, rtol=2e-6)
        assert safe[selected]
        exploring = plan['exploring']
        benefits = np.asarray(plan.get('task_relevance_benefits', []))
        assert enabled == (plan['route_wind_std'] > .30 and plan['route_force_std'] > .10
                            and bool(np.any(route_values > 0)) and protocol['exploration_mode'] == 'survey')
        task_index = int(np.argmin(np.where(safe, cost, np.inf)))
        assert int(saved['task_index']) == task_index
        assert exploring == (enabled and cost[selected] > cost[task_index]+1e-6 and bonus[selected] > 0)
        if not enabled:
            np.testing.assert_array_equal(bonus, np.zeros(5))
        allowed = safe & ((cost <= budget) if enabled else True)
        normalization = plan.get('score_normalization', 1.)
        assert normalization == 1.
        objective = cost-bonus
        scores = np.where(allowed, objective, np.inf)
        assert selected == int(np.argmin(scores))
        same(plan['selection_score'], scores[selected], atol=1e-6, rtol=1e-6)
        same(plan['expected_cost'], cost[selected], atol=1e-7, rtol=1e-7)
        same(plan['survey_fraction'], gain[selected], atol=1e-7, rtol=1e-7)
        np.testing.assert_array_equal(saved['theta'], candidates[selected])
        assert np.min(gain) >= -1e-7 and np.max(gain) <= 1+1e-7
        should_replay = planner_replay == 'all' or (planner_replay == 'edges' and index in (0, len(plans)-1))
        if should_replay:
            initials = saved['search_initial_parameters']
            with jax.experimental.disable_x64():
                result = replay(bp, legacy.float32(jnp.asarray(saved['initial_state'])),
                                legacy.float32(jnp.asarray(saved['initial_clarity'])),
                                *[jnp.asarray(saved[key]) for key in ('mean', 'construction', 'worlds', 'validation')],
                                jnp.asarray(initials[-1]), jax.random.PRNGKey(protocol['seed']+2000+index),
                                jnp.asarray(initials[:-1]), jnp.asarray(saved['target_positions'], jnp.float32),
                                jnp.asarray(saved['target_weights'], jnp.float32),
                                jnp.asarray(saved['route_indices'], jnp.int32), jnp.asarray(remaining, jnp.float32))
                jax.block_until_ready(result)
            assert int(result.index) == selected and bool(result.exploring) == exploring
            same(result.theta, saved['theta'], atol=1e-5, rtol=1e-5)
            same(result.task_costs, cost, atol=3e-5, rtol=3e-5)
            same(result.information, gain, atol=3e-5, rtol=3e-5)
            np.testing.assert_array_equal(result.contacts, contact)
            same(result.initial_budget, plan['initial_cost_budget'], atol=3e-5, rtol=3e-5)
            same(result.relevance_benefits, benefits, atol=3e-5, rtol=3e-5)
            same(result.score_normalization, normalization, atol=3e-5, rtol=3e-5)
            same(result.route_values, route_values, atol=3e-5, rtol=3e-5)
            same(result.validation_bonuses, bonus, atol=3e-5, rtol=3e-5)
            same(result.amortization_factor, amortization, atol=3e-5, rtol=3e-5)
            assert bool(result.information_enabled) == enabled and int(result.task_index) == task_index
            np.testing.assert_array_equal(result.candidate_indices, candidate_indices)
            same(result.candidates, candidates, atol=3e-5, rtol=3e-5)
            same(result.validation.prefix_max_slack, saved['validation_prefix_max_slack'], atol=3e-5, rtol=3e-5)
            np.testing.assert_array_equal(result.validation.prefix_infeasible_steps, saved['validation_prefix_infeasible_steps'])
            same(result.validation.states[selected], plan['predicted_states'], atol=3e-5, rtol=3e-5)
            same(result.validation.min_clearance[selected], plan['predicted_min_clearance'], atol=3e-5, rtol=3e-5)
            same(result.validation.max_slack[selected], plan['predicted_max_slack'], atol=3e-5, rtol=3e-5)
            same(result.validation.values[selected, :, 4], plan['predicted_monitoring_cost'], atol=3e-5, rtol=3e-5)
            same(result.validation.values[selected, :, 5], plan['predicted_safety_cost'], atol=3e-5, rtol=3e-5)
            for name, actual in zip(result.search._fields, result.search):
                expected = saved['search_'+name]
                same(actual, expected, atol=3e-5, rtol=3e-5)
                finite = np.isfinite(actual) & np.isfinite(expected)
                if finite.any() and np.asarray(actual).dtype.kind != 'b':
                    maximum_optimizer_error = max(maximum_optimizer_error, float(np.max(
                        np.abs(np.asarray(actual)[finite]-expected[finite]))))
            replay_indices.append(index)
        if plan['t'] < record['summary']['simulated_seconds'] and benefits.size:
            with jax.experimental.disable_x64():
                points = jnp.asarray(saved['target_positions'], jnp.float32)
                relevant = survey_monitoring.mapped_route_weights(
                    jnp.asarray(saved['target_weights'], jnp.float32),
                    jnp.asarray(saved['route_indices'], jnp.int32), jnp.asarray(route_values > 0))
                winds = jax.vmap(lambda field: WorldField(legacy.BASIS, field).at(points))(
                    jnp.asarray(saved['construction']))
                task_weights, _ = survey_monitoring.force_relevance(winds, points, relevant)
            next_belief = beliefs.get(index+1, beliefs['final'])
            with jax.experimental.enable_x64():
                _, before_cov = b.predict(points.astype(jnp.float64))
                _, after_cov = next_belief.predict(points.astype(jnp.float64))
                w = task_weights.astype(jnp.float64)
                before_variance = float(2*jnp.sum(w*before_cov[:, 0, 0]))
                after_variance = float(2*jnp.sum(w*after_cov[:, 0, 0]))
            achieved_changes.append(dict(t=plan['t'], weighted_variance_before=before_variance,
                                          weighted_variance_after=after_variance,
                                          achieved_reduction=before_variance-after_variance,
                                          achieved_fraction=(before_variance-after_variance)/before_variance if before_variance > 1e-20 else 0.,
                                          forecast_fraction=plan['survey_fraction']))
        check_time()
    assert maximum_equation_error < 1e-4
    return dict(saved_planning_calls=len(plans), posterior_field_sets_regenerated=len(plans),
                max_field_error=maximum_field_error, max_reverse_update_error=maximum_equation_error,
                every_candidate_selection_verified=True, full_optimizer_replayed_indices=replay_indices,
                max_full_optimizer_error=maximum_optimizer_error,
                actual_task_target_variance_changes=achieved_changes,
                actual_variance_weighting='Same pre-plan route-usefulness/force-relevance weights; same target points before and after actual committed-prefix observations. Negative achieved reductions are retained.')


def audit(directory, site=monitoring.physical.SITE, *, ids=None, allow_incomplete=False,
          planner_replay='edges', wall_limit=180.):
    started = time.perf_counter()
    def check_time():
        if time.perf_counter()-started > wall_limit:
            raise TimeoutError(f'Audit wall budget exceeded ({wall_limit:g} seconds)')
    if ids is None:
        ids = [path.name.removesuffix('-protocol.json') for path in sorted(directory.glob('s?-*-protocol.json'))]
    assert ids, 'No experiment protocols found'
    checks = []
    for ident in ids:
        protocol = json.loads((directory/f'{ident}-protocol.json').read_text())
        record = json.loads((directory/f'{ident}.json').read_text())
        assert record['summary']['completed'] or allow_incomplete, (ident, 'Incomplete experiment')
        sources = source_checks(directory, protocol)
        cfg = legacy.Settings(**protocol['legacy_settings'])
        physical = replace(legacy.physical_settings(cfg),
                           safety_margin=protocol['execution_belief']['safety_margin_m'])
        assert physical.safety_margin == .12 and protocol['execution_belief']['construction_fields'] == 8
        env, grid = monitoring.environment(physical)
        truth = monitoring.physical.load_wind(protocol['scenario'], site)
        assert hashlib.sha256((site/f'data/scenario{protocol["scenario"]}/stationary-velocity.f32').read_bytes()).hexdigest() == protocol['wind_sha256']
        raw = np.load(directory/f'{ident}-execution.npz')
        same(np.round(raw['samples'], 7), record['samples'], atol=0, rtol=0)
        for frame in record['clarity_frames']:
            same(np.round(raw['clarity'][round(frame['t']/.05)], 6), frame['q'], atol=0, rtol=0)
        plant = physical_replay(raw, physical, truth, env, check_time)
        beliefs, belief_result = observations_and_beliefs(
            directory, ident, record, protocol, raw, physical, env, grid, truth, check_time)
        planning = saved_planner_checks(directory, ident, record, protocol, beliefs, cfg, env,
                                       check_time, planner_replay)
        accepted = [plan for plan in record['plans'] if plan['t'] < record['summary']['simulated_seconds']]
        warm = [plan['wall_seconds'] for plan in accepted if not plan['timing']['compilation_included']]
        assert max(warm, default=0) <= protocol['runtime_limit_s']
        pending = [plan for plan in record['plans'] if plan['t'] >= record['summary']['simulated_seconds']]
        assert not pending or (allow_incomplete and record['summary']['stop_reason'] == 'Planning runtime budget exceeded')
        prefix_file = directory/f'resume-prefix-{ident}-execution.npz'
        if prefix_file.exists():
            prefix = np.load(prefix_file)
            for key in ('samples', 'states', 'clarity', 'observation_rows'):
                np.testing.assert_array_equal(prefix[key], raw[key][:len(prefix[key])])
        for i, rejected in enumerate(protocol.get('rejected_planning_calls', []), 1):
            old = np.load(directory/f'{ident}-rejected-{i:03d}.npz')
            repeated = np.load(directory/f'{ident}-plan-{round(rejected["t"]/cfg.commit):03d}.npz')
            for key in ('theta', 'initial_state', 'initial_clarity', 'mean', 'construction', 'worlds',
                        'validation', 'support_mask', 'target_positions', 'target_weights'):
                np.testing.assert_array_equal(old[key], repeated[key])
        result = dict(id=ident, simulated_seconds=record['summary']['simulated_seconds'],
                      completed=record['summary']['completed'], sources=sources,
                      physical_replay_errors=plant, belief_replay=belief_result, planner=planning,
                      warm_accepted_plan_max_s=max(warm, default=0),
                      rejected_planning_calls=protocol.get('rejected_planning_calls', []),
                      pending_unexecuted_plans=len(pending), resume_prefix_verified=prefix_file.exists())
        checks.append(result)
        print(json.dumps(result), flush=True)
        check_time()
    output = dict(checks=checks, wall_seconds=time.perf_counter()-started,
                  scope='All recorded physical execution, actual measurements, supported POAM updates/masks/maps, posterior planning fields, saved MBD reverse updates and selections. Full optimizer replay scope is listed per case. No global safety, convergence or exact future-POAM information-value claim.')
    write(directory/'survey-audit.json', output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--site', type=Path, default=monitoring.physical.SITE)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--planner-replay', choices=('none', 'edges', 'all'), default='edges')
    parser.add_argument('--wall-limit', type=float, default=180.)
    args = parser.parse_args()
    audit(args.directory, args.site, ids=args.ids, allow_incomplete=args.allow_incomplete,
          planner_replay=args.planner_replay, wall_limit=args.wall_limit)
