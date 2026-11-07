"""Package only the audited added online-survey comparison; preserve old data."""
import argparse
import gzip
import hashlib
import html
import io
import json
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = 'monitoring-survey-online-300'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_method(source):
    def inline(value):
        value = html.escape(value)
        value = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', value)
        return re.sub(r'\[([^]]+)\]\((https://[^)]+)\)', r'<a href="\2">\1</a>', value)
    blocks = []
    for block in source.split('\n\n'):
        if block.startswith('# '):
            blocks.append('<h1>'+inline(block[2:])+'</h1>')
        elif block.startswith('## '):
            blocks.append('<h2>'+inline(block[3:])+'</h2>')
        elif block.startswith('- '):
            blocks.append('<ul>'+''.join('<li>'+inline(line[2:])+'</li>' for line in block.splitlines())+'</ul>')
        elif block.strip():
            blocks.append('<p>'+inline(block)+'</p>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Online survey and disturbance-aware monitoring</title>'
            '<style>body{max-width:880px;margin:40px auto;padding:20px;color:#233e36;'
            'font:15px/1.75 system-ui}h1{font-size:30px;line-height:1.25}'
            'h2{font-size:21px;margin-top:32px}a{color:#176187}li{margin:10px 0}</style>'
            '<a href="../../index.html#survey-review">← Back to simulations</a>'+''.join(blocks)+'</html>')


def describe_run(record, environment):
    samples = np.asarray(record['samples'])
    summary = dict(record['summary'])
    final_time = float(samples[-1, 0])
    visits, arrivals, entries, dwell = [], {}, {}, {}
    for region in environment['regions']:
        label = region['label']
        bounds = np.asarray(region['bounds'])
        inside = np.all((samples[:, 1:3] >= bounds[0]) & (samples[:, 1:3] <= bounds[1]), axis=1)
        transitions = np.flatnonzero(inside & ~np.r_[False, inside[:-1]])
        arrivals[label] = float(samples[transitions[0], 0]) if len(transitions) else None
        entries[label] = int(len(transitions))
        dwell[label] = float(np.sum(np.diff(samples[:, 0])*inside[1:]))
        if inside.any():
            visits.append(label)
    summary.update(regions_visited=visits, first_region_arrival_s=arrivals,
                   first_A_arrival_s=arrivals.get('A'), region_entries=entries, region_dwell_s=dwell,
                   first_60s_mean_deficit=float(np.interp(60, samples[:, 0], samples[:, 14])/60),
                   last_60s_mean_deficit=float((samples[-1, 14]-np.interp(final_time-60, samples[:, 0], samples[:, 14]))/60))
    mode_counts = dict(survey=0, monitoring=0)
    mode_time = dict(survey=0., monitoring=0.)
    probes, prefix_errors = [], []
    for index, plan in enumerate(record['plans']):
        if plan['t'] >= final_time:
            continue
        mode = 'survey' if plan['exploring'] else 'monitoring'
        mode_counts[mode] += 1
        end = min(final_time, record['plans'][index+1]['t'] if index+1 < len(record['plans']) else final_time)
        mode_time[mode] += max(0., end-plan['t'])
        benefits = np.asarray(plan.get('task_relevance_benefits', []))
        if end-plan['t'] >= 5.-1e-7:
            actual = np.array([np.interp(plan['t']+5., samples[:, 0], samples[:, axis]) for axis in (1, 2)])
            predicted = np.asarray(plan['predicted_states'])[:, 20, :2]
            prefix_errors.append(float(np.linalg.norm(predicted-actual, axis=1).mean()))
        if benefits.size:
            assert benefits.shape == (3, 3)
            probes.append(dict(t=plan['t'], exploring=bool(plan['exploring']),
                               max_cost_improvement=float(benefits[:, 0].max()),
                               max_tracking_improvement_m=float(benefits[:, 1].max()),
                               rescued_contacts=int(np.count_nonzero(benefits[:, 2] > 0)),
                               information_enabled=bool(plan['information_enabled']),
                               route_values=plan['route_values'],
                               amortization_factor=plan['amortization_factor'],
                               selected_learning_bonus=plan['validation_bonuses'][plan['selected_candidate_index']],
                               selected_task_cost=plan['expected_cost'],
                               route_wind_std_m_s=plan['route_wind_std'],
                               route_force_std_m_s2=plan['route_force_std']))
    frames = record['belief_frames']
    changed = sum(not np.array_equal(a['mean'], b['mean']) for a, b in zip(frames, frames[1:]))
    first_mean, last_mean = np.asarray(frames[0]['mean']), np.asarray(frames[-1]['mean'])
    free = np.asarray(environment['free_indices'])
    summary.update(mode_counts=mode_counts, mode_time_s=mode_time,
                   wind_belief_mean_changed_frames=changed,
                   wind_belief_frame_count=len(frames),
                   mean_wind_change_m_s=float(np.mean(np.linalg.norm((last_mean-first_mean)[free], axis=1))),
                   first_median_wind_std_m_s=float(np.median(np.asarray(frames[0]['std'])[free])),
                   final_median_wind_std_m_s=float(np.median(np.asarray(frames[-1]['std'])[free])),
                   relevance_probe_calls=len(probes),
                   information_enabled_calls=sum(p['information_enabled'] for p in probes),
                   committed_endpoint_prediction_mean_error_m=float(np.mean(prefix_errors)) if prefix_errors else None,
                   committed_endpoint_prediction_max_error_m=max(prefix_errors) if prefix_errors else None,
                   relevance_probe_useful_calls=sum(p['max_cost_improvement'] > .01 or
                                                     p['max_tracking_improvement_m'] > .10 or
                                                     p['rescued_contacts'] > 0 for p in probes))
    return summary, probes


def reproduction_commands(protocols):
    commands = []
    for ident, protocol in protocols.items():
        cfg = protocol['legacy_settings']
        commands.append('python scripts/airflow/run_survey_monitoring.py '
                        f'--scenario {protocol["scenario"]} --method {protocol["method"]} '
                        f'--mode {protocol["exploration_mode"]} --seconds 300 --horizon {cfg["horizon"]} '
                        f'--levels {cfg["iterations"]} --proposals {cfg["proposals"]} '
                        f'--worlds {cfg["worlds"]} --validation {cfg["validation_worlds"]} '
                        f'--proposal-batch {cfg["batch_size"]} --validation-batch {cfg["validation_batch_size"]} '
                        f'--cost-allowance {protocol["cost_allowance"]} --seed {protocol["seed"]} '
                        f'--plan-limit {protocol["runtime_limit_s"]} --matched-prior '
                        '--site scenario-site --output reproduced')
    commands.append('python scripts/airflow/audit_survey_monitoring.py --directory reproduced '
                    '--site scenario-site --planner-replay all --wall-limit 180')
    return commands


def package(directory, site):
    audit_file = directory/'survey-audit.json'
    audit = json.loads(audit_file.read_text())
    verified = {case['id']: case for case in audit['checks']}
    expected = {f's{scenario}-{method}' for scenario in (1, 2) for method in ('aware', 'baseline')}
    assert set(verified) == expected, 'Package requires all four audited final cases'
    assert all(case['completed'] and case['simulated_seconds'] == 300 for case in verified.values())
    assert all(not case['sources']['unrecorded_transitive_dependencies'] for case in verified.values()), 'Missing dependency provenance'
    out = site/'data'/DESTINATION
    out.mkdir(parents=True, exist_ok=True)
    environment = json.loads((directory/'environment.json').read_text())
    runs, protocols, all_probes, rejected = [], {}, {}, []
    warm, first, initialization, sampling, optimization, diagnostic, total = ([] for _ in range(7))
    recorded_hashes = {}
    for scenario in (1, 2):
        for method in ('aware', 'baseline'):
            ident = f's{scenario}-{method}'
            record = json.loads((directory/f'{ident}.json').read_text())
            protocol = json.loads((directory/f'{ident}-protocol.json').read_text())
            protocols[ident] = protocol
            assert record['summary']['completed'] and record['summary']['simulated_seconds'] == 300
            assert protocol['exploration_mode'] == ('survey' if method == 'aware' else 'passive')
            for relative, recorded in protocol['source_sha256'].items():
                assert digest(ROOT/relative) == recorded, ('Current packaged source differs from run', ident, relative)
                assert relative not in recorded_hashes or recorded_hashes[relative] == recorded
                recorded_hashes[relative] = recorded
            summary, probes = describe_run(record, environment)
            record['summary'] = summary
            all_probes[ident] = probes
            for plan in record['plans']:
                timing = plan['timing']
                if timing['compilation_included']:
                    first.append(plan['wall_seconds'])
                else:
                    warm.append(plan['wall_seconds'])
                    initialization.append(timing['initialization'])
                    sampling.append(timing['field_sampling'])
                    optimization.append(timing['optimization_validation'])
                    diagnostic.append(timing.get('diagnostic_prediction_after_selection', 0.))
                    total.append(plan['wall_seconds']+timing.get('diagnostic_prediction_after_selection', 0.))
            rejected.extend(dict(id=ident, **call) for call in protocol.get('rejected_planning_calls', []))
            payload = json.dumps(record, separators=(',', ':')).encode()
            encoded = gzip.compress(payload, mtime=0)
            assert gzip.decompress(encoded) == payload
            filename = ident+'.json.gz'
            (out/filename).write_bytes(encoded)
            shutil.copyfile(directory/f'{ident}-protocol.json', out/f'{ident}-protocol.json')
            runs.append(dict(id=ident, scenario=scenario, method=method, file=filename,
                             decoded_sha256=hashlib.sha256(payload).hexdigest(), summary=summary))
    for name in ('environment.json', 'survey-audit.json'):
        shutil.copyfile(directory/name, out/name)
    (out/'relevance-probes.json').write_text(json.dumps(all_probes, indent=2))
    contact_free = sum(run['summary']['contact'] is None for run in runs)
    reached = sum('A' in run['summary']['regions_visited'] for run in runs)
    outcome = f'All four 300 s simulations completed; {contact_free} had no contact and {reached} reached Region A.'
    for scenario in (1, 2):
        pair = {run['method']: run['summary'] for run in runs if run['scenario'] == scenario}
        change = 100*(pair['aware']['integrated_deficit']/pair['baseline']['integrated_deficit']-1)
        arrival = pair['aware']['first_A_arrival_s']
        arrival_text = f'reached A at {arrival:.1f} s' if arrival is not None else 'did not reach A'
        outcome += (f' Scenario {scenario}: online survey {arrival_text}; its monitoring cost was '
                    f'{abs(change):.2f}% {"higher" if change > 0 else "lower"} than monitoring only.')
    outcome += ' These are paired development runs, not evidence of a general performance guarantee.'
    runtime = dict(warm_min_s=min(warm), warm_max_s=max(warm), warm_median_s=float(np.median(warm)),
                   warm_calls=len(warm), initialization_median_s=float(np.median(initialization)),
                   field_sampling_median_s=float(np.median(sampling)), optimization_median_s=float(np.median(optimization)),
                   diagnostic_prediction_median_s=float(np.median(diagnostic)), selection_plus_diagnostic_max_s=max(total),
                   first_call_min_s=min(first), first_call_max_s=max(first), limit_s=1., rejected_calls=rejected,
                   scope=protocol['runtime_scope'])
    assert max(warm) <= 1., 'Warm plan overrun must be rejected and explicitly recorded'
    cfg = protocol['legacy_settings']
    settings = protocol['settings'] | dict(horizon_seconds=cfg['horizon'], commit_seconds=cfg['commit'])
    manifest = dict(duration_s=300, state='complete', implementation='survey-mbd',
                    planner_settings=settings, legacy_settings=cfg, initial_wind_knowledge='cold-start',
                    execution_belief=protocol['execution_belief'], planning_safety=protocol['planning_safety'],
                    validation_safety=protocol['validation_safety'], prior_configuration=protocol['prior_configuration'],
                    method_labels=dict(aware='Online survey + monitoring', baseline='Monitoring only'),
                    method_descriptions=dict(aware='Survey task routes when wind knowledge can improve their execution; otherwise minimize clarity deficit.',
                                             baseline='Minimize clarity deficit with the same wind estimator, geometric starts and execution filter.'),
                    survey_gate=protocol['survey_gate'], inducing_allocation=protocol['inducing_allocation'],
                    outcome=outcome, runtime=runtime, runs=runs, verification='survey-audit.json',
                    source_available=True, method_document='method.html', relevance_probe_data='relevance-probes.json')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    document = ROOT/'docs/airflow/online_survey_monitoring.md'
    (out/'method.html').write_text(render_method(document.read_text()))
    commands = reproduction_commands(protocols)
    archive_sources = {str(path.relative_to(ROOT)) for path in (ROOT/'src').rglob('*.py')}
    archive_sources.update(recorded_hashes)
    archive_sources.update('scripts/airflow/'+name for name in
                           ('package_survey_monitoring.py', 'audit_survey_monitoring.py', 'observed_poam.py',
                            'supported_poam.py', 'survey_information.py', 'survey_routes.py', 'survey_monitoring.py',
                            'run_survey_monitoring.py', 'active_sensing_monitoring.py', 'exploration_monitoring.py',
                            'survey_forecast.py'))
    archive_sources.update('tests/'+name for name in
                           ('test_survey_information.py', 'test_survey_routes.py', 'test_survey_monitoring.py',
                            'test_observed_poam.py', 'test_active_sensing_monitoring.py', 'test_survey_forecast.py'))
    source_hashes = {relative: digest(ROOT/relative) for relative in sorted(archive_sources)}
    (out/'source-sha256.json').write_text(json.dumps(source_hashes, indent=2))
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(archive_sources):
            archive.write(ROOT/relative, relative)
        archive.write(document, 'METHOD.md')
        for name in ('pyproject.toml', 'uv.lock'):
            if (ROOT/name).exists():
                archive.write(ROOT/name, name)
        for path in out.glob('*.json'):
            archive.write(path, 'results/'+path.name)
        for run in runs:
            raw = np.load(directory/f'{run["id"]}-execution.npz')
            buffer = io.BytesIO()
            np.savez_compressed(buffer, samples=raw['samples'], observation_rows=raw['observation_rows'],
                                support_mask=raw['support_mask'])
            archive.writestr(f'results/{run["id"]}-controls-observations.npz', buffer.getvalue())
        for scenario in (1, 2):
            for name in ('stationary-velocity.f32', 'metadata.json'):
                archive.write(site/f'data/scenario{scenario}/{name}', f'scenario-site/data/scenario{scenario}/{name}')
        archive.write(site/'app.js', 'scenario-site/app.js')
        archive.writestr('REPRODUCE.txt',
                        'Install pyproject.toml dependencies. Set PYTHONPATH=src:scripts/airflow.\n\n'+
                        '\n'.join(commands)+'\n\nEach simulation should have an external 240 s wall cap. '
                        'The runner stops before applying any warm plan above one second; compilation is separate. '
                        'Explicit resumes must retain rejected calls. Complete planning/state/clarity archives '
                        'are regenerated by these commands; losslessly compressed viewer data is served beside this archive.\n')
    assert (out/'source.zip').stat().st_size < 4*1024**2, 'Unexpected source archive growth'
    print(json.dumps(dict(directory=str(out), runtime=runtime, outcome=outcome,
                          archive_bytes=(out/'source.zip').stat().st_size)), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--site', type=Path, required=True)
    args = parser.parse_args()
    package(args.directory, args.site)
