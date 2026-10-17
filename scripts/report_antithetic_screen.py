"""Audit both antithetic development pairs, retaining the earlier IID controls."""

import hashlib
import json
from pathlib import Path

import numpy as np
from report_wind_route_study import clarity_check, matched

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs/wind_route_study'


def main():
    status = json.loads((ROOT/'results/antithetic_screen_status_01.json').read_text())
    assert status['state'] == 'complete'
    results = {'label': 'Two exploratory development pairs; not held-out confirmation', 'cases': []}
    for name, old in [('antithetic_reversed_01', 'reversed_wind_01'),
                      ('antithetic_open_01', 'open_passage_wind_01')]:
        folder = ROOT/'results/budgeted_missions'/name
        paths = {m: folder/(m+'.npz') for m in ['proposed', 'current_belief']}
        assert matched(paths)
        methods = {}
        for method, path in paths.items():
            raw = json.loads(path.with_suffix('.json').read_text())
            assert raw['complete'] and raw['metadata']['outer_sampling'] == 'antithetic'
            prior = ROOT/'results/budgeted_missions'/old/(method+'.npz')
            old_raw = json.loads(prior.with_suffix('.json').read_text())
            assert old_raw['metadata'].get('outer_sampling', 'iid') == 'iid'
            for key in ['seed', 'physical_wind_spec', 'task_sha256', 'prior_regime',
                        'execution_dt', 'observation_dt', 'rapid_shared_initializations']:
                assert raw['metadata'][key] == old_raw['metadata'][key], key
            old_config = old_raw['metadata']['config'] | {'budgeted_policy': 'compact'}
            assert raw['metadata']['config'] == old_config
            with np.load(path, allow_pickle=False) as archive:
                data = {k: archive[k] for k in archive.files}
            with np.load(prior, allow_pickle=False) as previous:
                for key in ['initial_observation_positions', 'initial_observations', 'cells',
                            'physical_wind', 'targets', 'obstacles']:
                    np.testing.assert_array_equal(data[key], previous[key])
                np.testing.assert_array_equal(data['x'][0], previous['x'][0])
                np.testing.assert_array_equal(data['q'][0], previous['q'][0])
            a = json.loads(path.with_suffix('.audit.json').read_text())
            old_a = json.loads(prior.with_suffix('.audit.json').read_text())
            error = clarity_check(data, raw['metadata']['config'], raw['metadata']['execution_dt'])
            assert error < 2e-5
            methods[method] = {'audit': a, 'iid_audit': old_a, 'clarity_replay_max_error': error,
                               'antithetic_relative_to_iid_percent': 100*(
                                   a['integrated_whole_grid_deficit'] /
                                   old_a['integrated_whole_grid_deficit']-1),
                               'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        cp, cf = [methods[m]['audit']['integrated_whole_grid_deficit']
                  for m in ['proposed', 'current_belief']]
        results['cases'].append({'name': name, 'paired_proposed_regret_percent': 100*(cp/cf-1),
                                 'methods': methods})
    (DOC/'antithetic_results_01.json').write_text(json.dumps(results, indent=2)+'\n')
    for case in results['cases']:
        print(case['name'], 'paired proposed regret', case['paired_proposed_regret_percent'])
        for name, method in case['methods'].items():
            print(name, 'change vs iid', method['antithetic_relative_to_iid_percent'],
                  'contact intervals', method['audit']['contact_intervals'],
                  'timing', method['audit']['steady_cycle_s'])


if __name__ == '__main__':
    main()
