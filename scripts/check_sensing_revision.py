"""Check that a sensing-range rerun preserves each original paired task."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--updated', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    before, after = [json.loads(p.read_text()) for p in (args.previous, args.updated)]
    rows, pending = [], []
    for case, entry in after['cases'].items():
        for method in ('aware', 'blind_filter'):
            name = f'two_{case}_{method}_9101.json'
            old_path = ROOT / before['cases'][case]['root'] / case / name
            new_path = ROOT / entry['root'] / case / name
            if not new_path.exists():
                pending.append(f'{case}/{method}')
                continue
            old, new = [json.loads(p.read_text()) for p in (old_path, new_path)]
            if not new['completed']:
                pending.append(f'{case}/{method}')
                continue
            old_cfg, new_cfg = old['config'].copy(), new['config'].copy()
            old_angle = old_cfg.pop('sensor_half_angle_degrees', 45.)
            new_angle = new_cfg.pop('sensor_half_angle_degrees', 45.)
            changes = {'sensor_half_angle_degrees': [old_angle, new_angle]}
            for key, values in entry.get('additional_configuration_changes', {}).items():
                assert [old_cfg.pop(key), new_cfg.pop(key)] == values, (case, method, key)
                changes[key] = values
            assert old_cfg == new_cfg, (case, method, 'Additional configuration changed')
            assert (old_angle, new_angle) == (45., 25.)
            for key in ('environment', 'frames', 'seed', 'mission_seconds'):
                a, b = (old[key][0], new[key][0]) if key == 'frames' else (old[key], new[key])
                assert a == b, (case, method, key)
            rows.append({'case': case, 'method': method, 'configuration_changes': changes,
                         'same_environment_initial_state_and_information': True,
                         'old_cost': old['cost'], 'new_cost': new['cost'],
                         'new_contact_free_completion': new['success'],
                         'old_radius_at_3_2m': float(3.2 * np.tan(np.deg2rad(old_angle))),
                         'new_radius_at_3_2m': float(3.2 * np.tan(np.deg2rad(new_angle)))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'status': 'complete' if not pending else 'running',
                                   'verified_runs': rows, 'pending': pending}, indent=2) + '\n')
    print(f'Verified {len(rows)} matched reruns; {len(pending)} pending')


if __name__ == '__main__':
    main()
