"""Read saved results only; never imports JAX or starts a simulation."""
import argparse
import json
from pathlib import Path

from package_survey_monitoring import describe_run


def summarize(directory, identifiers=None):
    environment = json.loads((directory/'environment.json').read_text())
    if identifiers is None:
        identifiers = [p.name.removesuffix('-protocol.json') for p in sorted(directory.glob('s?-*-protocol.json'))]
    rows = []
    for ident in identifiers:
        path = directory/(ident+'.json')
        if not path.exists():
            rows.append(dict(id=ident, state='no published checkpoint'))
            continue
        record = json.loads(path.read_text())
        summary, _ = describe_run(record, environment)
        selected = ('completed', 'simulated_seconds', 'contact', 'integrated_deficit',
                    'first_A_arrival_s', 'region_entries', 'region_dwell_s', 'mode_time_s',
                    'min_clearance_m', 'infeasible_execution_steps', 'max_execution_slack',
                    'warm_plan_max_seconds', 'supported_inducing_points', 'support_capacity_saturated',
                    'information_enabled_calls', 'committed_endpoint_prediction_mean_error_m',
                    'committed_endpoint_prediction_max_error_m', 'stop_reason')
        rows.append(dict(id=ident, **{key: summary.get(key) for key in selected}))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--ids', nargs='+')
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory, args.ids), indent=2))
