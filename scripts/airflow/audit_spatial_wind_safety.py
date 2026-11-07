"""Read-only audit of recorded wind coverage; never changes a planner or replay.

True wind is used only here, for offline diagnostics. Coverage is evaluated at
saved belief-map cell centers, not claimed as continuous-space/time coverage.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]


def audit(records, site):
    environment = json.loads((records / 'environment.json').read_text())
    free = np.asarray(environment['free_indices'], dtype=int)
    cells = np.asarray(environment['cells'])
    reports = []
    for scenario in (1, 2):
        source = records / f's{scenario}-center-baseline1.json.gz'
        record = json.loads(gzip.decompress(source.read_bytes()))
        folder = site / f'data/scenario{scenario}'
        metadata = json.loads((folder / 'metadata.json').read_text())
        if metadata['time_dependence'] != 'none':
            raise ValueError('Temporal field is incompatible with this spatial-only audit')
        data = folder / metadata['data_file']
        nx, ny, _ = metadata['grid']
        if list(environment['grid']) != [nx, ny]:
            raise ValueError('Recorded belief and physical wind grids differ')
        truth = np.fromfile(data, dtype='<f4').reshape(-1, 3)[:, :2][free].astype(float)
        # On these cell centers this equals FixedWind's bilinear interpolation.
        expected_cells = (np.c_[free % nx, free // nx] + .5) * metadata['dx_m']
        np.testing.assert_allclose(cells, expected_cells, atol=1e-12)
        frames = []
        for frame in record['belief_frames']:
            mean = np.asarray(frame['mean'])[free]
            std = np.asarray(frame['std'])[free]
            error = np.linalg.norm(mean - truth, axis=1)
            radius = 3 * np.sqrt(std**2 + .1**2)
            frames.append(dict(t=frame['t'], vector_rmse_m_s=float(np.sqrt(np.mean(error**2))),
                mean_variance_m2_s2=float(np.mean(std**2)),
                confidence_disk_coverage=float(np.mean(error <= radius)),
                maximum_error_to_radius=float(np.max(error / radius)),
                maximum_excess_m_s=float(np.max(error - radius))))
        observations = np.asarray(record['observations'])
        distance, _ = cKDTree(observations[:, 1:3]).query(cells)
        final_error = np.linalg.norm(np.asarray(record['belief_frames'][-1]['mean'])[free] - truth, axis=1)
        proximity = {}
        for label, mask in [('within_0.25_m', distance <= .25), ('beyond_0.25_m', distance > .25)]:
            proximity[label] = dict(cell_fraction=float(np.mean(mask)),
                vector_rmse_m_s=float(np.sqrt(np.mean(final_error[mask]**2))) if mask.any() else None)
        samples = np.asarray(record['samples'])
        violations = np.flatnonzero(samples[1:, 11] == 0) + 1
        reports.append(dict(scenario=scenario, recorded_run_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            wind_data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(), time_dependence=metadata['time_dependence'],
            free_cells=len(free), max_true_wind_speed_on_free_grid_m_s=float(np.linalg.norm(truth, axis=1).max()),
            initial=frames[0], final=frames[-1], minimum_coverage=min(f['confidence_disk_coverage'] for f in frames),
            measurement_proximity=proximity, contact=record['summary']['contact'],
            infeasible_commands=len(violations), first_infeasible_time_s=float(samples[violations[0], 0]) if len(violations) else None,
            maximum_logged_slack=float(samples[:, 9].max()),
            maximum_control_component_m_s2=float(np.max(np.abs(samples[:, 5:7]))), frames=frames))
    return dict(scope='Recorded, rounded POAM maps at 2361 free cell centers; not a continuous or simultaneous confidence guarantee.',
        uncertainty='Epistemic posterior over a single fixed spatial vector field. Independent measurement noise sigma=0.1 m/s per component.',
        safety_disk='||w-mu||_2 <= 3*sqrt(sigma_post^2+0.1^2) m/s; a statistical assumption, not a physical hard bound.',
        prior_radius_m_s=float(3*np.sqrt(1.8**2+.1**2)),
        prior_max_wind_acceleration_at_rest_m_s2=float((.1/.680)*9*(1.8**2+.1**2)),
        actuator_component_bound_m_s2=1.6,
        actuator_norm_bound_m_s2=float(np.sqrt(2)*1.6),
        backup_horizon_s=1.2, backup_step_s=.15, construction_fields=2,
        safety_margin_m=.06, cbf_alpha_per_s=1., boundary_hocbf_rate_per_s=1.5,
        reports=reports)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, default=ROOT/'results/airflow/rmse_learning_20261104/checked300')
    parser.add_argument('--site', type=Path, default=ROOT.parent/'sites/belief-adaptive-monitoring/dist/revised')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.records, args.site)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    for r in report['reports']:
        print(json.dumps({k: v for k, v in r.items() if k != 'frames'}, indent=2))


if __name__ == '__main__':
    main()
