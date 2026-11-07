"""Export recorded geometric starts; no planning, CFD, or mission reruns."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT.parent/'sites/belief-adaptive-monitoring/dist/revised'
RAW = ROOT/'results/airflow/cluster_revision_20261103/v2/missions300'
TIMES = [0, 5, 10, 30, 60, 90, 120, 180, 240, 295]


def reference(theta, x):
    """Same piecewise-constant reference dynamics; display reconstruction only."""
    a = 1.6*np.tanh(np.asarray(theta, dtype=np.float64).reshape(-1, 2))
    state = np.asarray(x, dtype=np.float64).copy()
    points = [state[:2].copy()]
    for u in a:
        state = np.r_[state[:2]+.25*state[2:]+.5*.25**2*u, state[2:]+.25*u]
        points.append(state[:2].copy())
    return np.round(points, 6).tolist()


def main():
    directory = SITE/'data/monitoring-cluster-300'
    env = json.loads((directory/'environment.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    nx, ny = env['grid']
    cells = np.asarray(env['cells'])
    Q = np.asarray(env['process_noise'])
    target = np.asarray(env['targets'])
    free = np.asarray(env['free_indices'])
    output = SITE/'data/geometric-review'
    output.mkdir(exist_ok=True)
    sources = {}
    for run in manifest['runs']:
        if run['method'] != 'aware':
            continue
        with gzip.open(directory/run['file'], 'rt') as f:
            record = json.load(f)
        frames = []
        for t in TIMES:
            plan = next(p for p in record['plans'] if p['t'] == t)
            archive = RAW/f"{run['id']}-plan-{int(t/5):03d}.npz"
            sources[archive.name] = hashlib.sha256(archive.read_bytes()).hexdigest()
            with np.load(archive) as saved:
                q = saved['initial_clarity'].astype(float)
                state = saved['initial_state']
                params = saved['search_initial_parameters']
                assert params.shape == (8, 120)
                assert len(plan['initialization']['coarse_paths']) == 6
                # This is the initializer's approximate footprint value, not
                # the full MBD clarity objective or a wind-uncertainty map.
                future = q[None]/(1+np.array([3.75, 7.5, 15.])[:, None]*Q[None]*q[None])
                need = np.maximum(target[None]-future, 0).mean(axis=0)
                raw = np.zeros(nx*ny)
                raw[free] = need
                value = gaussian_filter(raw.reshape(ny, nx), .5/(np.sqrt(2)*.125), mode='constant')
                seeds = [reference(th, state) for th in params]
                selected = reference(saved['theta'], state)
                match = np.flatnonzero(np.all(params == saved['theta'][None], axis=1)).tolist()
            frames.append(dict(t=t, state=np.round(state, 6).tolist(),
                sensing_value=np.round(value, 6).ravel().tolist(),
                coarse=plan['initialization']['coarse_paths'],
                destinations=plan['initialization']['initial_destinations'],
                seeds=seeds, seed_costs=plan['seed_costs'],
                selected_reference=selected,
                forecast=[p[:2] for p in plan['predicted_states'][0]],
                executed=[r[1:3] for r in record['samples'] if t <= r[0] <= t+5],
                selected_origin=plan['selected_origin'], selected_seed_indices=match,
                selected_cost=plan['expected_cost'], forecast_contact=plan['predicted_contacts'][0],
                initialization=plan['initialization']))
        data=dict(id=run['id'], scenario=run['scenario'], case=run['case'], frames=frames)
        payload=json.dumps(data,separators=(',', ':'),allow_nan=False).encode()
        with gzip.GzipFile(output/f"{run['id']}.json.gz", 'wb', mtime=0) as f:
            f.write(payload)
    (output/'manifest.json').write_text(json.dumps(dict(
        conditions=manifest['conditions'], times=TIMES, horizon_s=15, commit_s=5,
        seed_order=['Hold/brake', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'Shifted previous'],
        sensing_value_description='No-sensing deficit averaged at 3.75, 7.5, 15 s and convolved with the squared Gaussian sensor footprint. Initialization heuristic only; distance and clearance also enter destination ranking.',
        reference_description='Nominal double-integrator references reconstructed from saved acceleration parameters; these are not wind-affected trajectories.',
        trajectory_description='Forecast and first 5 s of execution copied from the existing recordings. No new simulations.',
        source_sha256=sources),separators=(',', ':')))
    print(json.dumps(dict(recordings=12, planning_calls=120, bytes=sum(p.stat().st_size for p in output.iterdir()))))


if __name__ == '__main__':
    main()
