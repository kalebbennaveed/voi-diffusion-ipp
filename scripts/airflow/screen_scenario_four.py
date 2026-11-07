"""Bounded geometry-only screen of Scenario 4 with the unchanged seed generator.

No wind field, GP update, MBD optimization or robot execution is synthesized.
Every displayed path and acquisition mask comes from the current implementation.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from cluster_environment import environment
from geometric_task_mask import candidate_path_weights
from monitoring_routes import coarse_paths, clearance, visible
from task_wind_acquisition import deficit_weights

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'configs/scenarios/west_monitoring_yard_20261107.json'
OUT = ROOT / 'output/scenario4_design_20261107'


def connectivity(nodes, shape, obstacles, bounds, margin=.10):
    nx, ny = shape
    valid = clearance(nodes, obstacles, bounds) >= margin
    ids = np.arange(len(nodes)).reshape(ny, nx)
    rows, cols = [], []
    for di, dj in [(1, 0), (0, 1), (1, 1), (-1, 1)]:
        i0, i1 = max(0, -di), min(nx, nx-di)
        j0, j1 = max(0, -dj), min(ny, ny-dj)
        a = ids[j0:j1, i0:i1].ravel()
        b = ids[j0+dj:j1+dj, i0+di:i1+di].ravel()
        accepted = valid[a] & valid[b] & visible(nodes[a], nodes[b], obstacles, bounds, margin)
        rows.extend([a[accepted], b[accepted]])
        cols.extend([b[accepted], a[accepted]])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nodes), len(nodes)))
    _, labels = connected_components(graph[valid][:, valid], directed=False)
    sizes = np.bincount(labels)
    return dict(clearance_margin_m=margin, components=len(sizes),
                safe_nodes=int(valid.sum()), largest_component=int(sizes.max()),
                connected_fraction=float(sizes.max()/valid.sum()))


def segment_clearance(path, obstacles, bounds):
    worst = float(clearance(path, obstacles, bounds).min())
    for a, b in zip(path[:-1], path[1:]):
        d = b-a
        if np.dot(d, d) < 1e-16:
            continue
        fraction = np.clip((obstacles[:, :2]-a)@d/np.dot(d, d), 0., 1.)
        c = np.linalg.norm(a+fraction[:, None]*d-obstacles[:, :2], axis=1)-obstacles[:, 2]
        worst = min(worst, float(c.min()))
    return worst


def main():
    began = perf_counter()
    cfg = SimpleNamespace(region_target=.7, background_target=.35,
        clarity_decay=.16, background_decay=.02, safety_margin=.06,
        horizon_seconds=15., horizon=15., sensor_sigma=.5)
    env, grid, spec = environment(cfg, SPEC)
    nodes = np.asarray(grid['all_cells'])
    free = np.asarray(grid['free_indices'])
    obstacles, bounds = np.asarray(env.obstacles), np.asarray(env.bounds)
    poses = [('B · center', [1.95, 1.0]), ('B · west', [.8, .9]),
             ('B · east', [2.3, 1.2]), ('A · center', [1.5, 4.95]),
             ('A · west', [.8, 4.7]), ('A · east', [2.3, 5.15])]
    clarity_cases = [('Both need monitoring', .35, .35, .74),
                     ('A needs more', .15, .65, .74),
                     ('B needs more', .65, .15, .74),
                     ('Low clarity everywhere', .10, .10, .10)]
    cases = []
    union = np.zeros(len(nodes), bool)
    # Include states on the actual long alternative, not only favorable starts.
    initial_q = np.full(len(env.cells), .74)
    initial_q[grid['membership'].any(axis=1)] = .35
    seed_paths, _ = coarse_paths(np.asarray(spec['initial_state']), initial_q, env, cfg)
    longest = max(seed_paths, key=lambda p: np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    for i, p in enumerate(longest[1:-1]):
        poses.append((f'Outer alternative · point {i+1}', p.tolist()))
    for pose_name, position in poses:
        for clarity_name, q_a, q_b, q_bg in clarity_cases:
            x = np.r_[position, 0., 0.]
            q = np.full(len(env.cells), q_bg)
            q[grid['membership'][:, 0]] = q_a
            q[grid['membership'][:, 1]] = q_b
            paths, geometry = coarse_paths(x, q, env, cfg)
            need = deficit_weights(q, env.targets, env.process_noise, cfg.horizon)
            weights, details = candidate_path_weights(nodes, paths, env.cells, need,
                free, cfg.sensor_sigma, tube_radius=.25)
            active = weights > 0
            union |= active
            gaps = [segment_clearance(p, obstacles, bounds) for p in paths]
            assert all(visible(p[:-1], p[1:], obstacles, bounds, .10).all() for p in paths)
            assert abs(weights.sum()-1) < 1e-10
            cases.append(dict(id=len(cases), pose=pose_name, clarity=clarity_name,
                state=x.tolist(), q_A=q_a, q_B=q_b, q_background=q_bg,
                paths=[p.tolist() for p in paths], route_values=details,
                active_indices=np.flatnonzero(active).tolist(),
                weights=[float(v) for v in weights[active]],
                active_fraction=float(active.sum()/len(free)),
                eastern_yard_weight=float(weights[nodes[:, 0] >= 5.].sum()),
                minimum_path_clearance_m=min(gaps), geometry=geometry))
    connected = connectivity(nodes, grid['grid'], obstacles, bounds)
    assert connected['components'] == 1, 'An inaccessible yard would invalidate the comparison'
    # Repeated cold graph construction, same start/clarity, excluding Python imports.
    timings = []
    for _ in range(20):
        start = perf_counter()
        paths, _ = coarse_paths(np.asarray(spec['initial_state']), initial_q, env, cfg)
        need = deficit_weights(initial_q, env.targets, env.process_noise, cfg.horizon)
        candidate_path_weights(nodes, paths, env.cells, need, free, cfg.sensor_sigma, .25)
        timings.append(perf_counter()-start)
    eastern = nodes[free, 0] >= 5.
    result = dict(status='geometry only; no wind or method-performance result',
        environment=spec, grid=grid['grid'], nodes=nodes.tolist(), free_indices=free.tolist(),
        cases=cases, union_active_indices=np.flatnonzero(union).tolist(),
        summary=dict(tested_cases=len(cases), start_positions=len(poses),
            free_cells=len(free), free_area_m2=len(free)*spec['dx']**2,
            initial_active_fraction=cases[0]['active_fraction'],
            minimum_active_fraction=min(c['active_fraction'] for c in cases),
            maximum_active_fraction=max(c['active_fraction'] for c in cases),
            union_active_fraction=float(union.sum()/len(free)),
            eastern_yard_free_fraction=float(eastern.mean()),
            eastern_yard_union_fraction=float(union[free][eastern].mean()),
            maximum_eastern_yard_weight=max(c['eastern_yard_weight'] for c in cases),
            minimum_path_clearance_m=min(c['minimum_path_clearance_m'] for c in cases),
            connectivity=connected,
            geometry_and_mask_ms=dict(median=float(np.median(timings)*1000),
                maximum=float(max(timings)*1000), repetitions=len(timings)),
            total_screen_wall_seconds=perf_counter()-began),
        interpretation='Zero mask support is not zero GP covariance. Remote observations can still help through learned correlations. These geometric tests do not establish faster learning, safety, or better monitoring.',
        invariants=['Same 8 × 6 m workspace and 0.125 m grid',
            'Unchanged online geometric initializer and 0.25 m C2 task tubes',
            'No wind information used in this audit',
            'All eight obstacles and all exterior walls checked analytically',
            'Whole-map unknown-wind prior retained for both methods',
            'Background target 0 is an explicit shared task specification'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'geometry-audit.json').write_text(json.dumps(result, indent=2,
        allow_nan=False, default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist()))
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
