"""Post hoc full evaluation of the three next-ranked MBD proposals; no mission changes."""

import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.planning.budgeted import evaluate
from voi_diffusion_ipp.planning.sampling import couple_outer_samples
from voi_diffusion_ipp.viz.demo_data import serializable
from voi_diffusion_ipp.wind.poam import POAM

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT/'results/budgeted_missions/antithetic_reversed_01/proposed.npz'
    raw = json.loads(path.with_suffix('.json').read_text())
    meta = raw['metadata']
    assert raw['complete'] and source_hashes() == meta['source_sha256']
    assert raw['periods'][8]['backup_used']
    cfg = Config(**meta['config'])
    out = ROOT/'docs/wind_route_study/rejection_alternatives_01'
    out.mkdir(exist_ok=False)
    protocol = {'label': 'Post hoc diagnosis of the first observed cached/full rejection',
                'period': 8, 'time_s': 16, 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'next_distinct_proposals': 3, 'source_sha256': source_hashes(),
                'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'limits': 'A selected numerical failure, not a representative cache-error estimate. No retuning or rerun of the mission. Evaluate the next three distinct finite MBD proposals in cached-cost order using the original uncached checker. This is a failure-selected diagnostic, not a controller change or full planning latency benchmark.'}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    env, template, _, _, _ = make_two_region(meta['task_spec'], cfg)
    model = POAM.from_data(data['initial_observation_positions'], data['initial_observations'],
                          jax.random.PRNGKey(600000), len(template.z), noise_std=cfg.observation_std,
                          prior_std=cfg.wind_std, spectral_count=cfg.poam_spectral_nodes,
                          prior_support=template.prior_support)
    root = jax.random.PRNGKey(meta['seed'])
    for i in range(8):
        sl = slice(i*cfg.stage_steps, (i+1)*cfg.stage_steps)
        model.update(data['observation_positions'][sl], data['observations'][sl],
                     jax.random.fold_in(root, 500000+i), train_steps=cfg.poam_train_steps)
    b = model.planning_belief()
    mu, co = jax.jit(lambda v: v.predict(env.cells))(b)
    np.testing.assert_allclose(mu, data['mean'][8], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(jnp.sqrt(co[:, 0, 0]), data['std'][8], atol=3e-5, rtol=3e-5)
    keys = jax.random.split(jax.random.fold_in(root, 108), 4)
    eps = jax.random.normal(keys[0], (cfg.worlds+cfg.inner_samples, *b.sample_shape))
    ns = jax.random.normal(keys[1], (cfg.worlds, cfg.stage_steps+cfg.future_steps, 2))
    eps, ns = couple_outer_samples(eps, ns, cfg.worlds, meta['outer_sampling'])
    x, q = [jnp.asarray(data[k][8*40]) for k in ['x', 'q']]
    y = jnp.asarray(raw['periods'][8]['vector'])
    verify = jax.jit(lambda yy, xx, qq, bb, ee, nn: evaluate(
        yy, xx, qq, bb, None, nn, env, cfg, True, True, ee
    ))
    full = verify(y, x, q, b, eps, ns)
    assert not np.isfinite(full[0])
    original = np.asarray(raw['periods'][8]['selected_model_path'])
    replay = np.concatenate((np.asarray(full[1][0].x[0]), np.asarray(full[1][1].x[0, 1:])))
    np.testing.assert_allclose(original, replay, atol=3e-5, rtol=3e-5)
    results = {'protocol': protocol, 'full_objective': float(full[0]),
               'full_min_clearance': float(min(s.clearance.min() for s in full[1])), 'rows': []}
    proposals = np.asarray(raw['periods'][8]['history']['candidates']).reshape(-1, len(y))
    costs = np.asarray(raw['periods'][8]['history']['costs']).ravel()
    candidates = [int(k) for k in np.argsort(costs) if np.isfinite(costs[k])
                  and not np.array_equal(proposals[k], np.asarray(y))][:3]
    paths = []
    for rank, k in enumerate(candidates, 1):
        begin = time.perf_counter()
        c, segments = jax.block_until_ready(verify(jnp.asarray(proposals[k]), x, q, b, eps, ns))
        seconds = time.perf_counter()-begin
        row = {'alternative_rank': rank, 'proposal_index': k, 'cached_cost': float(costs[k]),
               'full_cost': float(c), 'finite': bool(np.isfinite(c)),
               'min_clearance': float(min(s.clearance.min() for s in segments)),
               'warm_verification_s': seconds,
               'original_cycle_plus_this_check_s': raw['periods'][8]['planning_cycle_s']+seconds}
        paths.append(np.concatenate((np.asarray(segments[0].x), np.asarray(segments[1].x[:, 1:])),axis=1))
        results['rows'].append(row)
        print(row, flush=True)
    np.savez_compressed(out/'alternatives.npz', vectors=proposals[candidates], paths=np.stack(paths))
    (out/'summary.json').write_text(json.dumps(serializable(results), indent=2)+'\n')


if __name__ == '__main__':
    main()
