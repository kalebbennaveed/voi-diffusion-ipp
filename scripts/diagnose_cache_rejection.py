"""Post hoc numerical diagnosis of a rejected cached forecast; no mission changes."""

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
from voi_diffusion_ipp.wind.cache import build_prior_cache
from voi_diffusion_ipp.wind.poam import POAM

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT/'results/budgeted_missions/antithetic_reversed_01/proposed.npz'
    raw = json.loads(path.with_suffix('.json').read_text())
    meta = raw['metadata']
    assert raw['complete'] and source_hashes() == meta['source_sha256']
    assert raw['periods'][8]['backup_used']
    cfg = Config(**meta['config'])
    out = ROOT/'docs/wind_route_study/cache_rejection_01'
    out.mkdir(exist_ok=False)
    protocol = {'label': 'Post hoc diagnosis of the first observed cached/full rejection',
                'period': 8, 'time_s': 16, 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'grid_widths': [129, 257, 513], 'source_sha256': source_hashes(),
                'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'limits': 'A selected numerical failure, not a representative cache-error estimate. No retuning or rerun of the mission. Compare original cache and finer caches to full continuous evaluation; keep every result.'}
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
    full = jax.jit(lambda yy, xx, qq, bb, ee, nn: evaluate(
        yy, xx, qq, bb, None, nn, env, cfg, True, True, ee
    ))(y, x, q, b, eps, ns)
    assert not np.isfinite(full[0])
    original = np.asarray(raw['periods'][8]['selected_model_path'])
    replay = np.concatenate((np.asarray(full[1][0].x[0]), np.asarray(full[1][1].x[0, 1:])))
    np.testing.assert_allclose(original, replay, atol=3e-5, rtol=3e-5)
    results = {'protocol': protocol, 'full_objective': float(full[0]),
               'full_min_clearance': float(min(s.clearance.min() for s in full[1])), 'rows': []}
    for width in protocol['grid_widths']:
        build = jax.jit(lambda belief, samples, w=width: build_prior_cache(belief, samples, w, True))
        cache = jax.block_until_ready(build(b, eps))
        begin = time.perf_counter()
        cache = jax.block_until_ready(build(b, eps))
        build_s = time.perf_counter()-begin
        cost, segments = jax.jit(lambda yy, xx, qq, bb, c, nn: evaluate(
            yy, xx, qq, bb, c, nn, env, cfg, True
        ))(y, x, q, b, cache, ns)
        trajectory = np.concatenate((np.asarray(segments[0].x), np.asarray(segments[1].x[:, 1:])), axis=1)
        full_path = np.concatenate((np.asarray(full[1][0].x), np.asarray(full[1][1].x[:, 1:])), axis=1)
        np.savez_compressed(out/f'grid_{width}.npz', path=trajectory, full_path=full_path,
                            first_clearance=np.asarray(segments[0].clearance),
                            future_clearance=np.asarray(segments[1].clearance))
        if width == 129:
            np.testing.assert_allclose(cost, raw['periods'][8]['cached_objective'], atol=3e-5, rtol=3e-5)
        row = {'grid_width': width, 'cost': float(cost),
               'min_clearance': float(min(s.clearance.min() for s in segments)),
               'max_state_error': float(np.max(np.abs(trajectory-full_path))),
               'warm_cache_build_s': build_s}
        results['rows'].append(row)
        print(row, flush=True)
        (out/'summary.json').write_text(json.dumps(serializable(results), indent=2)+'\n')


if __name__ == '__main__':
    main()
