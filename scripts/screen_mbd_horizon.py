"""Planning-only horizon/geometry screen; every 0.1 s control stays independent.

No results in this script are flight outcomes or evidence of a learning gain.
All runs share one random seed and the unmodified monitoring/constraint cost.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.quarter.continuous import ControlSettings, optimizer, rollout
from voi_diffusion_ipp.quarter.environment import environment, TABLE_SHAPE

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--scales', nargs='+', type=float, default=[.35, .5])
p.add_argument('--horizons', nargs='+', type=float, default=[20., 30.])
p.add_argument('--iterations', type=int, default=100)
p.add_argument('--proposals', type=int, default=512)
a = p.parse_args()
out = Path(a.output); out.mkdir(exist_ok=True, parents=True)
rows = []
for scale in a.scales:
    for horizon in a.horizons:
        cfg = ControlSettings(environment_scale=scale, inspection_scale=.55,
            horizon=horizon, iterations=a.iterations, proposals=a.proposals,
            worlds=1, validation_worlds=1, temperature=1., batch_size=256,
            initialization='zero')
        env = environment('two', scale, .55)
        x = jnp.array([-28*scale, 8*scale, 0., 0.])
        q = jnp.full(len(env.cells), .35).at[:42].set(.65)
        mean = jnp.zeros(TABLE_SHAPE); construction = jnp.zeros((2, *TABLE_SHAPE))
        start = time.perf_counter()
        result = optimizer(env, cfg, 'blind')(x, q, mean, construction,
            mean[None], mean[None], jnp.zeros(2*cfg.steps),
            jax.random.PRNGKey(8101), jnp.array(1.), jnp.array(True))
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        theta, score, _, audit, _ = result
        predicted = jax.jit(lambda t: rollout(t, x, q, mean, construction,
            mean, env, cfg, False))(theta)
        points = np.asarray(predicted.x[:, :2])
        distances = np.linalg.norm(points - np.array([28, -10])*scale, axis=1)
        row = dict(status='planning_only', scale=scale, horizon=horizon,
            iterations=cfg.iterations, proposals=cfg.proposals, seed=8101,
            cost=float(score), initial_cost=float(audit.initial_cost),
            monitor=float(predicted.monitor), penalty=float(predicted.penalty),
            minimum_clearance=float(predicted.min_clearance),
            east_distance=float(distances.min()),
            closest_time=float(distances.argmin()*cfg.dt), seconds=elapsed,
            empty_batches=int(np.asarray(audit.trace)[:, -1].sum()),
            points=points.tolist(),
            source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        rows.append(row)
        (out/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
        np.savez_compressed(out/f'scale_{scale:g}_horizon_{horizon:g}.npz',
            theta=np.asarray(theta), initial_state=np.asarray(x),
            clarity=np.asarray(q), **{k: np.asarray(v) for k,v in audit._asdict().items()})
        print({k:v for k,v in row.items() if k != 'points'}, flush=True)
        jax.clear_caches()
