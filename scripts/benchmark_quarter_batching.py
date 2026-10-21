"""Time identical recorded rollout work with different JAX batch widths."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.quarter.continuous import ControlSettings, rollout
from voi_diffusion_ipp.quarter.environment import environment


parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--batches', type=int, nargs='+', default=[64, 32, 16, 8])
args = parser.parse_args()
run = json.loads(args.source.read_text())
cfg = ControlSettings(**run['config'])
env = environment(run['layout'], cfg.environment_scale, cfg.inspection_scale)
with np.load(args.source.parent / run['plans'][0]['audit']) as data:
    inputs = [jnp.asarray(data['samples'][0])]
    inputs += [jnp.asarray(data[k]) for k in ('initial_state', 'clarity', 'mean', 'construction', 'performance_fields')]

report = dict(config=asdict(cfg), source=str(args.source.resolve()), comparisons=[])
original = None
for width in args.batches:
    @jax.jit
    def evaluate(theta, x, q, mean, construction, worlds):
        def one(th):
            def field(w):
                r = rollout(th, x, q, mean, construction, w, env, cfg, True)
                return r.monitor + r.penalty
            return jax.vmap(field)(worlds)
        return jax.lax.map(one, theta, batch_size=width)

    seconds = []
    for repeat in range(3):
        start = time.perf_counter()
        result = evaluate(*inputs)
        result.block_until_ready()
        seconds.append(time.perf_counter() - start)
        print('width', width, 'repeat', repeat, 'seconds', seconds[-1], flush=True)
    cost = np.asarray(result)
    if original is None:
        original = cost
    np.testing.assert_allclose(cost, original, rtol=3e-5, atol=3e-5)
    report['comparisons'].append(dict(width=width, first_seconds=seconds[0], warm_seconds=seconds[1:],
                                      max_cost_difference=float(np.max(np.abs(cost-original)))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    jax.clear_caches()
