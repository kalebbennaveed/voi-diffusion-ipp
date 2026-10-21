"""Time the complete planning path on recorded states and learned posteriors.

Compilation is reported separately. The measured call includes posterior field
draws, RRT initializations, all MBD levels, and independent final selection.
No prerecorded trajectory is returned by the benchmark.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from run_full_control_quarter import source_hashes, write
from voi_diffusion_ipp.quarter.continuous import ControlSettings, optimizer
from voi_diffusion_ipp.quarter.environment import environment, initial_belief, table_sampler
from voi_diffusion_ipp.quarter.rrt import demonstrations


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--record', type=Path, default=Path('output/european_quarter/planner_investigation/supported_forward_reverse300/two_forward_proposed_8101.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dt', type=float, default=.2)
    p.add_argument('--iterations', type=int, default=6)
    p.add_argument('--proposals', type=int, default=16)
    p.add_argument('--visits', type=int, default=1)
    p.add_argument('--times', type=float, nargs='+', default=[0., 10., 100., 200.])
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(a.record.read_text())
    cfg = replace(ControlSettings(**d['config']), horizon=10., commit=5., dt=a.dt,
                  iterations=a.iterations, proposals=a.proposals, rrt_visits=a.visits)
    env = environment(d['layout'], cfg.environment_scale, cfg.inspection_scale, d['case'])
    plan = optimizer(env, cfg, 'proposed')
    sample = table_sampler(cfg.construction_worlds+cfg.worlds+cfg.validation_worlds, cfg.environment_scale)
    condition = jax.jit(lambda b, x, y: b.condition(x, y, relocate=cfg.wind_relocate))
    beliefs = []; b = initial_belief(cfg); cursor = 0
    for t in a.times:
        while cursor < len(d['observations']) and d['observations'][cursor]['t'] <= t+1e-5:
            obs = d['observations'][cursor]
            b = condition(b, jnp.array([obs['position']], jnp.float32), jnp.array([obs['measurement']], jnp.float32))
            cursor += 1
        jax.block_until_ready(b)
        beliefs.append(b)
    results = []
    # Compile and warm once, then measure identical code on different states,
    # posterior distributions and RRT random seeds (no cache of output paths).
    for repeat, t in enumerate([a.times[0], *a.times]):
        i = max(0, repeat-1); b = beliefs[i]
        frame = min(d['frames'], key=lambda f: abs(f['t']-t))
        x = jnp.array([*frame['p'][:2], *frame['velocity'][:2]], jnp.float32)
        q = jnp.array(frame['q'], jnp.float32)
        previous = jnp.zeros(2*cfg.steps)
        key, ks, kp = jax.random.split(jax.random.PRNGKey(171+repeat), 3)
        jax.block_until_ready((b, x, q, previous, ks, kp))
        start = time.perf_counter()
        mean, fields = sample(b, ks)
        _, stable = sample(b, jax.random.PRNGKey(73))
        construction = stable[:cfg.construction_worlds]
        worlds = fields[cfg.construction_worlds:cfg.construction_worlds+cfg.worlds]
        validation = fields[-cfg.validation_worlds:]
        jax.block_until_ready((mean, construction, worlds, validation))
        sampled = time.perf_counter()
        demo, info = demonstrations(x, env, cfg, 901+repeat)
        demo = jnp.asarray(demo); demo.block_until_ready()
        initialized = time.perf_counter()
        found = plan(x, q, mean, construction, worlds, validation, previous, kp,
                     jnp.array(cfg.optimism), jnp.array(False), demo)
        jax.block_until_ready(found)
        end = time.perf_counter()
        row = dict(cold=repeat == 0, mission_time=t, field_seconds=sampled-start,
                   rrt_seconds=initialized-sampled, mbd_seconds=end-initialized,
                   total_seconds=end-start, score=float(found[1]),
                   decision_dimension=int(found[0].size), rrt_initials=info['found'])
        results.append(row); print(json.dumps(row), flush=True)
        write(a.output, dict(config=asdict(cfg), source_hashes=source_hashes(),
                            record=str(a.record), devices=list(map(str, jax.devices())), calls=results))


if __name__ == '__main__':
    main()
