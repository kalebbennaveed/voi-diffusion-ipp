"""Fixed-state runtime/quality comparison; does not run or publish a mission.

All budgets use identical reached states, posterior field samples, RRT random
seeds, 45 s horizon, and a separate validation bank. Compilation is reported
separately. Selected trajectories also receive a five-second nonlinear tracking
check from the recorded physical state, using 0.01 s physics.
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.quarter.continuous import ControlSettings, optimizer
from voi_diffusion_ipp.quarter.environment import environment, initial_belief, table_sampler, wind
from voi_diffusion_ipp.quarter.execution import executor
from voi_diffusion_ipp.quarter.rrt import demonstrations
from voi_diffusion_ipp.quarter import sensing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--budgets', nargs='+', default=['compact', 'smaller', 'reference'],
                        choices=['compact', 'smaller', 'lean', 'reference'])
    parser.add_argument('--indices', nargs='+', type=int, default=[0, 6, 18])
    args = parser.parse_args()
    record = json.loads(args.source.read_text())
    base = ControlSettings(**record['config'])
    env = environment(record['layout'], base.environment_scale, base.inspection_scale)
    sampler = table_sampler(base.construction_worlds + base.worlds + base.validation_worlds,
                            base.environment_scale)
    belief = initial_belief(base)
    condition = jax.jit(lambda b, p, y: b.condition(p, y))
    key = jax.random.PRNGKey(record['seed'])
    construction_key = jax.random.split(key, 3)[1]
    snapshots = []
    for index in range(max(args.indices) + 1):
        key, sample_key, plan_key = jax.random.split(key, 3)
        if index in args.indices:
            mean, fields = sampler(belief, sample_key)
            construction = fields[:base.construction_worlds]
            if base.common_construction:
                construction = sampler(belief, construction_key)[1][:base.construction_worlds]
            plan = record['plans'][index]
            with np.load(args.source.parent / plan['execution_audit']) as audit:
                packed = audit['initial_state'].astype(np.float32)
                np.testing.assert_allclose(mean, audit['mean'], atol=2e-6)
                np.testing.assert_allclose(construction, audit['construction'], atol=2e-6)
            if index:
                with np.load(args.source.parent / record['plans'][index - 1]['execution_audit']) as previous:
                    q = previous['clarity'][-1]
            else:
                q = np.full(len(env.cells), .35, dtype=np.float32)
                q[:42] = .65
            snapshots.append(dict(index=index, packed=jnp.asarray(packed),
                                  x=jnp.asarray(np.r_[packed[:2], packed[3:5]]), q=jnp.asarray(q),
                                  mean=mean, construction=construction,
                                  worlds=fields[base.construction_worlds:base.construction_worlds + base.worlds],
                                  validation=fields[-base.validation_worlds:], key=plan_key))
        if index < max(args.indices):
            obs = record['observations'][index]
            belief = condition(belief, jnp.asarray([obs['position']], jnp.float32),
                               jnp.asarray([obs['measurement']], jnp.float32))
    configurations = dict(reference=base,
                          compact=replace(base, dt=.2, iterations=8, proposals=32, batch_size=32),
                          smaller=replace(base, dt=.25, iterations=6, proposals=24, batch_size=24),
                          lean=replace(base, dt=.2, iterations=6, proposals=16, batch_size=16))
    report = dict(source=str(args.source.resolve()), mission=False,
                  description='Fixed-state benchmark; no mission performance claim. Cold-start RRT for each state, no inherited optimized trajectory. Same posterior samples and validation fields for every budget.',
                  devices=list(map(str, jax.devices())), xla_flags=os.environ.get('XLA_FLAGS', ''), budgets=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False))

    for label in args.budgets:
        cfg = configurations[label]
        plan = optimizer(env, cfg, 'proposed')
        execute = executor(env, cfg, lambda p: wind(p, record['case'], cfg.wind_strength,
                                                   cfg.environment_scale), filter_on=True)

        @jax.jit
        def tracking_quality(packed, theta, mean, construction, q):
            _, logs = execute(packed, theta, mean, construction)
            states, refs, _, _, _, slacks = logs

            def step(carry, state):
                q, position, cost, minimum = carry
                new_q = sensing.step(q, position, state[:2], env, cfg, .01, state[2])
                deficit = lambda v: jnp.maximum(env.targets - v, 0).mean()
                cost += .005 * (deficit(q) + deficit(new_q))
                minimum = jnp.minimum(minimum, env.clearance(jnp.r_[state[:2], state[3:5]]) - .275)
                return (new_q, state[:2], cost, minimum), None

            end, _ = jax.lax.scan(step, (q, packed[:2], jnp.array(0.), jnp.array(1e3)), states)
            return jnp.array([end[2], end[3], jnp.sqrt(jnp.mean(jnp.sum((states[:, :3] - refs) ** 2, axis=1))),
                              jnp.max(slacks), jnp.min(states[:, 2]), jnp.max(states[:, 2])])

        result = dict(name=label, config=asdict(cfg), records=[])
        report['budgets'].append(result)
        for state_number, state in enumerate(snapshots):
            start = time.perf_counter()
            seeds, _ = demonstrations(state['x'], env, cfg, record['seed'] + 100003 * state['index'])
            rrt_seconds = time.perf_counter() - start
            inputs = [state[k] for k in ('x', 'q', 'mean', 'construction', 'worlds', 'validation')]
            inputs += [jnp.zeros(2 * cfg.steps), state['key'], jnp.array(cfg.optimism),
                       jnp.array(True), jnp.asarray(seeds)]
            timings = []
            for repeat in range(3 if state_number == 0 else 1):
                start = time.perf_counter()
                found = plan(*inputs)
                jax.block_until_ready(found)
                timings.append(time.perf_counter() - start)
                print(label, 'state', state['index'], 'repeat', repeat, 'seconds', round(timings[-1], 3),
                      'validation', round(float(found[1]), 5), flush=True)
            metrics = np.asarray(tracking_quality(state['packed'], found[0], state['mean'],
                                                 state['construction'], state['q']))
            row = dict(planning_time_in_mission=state['index'] * cfg.commit,
                       compilation_and_first_seconds=timings[0] if state_number == 0 else None,
                       warm_seconds=timings[1:] if state_number == 0 else timings,
                       rrt_seconds=rrt_seconds, evaluations=int(found[3].evaluations),
                       dimension=2 * cfg.steps, validation_cost=float(found[1]),
                       validation_mean_cost=float(found[2]), initial_cost=float(found[3].initial_cost),
                       best_training_cost=float(found[3].trace[-1, 0]),
                       tracking_check_seconds=cfg.commit, physics_dt=.01,
                       tracking_deficit_integral=float(metrics[0]), minimum_clearance=float(metrics[1]),
                       tracking_rms=float(metrics[2]), maximum_slack=float(metrics[3]),
                       min_altitude=float(metrics[4]), max_altitude=float(metrics[5]),
                       finite=bool(np.isfinite(metrics).all()))
            result['records'].append(row)
            save()
            print('TRACKING CHECK', label, row, flush=True)
        result['median_warm_seconds'] = float(np.median([v for row in result['records'] for v in row['warm_seconds']]))
        save()
        jax.clear_caches()
    reference = next((b for b in report['budgets'] if b['name'] == 'reference'), None)
    if reference:
        for budget in report['budgets']:
            budget['speedup_vs_reference'] = reference['median_warm_seconds'] / budget['median_warm_seconds']
            for row, original in zip(budget['records'], reference['records']):
                row['validation_change_percent'] = 100 * (row['validation_cost'] / original['validation_cost'] - 1)
    save()
    print('SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
