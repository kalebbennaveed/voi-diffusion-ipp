"""Bounded, offline inference replay. No robot simulation or planner changes."""
import argparse
import gzip
import json
from pathlib import Path
import signal
import time

import jax
import jax.numpy as jnp
import numpy as np
import rmse_monitoring as rm
import cluster_environment as ce
from waypoint_poam import condition, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, default=rm.m.physical.ROOT/'results/airflow/rmse_learning_20261104/checked300')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh audit output')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('120s audit budget')))
    signal.alarm(120)
    start = time.perf_counter()
    env, grid, _ = ce.environment(rm.legacy.physical_settings(rm.settings()),
        rm.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
    basis, points, shape = rm.table_grid(env.bounds)
    initial = rm.initial_belief(env, points)
    pred = jax.jit(lambda b, p: b.predict(p))
    sample_moments = jax.jit(lambda b, p: b.sample_moments(p))
    query = np.asarray(env.cells)[::24] + np.array([.019, -.023])
    reports = []
    for scenario in (1, 2):
        stem = f's{scenario}-center-baseline1'
        record = json.loads(gzip.decompress((args.records/f'{stem}.json.gz').read_bytes()))
        protocol = json.loads((args.records/f'{stem}-protocol.json').read_text())
        observations = np.asarray(record['observations'])
        b = initial
        mask = jnp.zeros(len(b.mean), bool)
        mom = jax.tree.map(jnp.zeros_like, b.basis.parameters)
        var, step = mom, jnp.array(0)
        rng = np.random.default_rng(protocol['initial_condition']['seed'] + 441)
        for i, row in enumerate(observations):
            b, mask = condition(b, jnp.asarray(row[None, 1:3]), jnp.asarray(row[None, 3:5]), mask)
            if row[0] > 0 and abs(row[0]/2-round(row[0]/2)) < 1e-8:
                idx = rng.integers(0, i+1, size=32)
                b, mom, var, step, _ = train(b, jnp.asarray(observations[idx, 1:3]),
                    jnp.asarray(observations[idx, 3:5]), jnp.array(i+1, dtype=jnp.float64), mom, var, step)
        mu, cov = pred(b, jnp.asarray(grid['all_cells']))
        last = record['belief_frames'][-1]
        replay_error = float(np.max(np.abs(mu - np.asarray(last['mean']))))
        std_error = float(np.max(np.abs(jnp.sqrt(cov[:, 0, 0]) - np.asarray(last['std']))))
        # Saved maps are rounded to 5 decimals. No dependence on elapsed clock.
        assert replay_error < 6e-6 and std_error < 6e-6
        exact_mean, exact_cov = b.joint(jnp.asarray(query))
        sampled_mean, sampled_cov = sample_moments(b, jnp.asarray(query))
        grid_mean, grid_cov = pred(b, points)
        interpolated_mean = basis.evaluate(jnp.asarray(query), grid_mean.reshape(shape))
        interpolated_std = basis.evaluate(jnp.asarray(query), jnp.sqrt(grid_cov[:, 0, 0]).reshape(*shape[:2], 1))[:, 0]
        error_cov = np.asarray(sampled_cov-exact_cov)
        result = dict(scenario=scenario, actual_observations=len(observations),
            replay_mean_max_difference_m_s=replay_error, replay_std_max_difference_m_s=std_error,
            prior_amplitude_m2_s2=float(jnp.exp(b.basis.parameters['log_amplitude'])),
            inducing_count=len(b.mean), prior_support_count=len(b.basis.support), spectral_nodes=len(b.basis.frequencies),
            off_grid_queries=len(query), sample_mean_max_error_m_s=float(jnp.max(jnp.abs(sampled_mean-exact_mean))),
            sample_covariance_relative_frobenius_error=float(np.linalg.norm(error_cov)/np.linalg.norm(exact_cov)),
            sample_variance_max_absolute_error_m2_s2=float(np.max(np.abs(np.diag(error_cov)))),
            cache_mean_vector_max_error_m_s=float(jnp.max(jnp.linalg.norm(interpolated_mean-exact_mean, axis=1))),
            cache_std_max_absolute_error_m_s=float(jnp.max(jnp.abs(interpolated_std-jnp.sqrt(jnp.diag(exact_cov))))),
            elapsed_audit_seconds=time.perf_counter()-start)
        reports.append(result)
        print(json.dumps(result), flush=True)
    with args.output.open('x') as file:
        json.dump(dict(scope='Offline replay and numerical approximation audit; not a safety certificate.', reports=reports), file, indent=2)
    signal.alarm(0)


if __name__ == '__main__':
    main()
