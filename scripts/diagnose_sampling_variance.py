"""Same-state paired MBD optimization and independent-bank sampling diagnosis.

Antithetic outer field/noise draws retain correct Gaussian marginals. Inner
RPCBF draws stay independent and unchanged. This does not alter any mission.
"""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.checkpoint import source_hashes
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.planning.budgeted import evaluate, make_planner, warm_starts
from voi_diffusion_ipp.viz.demo_data import serializable
from voi_diffusion_ipp.wind.cache import build_prior_cache
from voi_diffusion_ipp.wind.poam import POAM

ROOT = Path(__file__).resolve().parents[1]


def scorer(config, env, x, q, belief):
    @jax.jit
    def score(vectors, cache, noise):
        def one(v):
            a, af = evaluate(v, x, q, belief, cache, noise, env, config, True)
            f, ff = evaluate(v, x, q, belief, cache, noise, env, config, False)
            return jnp.array([a, f, jnp.max(jnp.abs(af[0].x - ff[0].x))])
        return jax.lax.map(one, vectors)
    return score


def main():
    protocol_path = ROOT / 'docs/wind_route_study/sampling_protocol_01.json'
    protocol = json.loads(protocol_path.read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == protocol['driver_sha256']
    assert source_hashes() == protocol['source_sha256']
    out = ROOT / 'docs/wind_route_study/sampling_diagnosis_01'
    out.mkdir(exist_ok=False)
    result = {'protocol': protocol, 'rows': []}
    for job in protocol['jobs']:
        path = ROOT / job['input']
        raw = json.loads(path.with_suffix('.json').read_text())
        meta = raw['metadata']
        with np.load(path, allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
        cfg = Config(**meta['config'])
        assert cfg.worlds == 2 and cfg.budgeted_policy == 'compact'
        env, template, _, _, _ = make_two_region(meta['task_spec'], cfg)
        model = POAM.from_data(data['initial_observation_positions'], data['initial_observations'],
                               jax.random.PRNGKey(600000), len(template.z),
                               noise_std=cfg.observation_std, prior_std=cfg.wind_std,
                               spectral_count=cfg.poam_spectral_nodes,
                               prior_support=template.prior_support)
        root = jax.random.PRNGKey(meta['seed'])
        for i in range(job['period']):
            sl = slice(i*cfg.stage_steps, (i+1)*cfg.stage_steps)
            model.update(data['observation_positions'][sl], data['observations'][sl],
                         jax.random.fold_in(root, 500000+i), train_steps=cfg.poam_train_steps)
        b = model.planning_belief()
        mu, cov = jax.jit(lambda v, cells: v.predict(cells))(b, env.cells)
        np.testing.assert_allclose(mu, data['mean'][job['period']], atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(jnp.sqrt(cov[:, 0, 0]), data['std'][job['period']],
                                   atol=3e-5, rtol=3e-5)
        stride = round(cfg.stage_steps*cfg.dt/meta['execution_dt'])
        x, q = [jnp.asarray(data[k][job['period']*stride]) for k in ['x', 'q']]
        starts = warm_starts(x, env, cfg, True)
        targets = np.flatnonzero(np.asarray(env.targets)>0)
        plans = [make_planner(cfg, a, targets) for a in [True, False]]
        build = jax.jit(lambda belief, e: build_prior_cache(belief, e, 129, True))
        vectors, labels = list(starts), [f'initial_{i}' for i in range(len(starts))]
        for trial in range(protocol['optimization_banks']):
            keys = jax.random.split(jax.random.PRNGKey(981000+trial), 3)
            eps = jax.random.normal(keys[0], (cfg.worlds+cfg.inner_samples, *b.sample_shape))
            ns = jax.random.normal(keys[1], (cfg.worlds, cfg.stage_steps+cfg.future_steps, 2))
            cache = build(b, eps)
            for anticipate, plan in zip([True, False], plans, strict=True):
                best = jax.block_until_ready(plan(x, q, b, cache, ns, env, starts, keys[2]))
                vectors.append(best.vector)
                labels.append(f'{"active" if anticipate else "frozen"}_training_{trial}')
            print(job['name'], 'optimized', trial, flush=True)
        vectors = jnp.stack(vectors)

        score = scorer(cfg, env, x, q, b)
        iid, anti = [], []
        for bank in range(protocol['evaluation_banks']):
            keys = jax.random.split(jax.random.PRNGKey(982000+bank), 2)
            eps = jax.random.normal(keys[0], (2+cfg.inner_samples, *b.sample_shape))
            ns = jax.random.normal(keys[1], (2, cfg.stage_steps+cfg.future_steps, 2))
            iid.append(np.asarray(score(vectors, build(b, eps), ns)))
            paired = eps.at[1].set(-eps[0])
            paired_noise = ns.at[1].set(-ns[0])
            np.testing.assert_array_equal(paired[2:], eps[2:])
            anti.append(np.asarray(score(vectors, build(b, paired), paired_noise)))
            print(job['name'], 'evaluation bank', bank, flush=True)
        iid, anti = np.stack(iid), np.stack(anti)
        assert max(iid[:, :, 2].max(), anti[:, :, 2].max()) < 2e-4
        finite = np.isfinite(iid[:, :, :2]).all((0, 2)) & np.isfinite(anti[:, :, :2]).all((0, 2))
        # Fixed-candidate estimator variability, not error against hidden truth.
        vi = iid[:, finite, 0].var(0, ddof=1)
        va = anti[:, finite, 0].var(0, ddof=1)
        differences = []
        offset = len(starts)
        for trial in range(protocol['optimization_banks']):
            ai, fi = offset+2*trial, offset+2*trial+1
            # IID evaluation banks are independent of optimizer training draws.
            d = iid[:, ai, 0]-iid[:, fi, 0]
            differences.append({'training_bank': trial, 'finite': bool(np.isfinite(d).all()),
                                'mean_active_cost_difference': float(d.mean()),
                                'evaluation_standard_error': float(d.std(ddof=1)/np.sqrt(len(d)))})
        np.savez_compressed(out/(job['name']+'.npz'), vectors=np.asarray(vectors), iid=iid,
                            antithetic=anti, all_banks_finite=finite)
        row = {'job': job, 'labels': labels, 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'finite_candidates': int(finite.sum()), 'total_candidates': len(vectors),
               'median_antithetic_over_iid_variance': float(np.median(va/np.maximum(vi, 1e-12))),
               'mean_antithetic_over_iid_variance': float(va.mean()/max(vi.mean(), 1e-12)),
               'paired_optimized_choices': differences}
        result['rows'].append(row)
        (out/'summary.json').write_text(json.dumps(serializable(result), indent=2)+'\n')


if __name__ == '__main__':
    main()
