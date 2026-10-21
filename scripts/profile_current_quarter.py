"""Measure the current planner without tuning its numerical configuration.

The complete compiled optimizer is timed directly. Separately compiled stages
and nested component microbenchmarks are diagnostic timings, not an additive
partition of the fused optimizer. Their outputs are checked against that
optimizer before the measurements are written.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np

from run_full_control_quarter import source_hashes, write
from voi_diffusion_ipp.quarter.continuous import (
    ControlSettings, aggregate_costs, continuation, initial_sequences, optimizer, rollout,
)
from voi_diffusion_ipp.quarter.diffusion import lower_tail
from voi_diffusion_ipp.quarter.environment import environment, initial_belief, table_basis, table_sampler, wind
from voi_diffusion_ipp.quarter.execution import executor
from voi_diffusion_ipp.quarter.mbd_core import search
from voi_diffusion_ipp.quarter.rrt import demonstrations
from voi_diffusion_ipp.safety.flow_policy_value import values, filter_control
from voi_diffusion_ipp.wind.cache import TableMean
from voi_diffusion_ipp.wind.gp import InnerFields


def measure(fn, repeats=5):
    samples = []; result = None
    for _ in range(repeats+1):
        start = time.perf_counter(); result = fn(); jax.block_until_ready(result)
        samples.append(time.perf_counter()-start)
    warm = np.asarray(samples[1:])
    return result, dict(first_seconds=samples[0], warm_seconds=warm.tolist(),
                        median_seconds=float(np.median(warm)), p95_seconds=float(np.quantile(warm,.95)),
                        max_seconds=float(warm.max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--record', type=Path, default=Path('output/european_quarter/planner_investigation/supported_forward_reverse300/two_forward_proposed_8101.json'))
    p.add_argument('--plan-index', type=int, default=20)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--fixture', type=Path, help='Portable profile-state metadata JSON')
    p.add_argument('--export-fixture', type=Path, help='Write a portable profile-state metadata JSON and NPZ')
    p.add_argument('--export-only', action='store_true')
    a = p.parse_args(); a.output.parent.mkdir(parents=True, exist_ok=True)
    fixture=json.loads(a.fixture.read_text()) if a.fixture else None
    record = fixture['record_metadata'] if fixture else json.loads(a.record.read_text())
    cfg = ControlSettings(**record['config'])
    assert cfg.guidance_mode == 'off' and cfg.validate_initials
    env = environment(record['layout'], cfg.environment_scale, cfg.inspection_scale, record['case'])
    entry = dict(t=fixture['mission_time']) if fixture else record['plans'][a.plan_index]
    b=initial_belief(cfg)
    condition=jax.jit(lambda bb,p,y:bb.condition(p,y,relocate=cfg.wind_relocate))
    for obs in record['observations']:
        if obs['t']>entry['t']+1e-5:break
        b=condition(b,jnp.array([obs['position']],jnp.float32),jnp.array([obs['measurement']],jnp.float32))
    sample=table_sampler(cfg.construction_worlds+cfg.worlds+cfg.validation_worlds,cfg.environment_scale)
    if fixture:
        a.plan_index=fixture['plan_index']
        with np.load(a.fixture.parent/fixture['arrays']) as data:
            x,q,mean,construction,worlds,validation,key,previous,packed=[jnp.asarray(data[k]) for k in
                ('x','q','mean','construction','worlds','validation','key','previous','packed')]
    else:
        key=jax.random.PRNGKey(record['seed'])
        construction_key=jax.random.split(key,3)[1]
        for _ in range(a.plan_index+1):key,ks,kp=jax.random.split(key,3)
        mean,fields=sample(b,ks);_,stable=sample(b,construction_key)
        construction=stable[:cfg.construction_worlds]
        worlds=fields[cfg.construction_worlds:cfg.construction_worlds+cfg.worlds]
        validation=fields[-cfg.validation_worlds:];key=kp
        with np.load(a.record.parent/entry['execution_audit']) as audit:
            packed=jnp.asarray(audit['initial_state']);x=jnp.r_[packed[:2],packed[3:5]]
            np.testing.assert_allclose(mean,audit['mean'],atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(construction,audit['construction'],atol=2e-5,rtol=2e-5)
        if a.plan_index:
            with np.load(a.record.parent/record['plans'][a.plan_index-1]['execution_audit']) as audit:
                q=jnp.asarray(audit['clarity'][-1])
            previous=continuation(jnp.asarray(record['plans'][a.plan_index-1]['policy']),cfg)
        else:
            q=jnp.asarray(record['frames'][0]['q']);previous=jnp.zeros(2*cfg.steps)
    jax.block_until_ready((b,x,q,mean,construction,worlds,validation,key,previous))
    if a.export_fixture:
        destination=a.export_fixture;destination.parent.mkdir(parents=True,exist_ok=True)
        arrays=destination.with_suffix('.npz')
        np.savez_compressed(arrays,x=np.asarray(x),q=np.asarray(q),mean=np.asarray(mean),
            construction=np.asarray(construction),worlds=np.asarray(worlds),validation=np.asarray(validation),
            key=np.asarray(key),previous=np.asarray(previous),packed=np.asarray(packed))
        write(destination,dict(record_metadata={k:record[k] for k in ('config','layout','case','seed')}|
            dict(observations=[o for o in record['observations'] if o['t']<=entry['t']+1e-5]),
            mission_time=entry['t'],plan_index=a.plan_index,arrays=arrays.name,source_record=str(a.record),
            source_hashes=source_hashes()))
        print('EXPORTED',destination,arrays,flush=True)
    if a.export_only:return
    profile = dict(record=str(a.record), plan_index=a.plan_index, mission_time=entry['t'],
                   config=asdict(cfg), source_hashes=source_hashes(), jax=jax.__version__,
                   platform=platform.platform(), devices=list(map(str,jax.devices())),
                   timing_convention='Synchronized wall time; first invocation includes compilation; warm measurements follow; separately compiled stages are not additive',
                   timings={})
    def save(name, timing):
        profile['timings'][name]=timing; write(a.output,profile)
        print(name,json.dumps(timing),flush=True)
    demo, rrt = measure(lambda: demonstrations(np.asarray(x),env,cfg,record['seed']+100003*a.plan_index),a.repeats)
    # RRT returns ordinary host data, not accelerator arrays.
    demo = jnp.asarray(demo[0]); demo.block_until_ready(); save('rrt',rrt)
    plan = optimizer(env,cfg,'proposed')
    args=(x,q,mean,construction,worlds,validation,previous,key,jnp.array(cfg.optimism),jnp.array(False),demo)
    full, timing = measure(lambda:plan(*args),a.repeats); save('optimizer_fused',timing)

    def field_cost(theta, bank, x, q, mean, construction, filtered=True):
        def one(field):
            r=rollout(theta,x,q,mean,construction,field,env,cfg,filtered)
            return jnp.array([r.monitor,r.penalty])
        cost=jax.vmap(one)(bank)
        return aggregate_costs(cost[:,0],cost[:,1],cfg.penalty_aggregation)
    initials=jnp.concatenate((initial_sequences(x,cfg),demo))
    train=jax.jit(lambda x,q,mean,construction,worlds,previous,key,initials:
        search(lambda th:field_cost(th,worlds,x,q,mean,construction),initials,previous,key,cfg,
               jnp.array(cfg.optimism),jnp.array(cfg.mixture),jnp.array(False),None))
    trained,timing=measure(lambda:train(x,q,mean,construction,worlds,previous,key,initials),a.repeats); save('training_separate',timing)
    candidates=jnp.concatenate((trained.finalists,trained.initial_parameters))
    @jax.jit
    def select(candidates,x,q,mean,construction,validation):
        cost=jax.lax.map(lambda th:field_cost(th,validation,x,q,mean,construction),candidates,batch_size=cfg.batch_size)
        score=lower_tail(cost,cfg.optimism)
        valid=jnp.all(jnp.isfinite(cost),axis=1)&jnp.all(jnp.abs(candidates)<=cfg.latent_bound,axis=1)
        score=jnp.where(valid,score,jnp.inf); i=jnp.argmin(score)
        return candidates[i],score[i],i
    chosen,timing=measure(lambda:select(candidates,x,q,mean,construction,validation),a.repeats);save('selection_separate',timing)
    np.testing.assert_allclose(chosen[0],full[0],atol=3e-5,rtol=3e-5)
    np.testing.assert_allclose(chosen[1],full[1],atol=3e-5,rtol=3e-5)
    profile['split_matches_fused']=True
    batch=jax.jit(lambda parameters,bank,x,q,mean,construction:
        jax.lax.map(lambda th:field_cost(th,bank,x,q,mean,construction),parameters,batch_size=cfg.batch_size))
    _,timing=measure(lambda:batch(jnp.r_[initials,previous[None]],worlds,x,q,mean,construction),a.repeats);save('initial_evaluations_separate',timing)
    _,timing=measure(lambda:batch(full[3].samples[0],worlds,x,q,mean,construction),a.repeats);save('one_diffusion_batch_separate',timing)
    for enabled in (True,False):
        fn=jax.jit(lambda theta,bank,x,q,mean,construction:field_cost(theta,bank,x,q,mean,construction,enabled))
        _,timing=measure(lambda:fn(full[0],worlds,x,q,mean,construction),20);save('one_trajectory_four_fields_'+('filtered' if enabled else 'unfiltered'),timing)

    basis=table_basis(cfg.environment_scale);m=TableMean(basis,mean);inner=InnerFields(basis,construction)
    from dataclasses import replace
    fc=replace(cfg,safety_margin=cfg.execution_margin)
    def value_gradient(z,m,inner):
        def f(xx):
            v=values(xx,m,inner,env,fc)
            return v,v
        return jax.jacfwd(f,has_aux=True)(z)
    for name,fn in (
        ('constraint_values',lambda z,m,inner:values(z,m,inner,env,fc)),
        ('constraint_values_and_derivatives',value_gradient),
        ('one_filter_call',lambda z,m,inner:filter_control(z,jnp.array([.2,-.2]),m,inner,env,fc)),
    ):
        kernel=jax.jit(fn)
        _,timing=measure(lambda:kernel(x,m,inner),30);save(name,timing)

    b=initial_belief(cfg)
    condition=jax.jit(lambda bb,p,y:bb.condition(p,y,relocate=cfg.wind_relocate))
    for obs in record['observations']:
        if obs['t']>entry['t']+1e-5:break
        b=condition(b,jnp.array([obs['position']],jnp.float32),jnp.array([obs['measurement']],jnp.float32))
    jax.block_until_ready(b)
    sample=table_sampler(cfg.construction_worlds+cfg.worlds+cfg.validation_worlds,cfg.environment_scale)
    _,timing=measure(lambda:sample(b,jax.random.PRNGKey(47)),10);save('one_posterior_field_bank',timing)
    observation=min(record['observations'],key=lambda z:abs(z['t']-entry['t']))
    _,timing=measure(lambda:condition(b,jnp.array([observation['position']],jnp.float32),jnp.array([observation['measurement']],jnp.float32)),10)
    save('one_poam_update',timing)
    execute=executor(env,replace(cfg,commit=1.),lambda p:wind(p,record['case'],cfg.wind_strength,cfg.environment_scale))
    _,timing=measure(lambda:execute(packed,full[0],mean,construction),5);save('one_second_nonlinear_execution',timing)
    def entire_call():
        mu,fields=sample(b,jax.random.PRNGKey(47));_,stable=sample(b,jax.random.PRNGKey(73))
        jax.block_until_ready((mu,fields,stable))
        ds,_=demonstrations(np.asarray(x),env,cfg,record['seed']+100003*a.plan_index)
        ds=jnp.asarray(ds)
        return plan(x,q,mu,stable[:cfg.construction_worlds],fields[cfg.construction_worlds:cfg.construction_worlds+cfg.worlds],fields[-cfg.validation_worlds:],previous,key,jnp.array(cfg.optimism),jnp.array(False),ds)
    _,timing=measure(entire_call,a.repeats);save('complete_planning_call',timing)
    profile['counts']=dict(decision_dimension=2*cfg.steps,steps=cfg.steps,
        initial_trajectories=len(initials)+1,training_trajectories=int(trained.evaluations),
        validation_trajectories=len(candidates),performance_rollouts=int(trained.evaluations)*cfg.worlds,
        validation_rollouts=len(candidates)*cfg.validation_worlds,
        forecast_filter_calls=(int(trained.evaluations)*cfg.worlds+len(candidates)*cfg.validation_worlds)*cfg.steps,
        construction_trajectory_steps_per_filter=cfg.construction_worlds*cfg.backup_steps)
    write(a.output,profile)


if __name__=='__main__':main()
