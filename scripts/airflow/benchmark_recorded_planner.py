"""Isolated 15-second MBD timings from the actual final recorded posterior.

This inference-only checkpoint cannot resume sensor assimilation. POAM update
and training timings are reported separately by benchmark_full_learning.py.
Never substitute these warm timings for the original mission's wall times.
"""
import argparse,gzip,json,signal,time,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import full_learning_monitoring as model
import efficient_learning_monitoring as efficient
import cluster_environment as ce
from cached_poam_fields import CachedFields
from voi_diffusion_ipp.wind.poam import attentive_kernel,attention_weights
from voi_diffusion_ipp.wind.poam_belief import PosteriorBasis,POAMBelief,spectral_nodes


def restore_inference(path,support):
    z=np.load(path);params={k[6:]:jnp.asarray(z[k]) for k in z.files if k.startswith('param_')}
    centers=jnp.asarray(z['centers']);lengths=jnp.asarray(z['lengths'])
    K=attentive_kernel(params,support,support,lengths)
    L=jnp.linalg.cholesky((K+K.T)/2+1e-4*jnp.eye(len(support)))
    basis=PosteriorBasis(centers,params,lengths,spectral_nodes(32).astype(jnp.float64),support,
        jnp.zeros((0,0),dtype=jnp.float64),L,attention_weights(params,support),attention_weights(params,centers))
    n=len(centers);missing=jnp.full((n,n),jnp.nan)
    # Explicit invalid sufficient statistics prevent accidental online reuse.
    return POAMBelief(basis,jnp.asarray(z['mean']),jnp.asarray(z['covariance']),jnp.asarray(z['inverse']),
        jnp.full((n,2),jnp.nan),missing,missing,jnp.array(.1),missing)


def main():
    p=argparse.ArgumentParser();p.add_argument('recordings',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compare-model-trust',action='store_true')
    p.add_argument('--information-trigger',choices=['stalled','lower-cost'],default='stalled')
    p.add_argument('--at-final-state',action='store_true');p.add_argument('--timed-learning',action='store_true')
    p.add_argument('--validation-reference',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('240-second isolated timing limit')));signal.alarm(240)
    cfg=model.settings();root=model.m.physical.ROOT
    env,grid,_=ce.environment(model.legacy.physical_settings(cfg),root/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
    basis,points,shape=model.table_grid(env.bounds,.1);_,extended,_=model.table_grid(env.bounds,.4)
    outside=np.any((np.asarray(extended)<np.asarray(env.bounds[0]))|(np.asarray(extended)>np.asarray(env.bounds[1])),axis=1)
    support=jnp.asarray(np.vstack((np.asarray(env.cells),np.asarray(extended)[outside])))
    with jax.experimental.disable_x64():
        original=model.optimizer(model.legacy.float32(env),cfg,model.legacy.float32(basis),True)
        planner=efficient.optimizer(model.legacy.float32(env),cfg,model.legacy.float32(basis),True)
        trusted=efficient.optimizer(model.legacy.float32(env),cfg,model.legacy.float32(basis),True,wind_error_reserve=.25*cfg.umax,
            information_trigger=a.information_trigger,prepare_validation=a.information_trigger=='lower-cost') if a.compare_model_trust else None
        trust_reference=efficient.optimizer(model.legacy.float32(env),cfg,model.legacy.float32(basis),True,wind_error_reserve=.25*cfg.umax,
            prepare_validation=True) if a.information_trigger=='lower-cost' else None
        archived=None
        if a.validation_reference:
            import importlib.util,sys
            spec=importlib.util.spec_from_file_location('archived_validation',a.validation_reference)
            module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
            archived=module.optimizer(model.legacy.float32(env),cfg,model.legacy.float32(basis),True,control_fn=model.control,
                reuse_training=True,batch_validation=True,wind_error_reserve=.25*cfg.umax,prepare_validation=True,information_trigger=a.information_trigger)
    reports=[]
    for path in sorted(a.recordings.glob('s*-center-routeprecision-posterior.npz')):
        source=path.with_name(path.name.replace('-posterior.npz','.json.gz'));record=json.load(gzip.open(source))
        b=restore_inference(path,support);cache=CachedFields(points,shape,cfg,grid['all_cells'])
        mean,cov=cache.map(b);jax.block_until_ready(mean);frame=record['belief_frames'][-1]
        np.testing.assert_allclose(mean,frame['mean'],atol=5.1e-6,rtol=0)
        np.testing.assert_allclose(jnp.sqrt(cov[:,0,0]),frame['std'],atol=5.1e-6,rtol=0)
        x=jnp.asarray(record['samples'][-1][1:5] if a.at_final_state else record['handoff']['state'])
        q=jnp.asarray(record['clarity_frames'][-1]['q'] if a.at_final_state else record['handoff']['clarity'])
        if a.timed_learning:
            from reachable_learning import ReachableLearning,measured_route_mask
            learner=ReachableLearning(env,grid,cfg,basis,'mask',global_goals=True,vector_precision=.2,
                prefix_screen='braking-feasible',learned_route_bounds=True,diversify_stalls=True,route_precision=True)
        rows=[]
        for repeat in range(5):
            key=jax.random.PRNGKey(19001+repeat);ts=time.perf_counter()
            tables=cache.build(b,jnp.array(len(record['observations'])),key,jax.random.PRNGKey(279929));jax.block_until_ready(tables)
            fields=time.perf_counter()-ts;ts=time.perf_counter()
            seeds,previous,metadata=model.legacy.seed_bank(x,q,env,cfg,None);geometry=time.perf_counter()-ts
            args=model.legacy.float32((x,q,tables,seeds,previous));ts=time.perf_counter()
            with jax.experimental.disable_x64():result=planner(*args,key);jax.block_until_ready(result)
            elapsed=time.perf_counter()-ts
            # Original and reused-score code must select the same commands.
            with jax.experimental.disable_x64():reference=original(*args,key);jax.block_until_ready(reference)
            np.testing.assert_allclose(result[0],reference[0],rtol=1e-6,atol=1e-6)
            np.testing.assert_array_equal(result[-1]['feasible'],reference[-1]['feasible'])
            rows.append(dict(repeat=repeat,compilation_included=repeat==0,all_field_construction_s=fields,
                geometry_s=geometry,search_s=result[-1]['search_s'],validation_s=result[-1]['validation_s'],
                optimization_s=elapsed,request_including_all_fields_s=fields+geometry+elapsed,
                selected_controls_match=True,validation_feasibility_match=True,
                feasible=not bool(result[-1]['no_feasible'])))
            if trusted is not None:
                ts=time.perf_counter()
                with jax.experimental.disable_x64():tr=trusted(*args,key);jax.block_until_ready(tr)
                trusted_s=time.perf_counter()-ts
                rows[-1].update(model_trust_optimization_s=trusted_s,model_trust_request_including_all_fields_s=fields+geometry+trusted_s,
                    model_trust_feasible=not bool(tr[-1]['no_feasible']),
                    selected_forecast_error_bound_m_s2=float(tr[-1]['forecast_error_bounds'][tr[-1]['selected']]),
                    model_trust_changes_selected_controls=not bool(np.allclose(tr[0],result[0],rtol=1e-6,atol=1e-6)))
                if trust_reference is not None:
                    ts=time.perf_counter()
                    with jax.experimental.disable_x64():base=trust_reference(*args,key);jax.block_until_ready(base)
                    base_s=time.perf_counter()-ts
                    np.testing.assert_array_equal(base[0],tr[0])
                    rows[-1].update(before_information_rule_optimization_s=base_s,information_rule_preserves_mbd_controls=True,
                        learning_requested=bool(tr[-1]['request_targeted_learning']))
                if archived is not None:
                    ts=time.perf_counter()
                    with jax.experimental.disable_x64():prior=archived(*args,key);jax.block_until_ready(prior)
                    before_s=time.perf_counter()-ts
                    np.testing.assert_array_equal(prior[0],tr[0]);np.testing.assert_array_equal(prior[-1]['feasible'],tr[-1]['feasible'])
                    np.testing.assert_allclose(prior[-1]['costs'],tr[-1]['costs'],rtol=1e-6,atol=1e-6)
                    rows[-1].update(before_validation_compilation_s=before_s,compiled_validation_preserves_controls_and_costs=True)
                if a.timed_learning:
                    if repeat==0:
                        actual=np.asarray(record['observations'])[:,1:3]
                        hm=np.asarray(basis.evaluate(jnp.asarray(actual),tables.mean));hs=np.asarray(basis.evaluate(jnp.asarray(actual),tables.std))[:,0]
                        _,bounds,_=measured_route_mask(actual,hm,hs,actual,cfg.umax)
                        learner.policy.history=[(z.copy(),float(f)) for z,f in zip(actual,bounds)]
                    learner.task_reference=np.asarray(tr[-1]['learning_reference']);ts=time.perf_counter()
                    mu,cov=b.predict(x[None,:2]);jax.block_until_ready(mu)
                    learner.plan(np.asarray(x),np.asarray(q),b,cache,tables,300.+repeat*.2,np.asarray(mu[0]),float(jnp.sqrt(cov[0,0,0])))
                    acquisition=time.perf_counter()-ts
                    rows[-1].update(information_acquisition_s=acquisition,
                        request_with_acquisition_s=fields+geometry+trusted_s+acquisition,
                        acquisition_scope='Timed even if this particular sampled MBD bank did not request learning; no observation is assimilated.')
            print(json.dumps(dict(id=record['id'],**rows[-1])),flush=True)
        reports.append(dict(id=record['id'],source=str(source),posterior_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            horizon_seconds=cfg.horizon,device=str(jax.devices()),scope=('Final recorded posterior and final executed state and clarity. ' if a.at_final_state else 'Final recorded posterior evaluated at the earlier actual handoff state and clarity. ')+
                'This is an isolated compute comparison, not replay of a historical decision. Includes all field banks, geometry and MBD. Excludes POAM assimilation/training. First repetition excluded from warm statistics.',rows=rows))
        (a.output/'report.json').write_text(json.dumps(reports,indent=2,allow_nan=False))
    signal.alarm(0)


if __name__=='__main__':main()
