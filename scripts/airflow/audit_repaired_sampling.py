"""Reconstruct the validated actual-data fit and audit downstream field numerics."""
import json,pickle,signal,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import rmse_monitoring as rm
import cluster_environment as ce
from spatial_poam import SpatialPOAM

ROOT=rm.m.physical.ROOT
OUT=ROOT/'output/gp_safety_20261104/sampling'
SOURCE=ROOT/'output/gp_repair_20261104/resolved768'

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('240s audit limit')))
    signal.alarm(240)
    env,grid,_=ce.environment(rm.legacy.physical_settings(rm.settings()),ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
    _,points,_=rm.table_grid(env.bounds)
    pred=jax.jit(lambda b,p:b.predict(p));moments=jax.jit(lambda b,p:b.sample_moments(p))
    joint=jax.jit(lambda b,p:b.joint(p));reports=[]
    for scenario in (1,2):
        started=time.perf_counter();data=np.load(SOURCE/f's{scenario}-inputs.npz')
        gp=SpatialPOAM(env.cells,env.bounds,prior_support=np.asarray(points)[::3],training_seed=20261104+scenario)
        gp.rng.normal(size=data['y'].shape);gp.rng.permutation(len(data['x']))
        order=data['order'];times=[]
        for start in range(0,len(order),50):
            ids=order[start:start+50];stamp=time.perf_counter()
            gp.update(data['x'][ids],data['y'][ids],train_kernel=(start+len(ids))%10==0)
            jax.block_until_ready(gp.belief);times.append(time.perf_counter()-stamp)
        b=gp.belief
        with (OUT/f's{scenario}-profile.pkl').open('wb') as f:pickle.dump(gp,f)
        query=jnp.asarray(data['query'][::10]);mu,cov=joint(b,query);sm,sc=moments(b,query)
        dense=pred(b,jnp.asarray(np.vstack((data['x'],data['query']))))
        saved=np.load(SOURCE/f's{scenario}-online_original.npz')
        np.testing.assert_allclose(dense[0],saved['mean'],atol=1e-9)
        np.testing.assert_allclose(dense[1][:,0,0],saved['variance'],atol=1e-9)
        row=dict(scenario=scenario,fit_matches_validated_predictions=True,
            sampler_relative_covariance_error=float(np.linalg.norm(sc-cov)/np.linalg.norm(cov)),
            sampler_max_variance_error=float(np.max(np.abs(np.diag(sc-cov)))),caches=[])
        for spacing in (.2,.1):
            basis,pts,shape=rm.table_grid(env.bounds,spacing)
            mean,variance=pred(b,pts)
            interp=basis.evaluate(query,mean.reshape(shape));sigma=basis.evaluate(query,jnp.sqrt(variance[:,0,0]).reshape(*shape[:2],1))[:,0]
            row['caches'].append(dict(spacing_m=spacing,
                mean_vector_rmse_m_s=float(jnp.sqrt(jnp.mean(jnp.sum((interp-mu)**2,axis=1)))),
                mean_vector_max_error_m_s=float(jnp.linalg.norm(interp-mu,axis=1).max()),
                std_max_error_m_s=float(jnp.abs(sigma-jnp.sqrt(jnp.diag(cov))).max())))
        row.update(warm_gp_batch_median_s=float(np.median(times[2:-1])),wall_seconds=time.perf_counter()-started)
        reports.append(row);print(json.dumps(row),flush=True)
        (OUT/'report.json').write_text(json.dumps(reports,indent=2))
    signal.alarm(0)

if __name__=='__main__':main()
