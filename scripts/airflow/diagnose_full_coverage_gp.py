"""Controlled GP-only test: every free cell measured, no planner or safety.

The sampling order depends only on geometry and a fixed random seed. Truth is
used by the synthetic local sensor and evaluator, never by GP fitting.
"""
import argparse
import gzip
import hashlib
import json
import signal
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist
import cluster_environment as ce
import rmse_monitoring as rm
from waypoint_poam import condition, train
from observed_poam import _double
from voi_diffusion_ipp.wind.poam import POAM, select_inducing, attentive_kernel, variational_from_root, chol_solve


def metrics(mu, variance, truth):
    error = np.linalg.norm(mu-truth, axis=1)
    return dict(vector_rmse_m_s=float(np.sqrt(np.mean(error**2))),
        maximum_vector_error_m_s=float(error.max()),
        mean_std_m_s=float(np.sqrt(np.maximum(variance,0)).mean()),
        latent_three_sigma_disk_coverage=float(np.mean(error<=3*np.sqrt(np.maximum(variance,0)))),
        safety_disk_coverage=float(np.mean(error<=3*np.sqrt(np.maximum(variance,0)+.01))))


def full_gp(kernel, x, y, query):
    K = np.array(kernel(x,x))
    L, lower = cho_factor(K+.010001*np.eye(len(x)), lower=True)
    alpha = cho_solve((L,lower),y)
    k = np.asarray(kernel(query,x))
    mu = k@alpha
    residual = solve_triangular(L,k.T,lower=True)
    # Every attentive/RBF kernel in this diagnostic has constant diagonal.
    var = float(K[0,0])-np.einsum('ij,ij->j',residual,residual)
    return mu,var


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inducing',type=int,default=128)
    p.add_argument('--normalize-attention',action='store_true')
    p.add_argument('--amplitude-rate',type=float,default=.001)
    p.add_argument('--scenarios',default='1,2')
    p.add_argument('--skip-comparators',action='store_true')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--independent-batches',action='store_true')
    p.add_argument('--observation-batch',type=int,default=1)
    p.add_argument('--lengths',default='.3,.8,1.3,1.8')
    p.add_argument('--seed-offset',type=int,default=0)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new output directory')
    a.output.mkdir(parents=True)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s diagnostic budget')))
    signal.alarm(180);start=time.perf_counter()
    env,grid,spec=ce.environment(rm.legacy.physical_settings(rm.settings()),
        rm.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
    _,pts,_=rm.table_grid(env.bounds)
    options=dict(prior_std=1.8,attention_bounds=env.bounds if a.normalize_attention else None,
                 lengths=tuple(map(float,a.lengths.split(','))))
    model=POAM(env.cells[:a.inducing],jax.random.PRNGKey(731),**options)
    z=select_inducing(model.parameters,env.cells,model.lengths,a.inducing)
    initial=_double(POAM(z,jax.random.PRNGKey(731),**options).planning_belief())
    x=np.asarray(env.cells);rng=np.random.default_rng(71104)
    raw=rng.uniform([.0625,.0625],[7.9375,5.9375],size=(2000,2))
    query=raw[np.asarray(jax.vmap(lambda p:env.clearance(jnp.r_[p,0.,0.]))(jnp.asarray(raw)))>.07][:1000]
    pred=jax.jit(lambda b,x:b.predict(x));reports=[]
    for scenario in map(int,a.scenarios.split(',')):
        truth=rm.m.physical.load_wind(scenario,rm.m.physical.SITE/'revised')
        actual=np.asarray(truth.at(jnp.asarray(x)));actual_q=np.asarray(truth.at(jnp.asarray(query)))
        rng=np.random.default_rng(20261104+scenario+a.seed_offset);y=actual+.1*rng.normal(size=actual.shape)
        order=rng.permutation(len(x));b=initial;mask=jnp.zeros(a.inducing,bool)
        moment=jax.tree.map(jnp.zeros_like,b.basis.parameters);variance=moment;step=jnp.array(0)
        history=[]
        timings=[]
        for begin in range(0,len(order),a.observation_batch):
            ids=order[begin:begin+a.observation_batch];i=begin+len(ids)-1
            stamp=time.perf_counter()
            b,mask=condition(b,jnp.asarray(x[ids]),jnp.asarray(y[ids]),mask)
            if (i+1)%10==0:
                idx=order[rng.integers(0,i+1,size=(4,a.batch_size) if a.independent_batches else a.batch_size)]
                b,moment,variance,step,loss=train(b,jnp.asarray(x[idx]),jnp.asarray(y[idx]),jnp.array(i+1,dtype=jnp.float64),moment,variance,step,a.amplitude_rate)
            jax.block_until_ready(b);timings.append(time.perf_counter()-stamp)
            if (i+1)%500==0 or i==len(x)-1:
                mu,cov=pred(b,jnp.asarray(x));row=dict(observations=i+1,**metrics(np.asarray(mu),np.asarray(cov)[:,0,0],actual))
                history.append(row);print(json.dumps(dict(scenario=scenario,model='online_original',**row)),flush=True)
        results={}
        q_all=np.vstack((x,query));expected=np.vstack((actual,actual_q))
        def save(name,mu,var):
            mu=np.asarray(mu);var=np.asarray(var)
            results[name]=dict(measured_cells=metrics(mu[:len(x)],var[:len(x)],actual),
                off_grid=metrics(mu[len(x):],var[len(x):],actual_q))
            np.savez_compressed(a.output/f's{scenario}-{name}.npz',mean=mu,variance=var)
            print(json.dumps(dict(scenario=scenario,model=name,**results[name])),flush=True)
        mu,cov=pred(b,jnp.asarray(q_all));save('online_original',mu,np.asarray(cov)[:,0,0])
        kernel=lambda u,v:attentive_kernel(b.basis.parameters,jnp.asarray(u),jnp.asarray(v),b.basis.lengths)
        K=kernel(b.basis.centers,b.basis.centers);Kuf=kernel(b.basis.centers,x)
        mean,cov=variational_from_root(K,Kuf@jnp.asarray(y),Kuf,.1)
        batch=b._replace(mean=mean,covariance=cov);mu,var=pred(batch,jnp.asarray(q_all));save('batch_same_sparse_kernel',mu,np.asarray(var)[:,0,0])
        if not a.skip_comparators:
            mu,var=full_gp(kernel,x,y,q_all);save('full_gp_same_kernel',mu,var)
        # Exact RBF comparators diagnose spatial resolution only, not replacements.
        for length in (() if a.skip_comparators else (.125,.3)):
            kernel_rbf=lambda u,v:3.24*np.exp(-cdist(np.asarray(u),np.asarray(v),'sqeuclidean')/(2*length**2))
            mu,var=full_gp(kernel_rbf,x,y,q_all);save(f'full_gp_rbf_{length}',mu,var)
        np.savez_compressed(a.output/f's{scenario}-inputs.npz',x=x,y=y,order=order,query=query,
            true_at_measurements=actual,true_at_query=actual_q,centers=b.basis.centers,
            **{f'parameter_{k}':np.asarray(v) for k,v in b.basis.parameters.items()})
        reports.append(dict(scenario=scenario,actual_measurements=len(x),off_grid_queries=len(query),history=history,results=results,
            update_timing_s=dict(median=float(np.median(timings[2:-1])),max=float(np.max(timings[2:-1]))),update_timings_s=timings))
        (a.output/'report.json').write_text(json.dumps(dict(protocol='Every free cell measured once with 0.1m/s independent sensor noise; no motion constraints or RPCBF. Geometric order and noise fixed independently of truth. Full GP/RBF fits are offline comparators only.',inducing=a.inducing,lengths=options['lengths'],observation_batch=a.observation_batch,normalized_attention=a.normalize_attention,amplitude_rate=a.amplitude_rate,batch_size=a.batch_size,independent_batches=a.independent_batches,seed_offset=a.seed_offset,reports=reports,wall_seconds=time.perf_counter()-start),indent=2))
    signal.alarm(0)


if __name__=='__main__':main()
