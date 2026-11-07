"""Inference-only probe at a recorded stalled state; no hidden physical field."""
import argparse,gzip,json,pickle,time,signal
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import full_learning_monitoring as model
import cluster_environment as ce
from information_paths import InformationPathPlanner
from cached_poam_fields import CachedFields
from task_wind_acquisition import deficit_weights
from voi_diffusion_ipp.wind.poam import attention_weights

def main():
    p=argparse.ArgumentParser();p.add_argument('record',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frontier-graph',action='store_true');p.add_argument('--frontier-prefixes',action='store_true');a=p.parse_args()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120s diagnostic guard')));signal.alarm(120)
    root=model.m.physical.ROOT;cfg=model.settings();env,grid,_=ce.environment(model.legacy.physical_settings(cfg),root/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
    basis,points,shape=model.table_grid(env.bounds,.1);cache=CachedFields(points,shape,cfg,grid['all_cells'])
    r=json.load(gzip.open(a.record));data=np.load(a.record.with_name(a.record.name.replace('.json.gz','-posterior.npz')))
    with (root/f'output/gp_safety_20261104/sampling/s{r["scenario"]}-dense-profile.pkl').open('rb') as f:template=pickle.load(f).belief
    # Only prediction/sampling is used here. No assimilation is performed with
    # this inference checkpoint: the online sufficient statistics are absent.
    params={k[6:]:jnp.asarray(data[k]) for k in data.files if k.startswith('param_')};z=jnp.asarray(data['centers'])
    basis_gp=template.basis._replace(parameters=params,centers=z,lengths=jnp.asarray(data['lengths']),
        center_weights=attention_weights(params,z),support_weights=attention_weights(params,template.basis.support))
    b=template._replace(basis=basis_gp,mean=jnp.asarray(data['mean']),covariance=jnp.asarray(data['covariance']),inverse=jnp.asarray(data['inverse']))
    K=b.kernel(b.basis.support,b.basis.support);L=jnp.linalg.cholesky((K+K.T)/2+1e-4*jnp.eye(len(K)))
    b=b._replace(basis=b.basis._replace(support_cholesky=L,support_inverse=jsp.linalg.cho_solve((L,True),jnp.eye(len(K)))))
    x=jnp.asarray(r['samples'][-1][1:5]);q=jnp.asarray(r['clarity_frames'][-1]['q'])
    tables=cache.build(b,jnp.array(len(r['observations'])),jax.random.PRNGKey(7101),jax.random.PRNGKey(279929));jax.block_until_ready(tables)
    planner=InformationPathPlanner(env,grid,cfg,basis,frontier_graph=a.frontier_graph,frontier_prefixes=a.frontier_prefixes);rows=[]
    for kind in ['all','mask']:
        weights=np.zeros(len(grid['all_cells']));weights[grid['free_indices']]=1. if kind=='all' else deficit_weights(np.asarray(q),env.targets,env.process_noise,cfg.horizon)
        weights/=weights.sum()
        for repeat in range(2):
            start=time.perf_counter();refs,aa,tables,decision=planner.plan(x,q,b,cache,tables,weights,jax.random.PRNGKey(7101))
            row=dict(kind=kind,repeat=repeat,seconds=time.perf_counter()-start,**decision)
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['predicted_states','candidate_paths','path','selected_parameters','initialization']}),flush=True)
    a.output.parent.mkdir(exist_ok=True,parents=True)
    a.output.write_text(json.dumps(dict(record=str(a.record),state=np.asarray(x).tolist(),scope='Frozen recorded posterior; model forecasts only, no true wind and no claimed executed mission. Timings are a development probe, not an isolated benchmark.',results=rows),indent=2,default=lambda x:np.asarray(x).tolist()))
    signal.alarm(0)

if __name__=='__main__':main()
