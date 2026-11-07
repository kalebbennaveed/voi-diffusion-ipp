"""Actual-reading replay to validate spatial length scales; no synthetic data."""
import argparse,gzip,json,pickle,signal,time
from pathlib import Path
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import full_learning_monitoring as model
import cluster_environment as ce
from spatial_poam import SpatialPOAM,SpatialPOAMSettings
from cached_poam_fields import CachedFields

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--profiles',default='.1,.2,.5,1;.05,.2,.5,1');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600s bounded actual-data calibration replay')));signal.alarm(600)
source=Path('output/matched_duration_20261105/baseline-pilot-failed/s1-center-baseline1.json.gz');r=json.load(gzip.open(source))
cfg=model.settings();pcfg=model.legacy.physical_settings(cfg);env,grid,spec=ce.environment(pcfg,model.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
basis,points,shape=model.table_grid(env.bounds,.1);_,extended,_=model.table_grid(env.bounds,.4)
outside=np.any((np.asarray(extended)<np.asarray(env.bounds[0]))|(np.asarray(extended)>np.asarray(env.bounds[1])),axis=1)
support=np.vstack((np.asarray(env.cells),np.asarray(extended)[outside]));truth=model.m.physical.load_wind(1,model.m.physical.SITE/'revised')
actual=np.asarray(truth.at(jnp.asarray(grid['all_cells'])));free=grid['free_indices']
@jax.jit
def local(b,x):
 mean,cov=b.predict(x[None]);return mean[0],jnp.sqrt(cov[0,0,0])
reports=[]
for profile in a.profiles.split(';'):
 start=time.perf_counter();lengths=tuple(map(float,profile.split(',')));label=str(lengths[0]);seed=next(c['seed'] for c in ce.CASES if c['id']=='center')
 gp=SpatialPOAM(env.cells,env.bounds,settings=SpatialPOAMSettings(lengths_m=lengths),prior_support=support,training_seed=seed+441,factorized_support=True)
 cached=CachedFields(points,shape,cfg,grid['all_cells']);innovations=[];maprows=[]
 for k,observation in enumerate(r['observations']):
  t,x,y,wx,wy=observation;pos=jnp.array([x,y]);reading=np.array([wx,wy]);mu,std=local(gp.belief,pos);jax.block_until_ready(mu)
  error=float(np.linalg.norm(reading-np.asarray(mu)));z=error/np.sqrt(float(std)**2+.01)
  innovations.append(dict(t=t,mean=np.asarray(mu).tolist(),std=float(std),normalized_innovation=z,error_m_s=error))
  gp.update(pos[None],jnp.asarray(reading)[None],train_kernel=(k+1)%50==0);jax.block_until_ready(gp.belief)
  if t in (138,300,333,338):
   mean,cov=cached.map(gp.belief);jax.block_until_ready(mean);err=np.linalg.norm(np.asarray(mean)[free]-actual[free],axis=1);sigma=np.sqrt(np.asarray(cov)[free,0,0])
   maprows.append(dict(t=t,rmse_m_s=float(np.sqrt(np.mean(err**2))),safety_coverage=float(np.mean(err<=3*np.sqrt(sigma*sigma+.01)))))
   if t==333:
    with (a.output/f'length-{label}-t333.pkl').open('wb') as f:pickle.dump(gp,f)
  if t%60==0:print(json.dumps(dict(event='replay',profile=lengths,t=t,wall_s=time.perf_counter()-start)),flush=True)
 allz=np.array([x['normalized_innovation'] for x in innovations]);critical=[x for x in innovations if 333<=x['t']<=339.8]
 result=dict(lengths=lengths,premeasurement_three_sigma_coverage=float(np.mean(allz<=3)),quantile95=float(np.quantile(allz,.95)),quantile99=float(np.quantile(allz,.99)),max_innovation=float(allz.max()),map_checkpoints=maprows,critical=critical,wall_s=time.perf_counter()-start)
 reports.append(result);(a.output/'report.json').write_text(json.dumps(reports,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='critical'}),flush=True)
 (a.output/f'length-{label}-innovations.json').write_text(json.dumps(innovations))
signal.alarm(0)
