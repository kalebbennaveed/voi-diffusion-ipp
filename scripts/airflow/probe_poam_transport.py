"""Actual-observation replay and full-data variational audit; no wind truth."""
import signal,json,gzip,time
from pathlib import Path
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import implicit_monitoring_v4 as im
import cluster_environment as ce
from supported_poam_identity import condition_supported as identity
from voi_diffusion_ipp.wind.poam import variational_from_root


def main():
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('90 s transport audit limit')));signal.alarm(90)
 root=im.m.physical.ROOT;source=root/'results/airflow/implicit_v4_20261104/confirmed300'
 cfg=im.settings();pcfg=replace(im.legacy.physical_settings(cfg),safety_margin=cfg.safety_margin)
 env,grid,spec=ce.environment(pcfg,root/'results/airflow/cluster_revision_20261103/v2/environment-spec.json');basis,points,shape=im.table_grid(env.bounds)
 initial=im.initial_belief(env,points);reports=[]
 for scenario in [1,2]:
  r=json.load(gzip.open(source/f's{scenario}-center-aware.json.gz','rt'))
  obs=np.asarray([o for o in r['observations'] if o[0]<=295]);z=np.load(source/f's{scenario}-center-aware-plan-059.npz')
  beliefs={};times={}
  for name,update in [('recorded_transport',im.condition_supported),('identity_transport',identity)]:
   b=initial;mask=jnp.zeros(128,bool);ts=time.perf_counter()
   for o in obs:
    b,mask=update(b,jnp.asarray(o[None,1:3]),jnp.asarray(o[None,3:5]),mask)
   jax.block_until_ready(b);times[name]=time.perf_counter()-ts;beliefs[name]=b
  old=beliefs['recorded_transport'];assert np.allclose(old.mean,z['posterior_mean'])
  np.testing.assert_array_equal(old.basis.centers,beliefs['identity_transport'].basis.centers)
  b=beliefs['identity_transport'];K=b.kernel(b.basis.centers,b.basis.centers);Kuf=b.kernel(b.basis.centers,jnp.asarray(obs[:,1:3]))
  mean,cov=variational_from_root(K,Kuf@jnp.asarray(obs[:,3:5]),Kuf,b.noise_std)
  beliefs['batch_fixed_final_inducing']=b._replace(mean=mean,covariance=cov)
  metrics={}
  for name,b in beliefs.items():
   mean,cov=b.predict(env.cells);mu=np.asarray(mean);var=np.asarray(cov)[:,0,0];support=np.clip(1-var/float(jnp.exp(b.basis.parameters['log_amplitude'])),0,1);mag=np.linalg.norm(mu,axis=1)
   local,_=b.predict(jnp.asarray(obs[:,1:3]));err=np.linalg.norm(np.asarray(local)-obs[:,3:5],axis=1)
   metrics[name]=dict(max_mean_free=float(mag.max()),max_mean_low_support=float(mag[support<.1].max()),p95_mean_low_support=float(np.percentile(mag[support<.1],95)),mean_measurement_residual=float(err.mean()),p95_measurement_residual=float(np.percentile(err,95)),min_sigma=float(np.sqrt(np.maximum(var,0)).min()),max_sigma=float(np.sqrt(var).max()))
  row=dict(scenario=scenario,observations=len(obs),max_measured_speed=float(np.linalg.norm(obs[:,3:5],axis=1).max()),metrics=metrics,replay_seconds=times);reports.append(row);print(json.dumps(row),flush=True)
 (root/'output/diagnostics/implicit_v2_20261104/poam-transport.json').write_text(json.dumps(reports,indent=2));signal.alarm(0)

if __name__=='__main__':main()
