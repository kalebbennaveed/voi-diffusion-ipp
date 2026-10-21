"""Replay the same real observation stream through alternative GP settings.

True wind is held-out scoring data only. No test-field value enters inference.
This isolates estimator representation/calibration from route-selection changes.
"""
import json,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from voi_diffusion_ipp.quarter.environment import initial_belief,wind
from voi_diffusion_ipp.quarter.continuous import ControlSettings
from voi_diffusion_ipp.wind.poam import POAM

root=Path('output/european_quarter/planner_investigation');source=root/'calibrated300/two_crosswind_proposed_8101.json';d=json.loads(source.read_text());cfg=ControlSettings(**d['config']);centers=initial_belief(cfg).basis.centers
xx,yy=np.meshgrid(np.linspace(-18,18,31)*cfg.environment_scale,np.array([8.,-23.,27.])*cfg.environment_scale);locations=jnp.array(np.c_[xx.ravel(),yy.ravel()],jnp.float32);truth=np.array(wind(locations,'crosswind',cfg.wind_strength,cfg.environment_scale));rows=[]
for name,factor,relocate,spectral in [('fixed',1.,False,16),('relocated',1.,True,16),('short_fixed',.4,False,16),('short_relocated',.4,True,64),('shorter_relocated',.2,True,64)]:
    model=POAM(np.array(centers),jax.random.PRNGKey(731),noise_std=cfg.observation_std,lengths=(5*cfg.environment_scale*factor,13*cfg.environment_scale*factor),prior_std=cfg.prior_std,spectral_count=spectral);b=model.planning_belief()
    condition=jax.jit(lambda b,p,y:b.condition(p,y,relocate=relocate));predict=jax.jit(lambda b,p:b.predict(p));residuals=[];coverage=[];nll=[];history=[];started=time.perf_counter()
    for i,o in enumerate(d['observations']):
        point=jnp.array([o['position']],jnp.float32);y=np.array(o['measurement']);mu,cov=map(np.array,predict(b,point));variance=max(float(cov[0,0,0]),1e-9);actual=np.array(wind(point,'crosswind',cfg.wind_strength,cfg.environment_scale))[0]
        residuals.append((mu[0]-actual)**2);coverage.extend(abs(mu[0]-actual)<=1.96*np.sqrt(variance));obs_var=variance+cfg.observation_std**2;nll.append(float(.5*np.sum(np.log(2*np.pi*obs_var)+(y-mu[0])**2/obs_var)))
        b=condition(b,point,jnp.array([y],jnp.float32))
        if (i+1)%30==0:
            m,c=map(np.array,predict(b,locations));history.append(dict(t=o['t'],held_out_corridor_rmse=float(np.sqrt(np.mean((m-truth)**2))),mean_std=float(np.sqrt(c[:,0,0]).mean())))
    m,c=map(np.array,predict(b,locations));_,exact=b.joint(locations);_,sampled=b.sample_moments(locations);exact=np.array(exact);sampled=np.array(sampled)
    row=dict(variant=name,length_factor=factor,inducing_relocation=relocate,spectral_nodes=spectral,observation_count=len(d['observations']),
             prequential_latent_rmse=float(np.sqrt(np.mean(residuals))),latent_95_percent_coverage=float(np.mean(coverage)),mean_observation_nll=float(np.mean(nll)),
             held_out_corridor_rmse=float(np.sqrt(np.mean((m-truth)**2))),covariance_sampling_relative_error=float(np.linalg.norm(sampled-exact)/np.linalg.norm(exact)),history=history,seconds=time.perf_counter()-started)
    rows.append(row);print(row,flush=True);(root/'estimator_replay.json').write_text(json.dumps(rows,indent=2));jax.clear_caches()
