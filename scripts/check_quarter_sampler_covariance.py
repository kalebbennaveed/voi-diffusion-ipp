"""Check the actual continuous field sampler against the GP predictive covariance."""
import json,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from voi_diffusion_ipp.quarter.continuous import ControlSettings
from voi_diffusion_ipp.quarter.environment import initial_belief,table_sampler
from voi_diffusion_ipp.wind.poam_belief import spectral_nodes
from voi_diffusion_ipp.wind.poam import POAM

root=Path('output/european_quarter/planner_investigation');d=json.loads((root/'calibrated300/two_crosswind_proposed_8101.json').read_text());cfg=ControlSettings(**d['config']);b=initial_belief(cfg);condition=jax.jit(lambda b,p,y:b.condition(p,y))
for start in range(0,len(d['observations']),5):
    obs=d['observations'][start:start+5];b=condition(b,jnp.array([o['position'] for o in obs],jnp.float32),jnp.array([o['measurement'] for o in obs],jnp.float32))
rng=np.random.default_rng(452);bounds=np.array(d['environment']['bounds']);points=rng.uniform(bounds[0],bounds[1],size=(500,2));valid=np.ones(len(points),bool)
for r in d['environment']['buildings']:valid&=~(np.all(points>=r[:2],axis=1)&np.all(points<=r[2:4],axis=1))
points=jnp.array(points[valid][:120],jnp.float32);exact=np.array(b.joint(points)[1]);rows=[]
cases=[]
for n in [16,32,64,128,256,512]:cases.append((f'spectral_{n}',b._replace(basis=b.basis._replace(frequencies=spectral_nodes(n)))))
for n in [11,17,25]:
    xx,yy=np.meshgrid(np.linspace(-45,45,n)*cfg.environment_scale,np.linspace(-35,44,n)*cfg.environment_scale)
    model=POAM(np.array(b.basis.centers),jax.random.PRNGKey(731),noise_std=cfg.observation_std,lengths=(5*cfg.environment_scale,13*cfg.environment_scale),prior_std=cfg.prior_std,spectral_count=16,prior_support=np.c_[xx.ravel(),yy.ravel()].astype(np.float32))
    cases.append((f'support_{n}x{n}',b._replace(basis=model.planning_belief().basis)))
for label,belief in cases:
    start=time.perf_counter();sampled=np.array(jax.jit(lambda b,p:b.sample_moments(p)[1])(belief,points));diagonal=np.diag(exact);got=np.diag(sampled)
    sample=table_sampler(14,cfg.environment_scale);timings=[]
    for repeat in range(2):
        s=time.perf_counter();values=sample(belief,jax.random.PRNGKey(922));jax.block_until_ready(values);timings.append(time.perf_counter()-s)
    row=dict(variant=label,covariance_relative_error=float(np.linalg.norm(sampled-exact)/np.linalg.norm(exact)),marginal_variance_relative_error=float(np.linalg.norm(got-diagonal)/np.linalg.norm(diagonal)),maximum_std_difference=float(np.max(abs(np.sqrt(np.maximum(got,0))-np.sqrt(np.maximum(diagonal,0))))),table_compile_seconds=timings[0],table_warm_seconds=timings[1],seconds=time.perf_counter()-start)
    rows.append(row);print(row,flush=True);(root/'sampler_covariance.json').write_text(json.dumps(rows,indent=2));jax.clear_caches()
