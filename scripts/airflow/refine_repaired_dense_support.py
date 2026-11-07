"""Refine the existing support-corrected sampler, preserving the exact GP fit."""
import json,pickle,signal,time
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import rmse_monitoring as rm
from voi_diffusion_ipp.wind.poam import attention_weights
from voi_diffusion_ipp.wind.poam_belief import spectral_nodes
ROOT=rm.m.physical.ROOT;OUT=ROOT/'output/gp_safety_20261104/sampling'
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s support audit limit')));signal.alarm(180)
reports=[]
for scenario in (1,2):
 with (OUT/f's{scenario}-profile.pkl').open('rb') as f:gp=pickle.load(f)
 data=np.load(ROOT/f'output/gp_repair_20261104/resolved768/s{scenario}-inputs.npz')
 query=jnp.asarray(data['query'][::10]);b=gp.belief
 _,exact=jax.jit(lambda b,p:b.joint(p))(b,query)
 rows=[]
 for spacing in (.125,):
  _,extended,_=rm.table_grid(jnp.array([[0.,0.],[8.,6.]]),.4)
  exterior=np.asarray(extended)[np.any((np.asarray(extended)<0)|(np.asarray(extended)>np.array([8.,6.])),axis=1)]
  support=jnp.asarray(np.vstack((data['x'],exterior)))
  start=time.perf_counter();K=b.kernel(support,support)
  L=jnp.linalg.cholesky((K+K.T)/2+1e-4*jnp.eye(len(K)));inv=jsp.linalg.cho_solve((L,True),jnp.eye(len(K)))
  basis=b.basis._replace(support=support,support_weights=attention_weights(b.basis.parameters,support),support_cholesky=L,support_inverse=inv)
  test=b._replace(basis=basis)
  _,cov=jax.jit(lambda b,p:b.sample_moments(p))(test,query);jax.block_until_ready(cov)
  rows.append(dict(spacing_m=spacing,support_points=len(support),relative_covariance_error=float(np.linalg.norm(cov-exact)/np.linalg.norm(exact)),max_variance_error=float(np.abs(np.diag(cov-exact)).max()),seconds=time.perf_counter()-start))
  print(json.dumps(dict(scenario=scenario,**rows[-1])),flush=True)
  if spacing==.125:
   gp.belief=test
   with (OUT/f's{scenario}-dense-profile.pkl').open('wb') as f:pickle.dump(gp,f)
 reports.append(dict(scenario=scenario,results=rows));(OUT/'dense-support-refinement.json').write_text(json.dumps(reports,indent=2))
signal.alarm(0)
