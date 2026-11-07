"""Convergence of existing spectral quadrature; no change to GP inference."""
import json,pickle,signal,time
import numpy as np
import jax
import jax.numpy as jnp
import rmse_monitoring as rm
from voi_diffusion_ipp.wind.poam_belief import spectral_nodes
ROOT=rm.m.physical.ROOT
OUT=ROOT/'output/gp_safety_20261104/sampling'
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s refinement budget')));signal.alarm(180)
report=[]
for scenario in (1,2):
    with (OUT/f's{scenario}-profile.pkl').open('rb') as f:gp=pickle.load(f)
    inp=np.load(ROOT/f'output/gp_repair_20261104/resolved768/s{scenario}-inputs.npz')
    query=jnp.asarray(inp['query'][::10]);b=gp.belief
    _,exact=jax.jit(lambda b,p:b.joint(p))(b,query)
    rows=[]
    for count in (32,64,128,256,512):
        start=time.perf_counter();test=b._replace(basis=b.basis._replace(frequencies=spectral_nodes(count)))
        _,cov=jax.jit(lambda b,p:b.sample_moments(p))(test,query);jax.block_until_ready(cov)
        relative=float(np.linalg.norm(cov-exact)/np.linalg.norm(exact))
        rows.append(dict(nodes=count,relative_covariance_error=relative,max_variance_error=float(np.abs(np.diag(cov-exact)).max()),seconds=time.perf_counter()-start))
        print(json.dumps(dict(scenario=scenario,**rows[-1])),flush=True)
    report.append(dict(scenario=scenario,results=rows))
    (OUT/'refinement.json').write_text(json.dumps(report,indent=2))
signal.alarm(0)
