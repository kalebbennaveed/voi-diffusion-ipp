"""Held-out physical-field diagnostic; truth is never passed to the planner."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
from run_compact_mbd_study import configuration
from voi_diffusion_ipp.quarter.environment import initial_belief
from voi_diffusion_ipp.quarter.benchmark_environments import SCENARIOS,make_environment,flow,flow_parameters,survey_locations

prediction=jax.jit(lambda b,p:b.mean_at(p))
rows=[]
for s in SCENARIOS:
 env=make_environment(s.name);pos=survey_locations(env,64)
 observed=np.asarray(flow(pos,flow_parameters(s))).copy()+.25*np.random.default_rng(16102).normal(size=(64,2))
 cfg=configuration(geometry='suite',wind_inducing_count=64,wind_kernel_lengths=(.6,1.8),prior_std=1.8,
     initial_wind_observations=tuple(tuple(r) for r in np.c_[pos,observed].tolist()))
 b=initial_belief(cfg)
 lo,hi=np.asarray(env.bounds);xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],61),np.linspace(lo[1],hi[1],61));p=np.c_[xx.ravel(),yy.ravel()]
 pred=np.asarray(prediction(b,jnp.asarray(p)));true=np.asarray(flow(p,flow_parameters(s)))
 active=np.linalg.norm(true,axis=1)>1.5
 error=float(np.sqrt(np.mean((pred[active]-true[active])**2)))
 row={'case':s.name,'active_wind_rmse':error,'all_wind_rmse':float(np.sqrt(np.mean((pred-true)**2))),'station_count':64,'inducing_count':64,'kernel_lengths':[.6,1.8]}
 rows.append(row);print(row,flush=True)
p=Path('output/disturbance_suite/mapping_diagnostic64.json');p.write_text(json.dumps(rows,indent=2))
