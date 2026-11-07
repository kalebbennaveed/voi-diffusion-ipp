"""Separate belief-sensitivity diagnostic; never feeds a mission or controller.

Compare identical starts under a cold posterior, a posterior fitted to an
explicit noisy spatial survey, and an explicitly privileged known-field check.
The survey is diagnostic initial information, not robot exploration credit.
"""
from pathlib import Path
import json,time
from dataclasses import replace
import numpy as np
import jax,jax.numpy as jnp
import monitoring_mbd as m
from voi_diffusion_ipp.wind.gp import InnerFields
from voi_diffusion_ipp.wind.oracle import KnownWind,KnownBasis

if __name__=='__main__':
    root=Path('results/airflow/monitoring_coarse_final')
    protocol=json.loads((root/'s2-aware-protocol.json').read_text());cfg=m.Settings(**protocol['settings'])
    env,_=m.environment(cfg);raw=np.load(root/'s2-aware-plan-000.npz')
    truth=m.physical.load_wind(2)
    xx,yy=np.meshgrid(np.arange(.25,8,.5),np.arange(.25,4.625,.5));positions=np.c_[xx.ravel(),yy.ravel()]
    positions=positions[m.physical.clearance(positions)>.12]
    values=np.asarray(truth.at(positions))+cfg.observation_std*np.random.default_rng(1101).normal(size=positions.shape)
    model=m.estimator(cfg,jax.random.PRNGKey(protocol['seed']+101))
    model.update(jnp.asarray(positions),jnp.asarray(values),jax.random.PRNGKey(1102),train_steps=cfg.poam_train_steps,relocate=True)
    b=model.planning_belief();eps=jax.random.normal(jax.random.PRNGKey(protocol['seed']+1000),(cfg.inner_samples+cfg.worlds,*b.sample_shape));fields=b.inner_fields(eps)
    inner=InnerFields(fields.basis,fields.coefficients[:cfg.inner_samples]);outer=InnerFields(fields.basis,fields.coefficients[cfg.inner_samples:])
    known=KnownWind(KnownBasis(truth));ki=known.inner_fields(jnp.zeros((cfg.inner_samples,1,2)));ko=known.inner_fields(jnp.zeros((cfg.worlds,1,2)))
    x=jnp.asarray(raw['initial_state']);q=jnp.asarray(raw['initial_clarity'])
    survey=jax.jit(lambda th:m.forecast_cost(th,x,q,b,inner,outer,env,cfg,True))
    oracle=jax.jit(lambda th:m.forecast_cost(th,x,q,known,ki,ko,env,cfg,True))
    rows=[]
    for i,theta in enumerate(raw['initials']):
        start=time.perf_counter();row=dict(seed=i,survey_cost=float(survey(jnp.asarray(theta))),known_field_diagnostic_cost=float(oracle(jnp.asarray(theta))),seconds=time.perf_counter()-start)
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(scope='Belief-sensitivity diagnostic only; no mission uses these measurements or the known-field controller.',scenario=2,survey_points=len(positions),survey_positions=positions.tolist(),survey_values=values.tolist(),observation_std=cfg.observation_std,settings=protocol['settings'],rows=rows)
    (root.parent/'monitoring_belief_diagnostic.json').write_text(json.dumps(report,indent=2))
