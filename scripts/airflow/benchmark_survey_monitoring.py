"""Bounded synchronized timing gate for the added survey planner."""
import json,time
from pathlib import Path
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import legacy_monitoring as l
import monitoring_mbd as m
import survey_monitoring as active
import survey_routes
from observed_poam import condition_observed

def main():
    cfg=replace(l.Settings(),safety_margin=.12,construction_worlds=8,prediction_filter='hard',iterations=2,proposals=14,batch_size=7,worlds=1,validation_worlds=1,validation_batch_size=5)
    env,_=m.environment(l.physical_settings(cfg));x=jnp.array([1.45,.875,0.,0.]);q=jnp.full(len(env.cells),.74)
    b=l.matched_prior(env)
    # Synthetic local zero-wind observation only for the timing gate.
    b=condition_observed(b,x[None,:2],jnp.zeros((1,2)))
    bp,xp,qp=l.float32((b,x,q));report=[]
    sampler=active.field_sampler(cfg)
    with jax.experimental.disable_x64():plan=active.optimizer(l.float32(env),cfg)
    for i in range(3):
        t=time.perf_counter();seeds,prev,meta,targets,weights=survey_routes.seed_bank(x,q,env,cfg,b);geometry=time.perf_counter()-t;t0=time.perf_counter()
        mean,inner,worlds,validation=l.float32(sampler(b,jax.random.PRNGKey(9101),jax.random.PRNGKey(279929)));jax.block_until_ready(validation);sampling=time.perf_counter()-t0;t0=time.perf_counter()
        with jax.experimental.disable_x64():
            r=plan(bp,xp,qp,mean,inner,worlds,validation,l.float32(prev),jax.random.PRNGKey(10101),l.float32(seeds),jnp.asarray(targets,jnp.float32),jnp.asarray(weights,jnp.float32),jnp.asarray(meta['task_route_indices'],jnp.int32),jnp.array(300.,jnp.float32));jax.block_until_ready(r)
        row=dict(compile=i==0,geometry=geometry,fields=sampling,optimizer=time.perf_counter()-t0,total=time.perf_counter()-t,explore=bool(r.exploring),score=float(r.score),gain=float(r.information[r.index]),task=float(r.expected),uncertainty=float(r.route_uncertainty),relevance=float(r.route_relevance))
        report.append(row);Path('results/airflow/survey_monitoring_benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
        if i and row['total']>1:break
if __name__=='__main__':main()
