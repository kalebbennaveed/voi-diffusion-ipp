import json,pickle,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import rmse_monitoring as im
import cluster_environment as ce
from cached_poam_fields import CachedFields
cfg=im.settings();root=im.m.physical.ROOT
env,grid,_=ce.environment(im.legacy.physical_settings(cfg),root/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
basis,points,shape=im.table_grid(env.bounds,.1)
with (root/'output/gp_safety_20261104/sampling/s1-dense-profile.pkl').open('rb') as f:gp=pickle.load(f)
b=gp.belief;cache=CachedFields(points,shape,cfg,grid['all_cells']);old=im.field_builder(points,shape,cfg)
args=(b,jnp.array(2361),jax.random.PRNGKey(29),jax.random.PRNGKey(33))
ref=old(*args);jax.block_until_ready(ref)
rows=[]
for i in range(3):
    t=time.perf_counter();new=cache.build(*args);jax.block_until_ready(new);elapsed=time.perf_counter()-t
    error={name:float(np.max(np.abs(np.asarray(a)-np.asarray(c)))) for name,a,c in zip(ref._fields,ref,new)}
    np.testing.assert_allclose(new.mean,ref.mean,atol=2e-6,rtol=1e-6)
    np.testing.assert_allclose(new.std,ref.std,atol=2e-6,rtol=1e-6)
    for field in ['planning','safety','validation']:np.testing.assert_allclose(getattr(new,field),getattr(ref,field),atol=2e-6,rtol=1e-6)
    rows.append(dict(iteration=i,seconds=elapsed,max_errors=error));print(json.dumps(rows[-1]),flush=True)
t=time.perf_counter();ss=cache.safety(b,new,args[-1]);jax.block_until_ready(ss);print('safety_seconds',time.perf_counter()-t)
(root/'output/gp_safety_20261104/cached-field-check.json').write_text(json.dumps(rows,indent=2))
