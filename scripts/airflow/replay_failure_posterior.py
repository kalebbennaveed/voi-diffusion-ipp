"""Recover exact POAM checkpoints from recorded actual observations only."""
import gzip,json,pickle,signal,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import full_learning_monitoring as model
import cluster_environment as ce
from spatial_poam import SpatialPOAM,SpatialPOAMSettings

out=Path('output/matched_duration_20261105/confidence-diagnostic');out.mkdir(exist_ok=True)
source=out.parent/'resolved-variance-failed/s1-center-baseline1.json.gz'
r=json.load(gzip.open(source));start=time.perf_counter()
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('400 s replay limit')));signal.alarm(400)
cfg=model.settings();pcfg=model.legacy.physical_settings(cfg)
env,grid,spec=ce.environment(pcfg,model.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
_,extended,_=model.table_grid(env.bounds,.4)
outside=np.any((np.asarray(extended)<np.asarray(env.bounds[0]))|(np.asarray(extended)>np.asarray(env.bounds[1])),axis=1)
support=np.vstack((np.asarray(env.cells),np.asarray(extended)[outside]))
seed=next(c['seed'] for c in ce.CASES if c['id']=='center')
gp=SpatialPOAM(env.cells,env.bounds,settings=SpatialPOAMSettings(lengths_m=(.1,.2,.5,1.),observation_first_inducing=True),prior_support=support,training_seed=seed+441,factorized_support=True)
for k,row in enumerate(r['observations']):
 t,x,y,wx,wy=row
 if t>285:break
 gp.update(jnp.array([[x,y]]),jnp.array([[wx,wy]]),train_kernel=(k+1)%50==0);jax.block_until_ready(gp.belief)
 if t in (270,285):
  with (out/f'posterior-t{int(t)}.pkl').open('wb') as f:pickle.dump(gp,f)
  print(json.dumps(dict(checkpoint=t,observations=len(gp.positions),wall_seconds=time.perf_counter()-start)),flush=True)
 if t%60==0:print(json.dumps(dict(t=t,wall_seconds=time.perf_counter()-start)),flush=True)
reference=np.load(out.parent/'resolved-variance-failed/s1-center-baseline1-handoff.npz')
with (out/'posterior-t270.pkl').open('rb') as f:handoff=pickle.load(f)
for name,a in [('mean',handoff.belief.mean),('covariance',handoff.belief.covariance),('inducing',handoff.belief.basis.centers)]:
 np.testing.assert_allclose(a,reference[name],atol=1e-10,rtol=1e-10)
(out/'replay-verification.json').write_text(json.dumps(dict(recorded_handoff_reproduced=True,observations_270=len(handoff.positions),wall_seconds=time.perf_counter()-start),indent=2))
signal.alarm(0)
