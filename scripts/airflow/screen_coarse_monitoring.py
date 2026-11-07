"""Full-model seed diagnostics before an expensive closed-loop mission."""
import json,time
from pathlib import Path
from dataclasses import replace
import numpy as np
import jax,jax.numpy as jnp
import monitoring_mbd as m, monitoring_routes as routes
from voi_diffusion_ipp.wind.gp import InnerFields,WorldField
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
cfg=replace(m.Settings(),horizon_seconds=30,reference_tracking=True,multimode=True,proposals=16,tracking_kp=4.411764705882352,tracking_kv=7.352941176470588)
env,_=m.environment(cfg);x=jnp.array([1.45,.875,0.,0.]);q=jnp.full(len(env.cells),cfg.initial_q)
seeds,info=routes.seed_bank(np.asarray(x),np.asarray(q),env,cfg)
raw=np.load('results/airflow/monitoring_mbd_verified/s2-aware-plan-000.npz')
template=m.estimator(cfg,jax.random.PRNGKey(0)).planning_belief()
b=jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template),[jnp.asarray(raw[k]) for k in sorted(raw.files) if k.startswith('belief_leaf_')])
inner=InnerFields(b.basis,jnp.asarray(raw['inner_coefficients']));outer=InnerFields(b.basis,jnp.asarray(raw['outer_coefficients']))
evaluate=jax.jit(lambda th:m.forecast_cost(th,x,q,b,inner,outer,env,cfg,True))
small=env._replace(cells=env.cells[:1],targets=env.targets[:1],process_noise=env.process_noise[:1])
nominal=jax.jit(lambda th:m.forecast(th,x,q[:1],None,None,m.StillAir(),small,cfg,False))
fig,axes=plt.subplots(2,4,figsize=(16,10));reports=[]
for i,seed in enumerate(seeds):
 t=time.perf_counter();r=nominal(jnp.asarray(seed));cost=float(evaluate(jnp.asarray(seed)))
 a=axes.flat[i];a.set_aspect('equal');a.set_xlim(0,8);a.set_ylim(0,4.625)
 for cx,cy,rad in np.asarray(env.obstacles):a.add_patch(plt.Circle((cx,cy),rad,color='slategray'))
 if i>=2:a.plot(np.array(info['coarse_paths'][i-2])[:,0],np.array(info['coarse_paths'][i-2])[:,1],'--',alpha=.4)
 ref,acc=m.nominal_reference(jnp.asarray(seed),x,cfg);a.plot(np.array(ref)[:,0],np.array(ref)[:,1],color='darkgreen')
 a.plot(np.array(r.states)[:,0],np.array(r.states)[:,1],color='orange',alpha=.5)
 report=dict(index=i,cost=cost if np.isfinite(cost) else None,nominal_gap=float(r.min_clearance),nominal_contact=bool(r.collided),seconds=time.perf_counter()-t)
 a.set_title(f"Seed {i} · cost {report['cost']}\nNominal clearance {report['nominal_gap']:.3f} m",fontsize=9);reports.append(report);print(json.dumps(report),flush=True)
fig.tight_layout();fig.savefig('/private/tmp/coarse-seed-screen.png',dpi=140)
Path('/private/tmp/coarse-seed-screen.json').write_text(json.dumps(dict(seeds=reports,initialization=info)))
