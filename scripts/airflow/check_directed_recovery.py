"""30 s counterfactual at the recorded Scenario 2 survey deadlock.

Start from the displayed state/clarity (recording precision) and exact saved
POAM. Restore the recorded recovery goal and measured stagnation. This is a
bounded recovery diagnostic, not a continuation or replacement full mission.
"""
import argparse,gzip,json,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import full_learning_monitoring as model
import cluster_environment as ce
from cached_poam_fields import CachedFields
from exact_poam_variance import ResolvedVarianceBasis
from reachable_learning import ReachableLearning
from geometric_task_mask import CandidatePathAcquisition
from profile_io import load_profile
from guarded_waypoints import holding_force_bound

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=Path('output/path_mask_20261105');source=root/'calibration180';out=args.output;out.mkdir(parents=True,exist_ok=False)
with gzip.open(source/'s2-center-pathmask.json.gz','rt') as f:record=json.load(f)
gp=load_profile(source/'s2-center-pathmask-final-profile.pkl');cfg=model.settings();pcfg=model.legacy.physical_settings(cfg)
env,grid,spec=ce.environment(pcfg,model.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
basis,points,shape=model.table_grid(env.bounds,.1);basis=ResolvedVarianceBasis(basis.lower,basis.spacing)
cache=CachedFields(points,shape,cfg,grid['all_cells'],exact_variance=True)
reacher=ReachableLearning(env,grid,cfg,basis,'mask',global_goals=True,prefix_screen='braking-feasible',diversify_stalls=True)
acquisition=CandidatePathAcquisition(env,grid,cfg)
x=jnp.asarray(record['samples'][-1][1:5]);q=jnp.asarray(record['clarity_frames'][-1]['q']);initial=np.asarray(x).copy()
clock=record['summary']['simulated_seconds'];policy=reacher.policy
policy.retreat=np.vstack((np.asarray(x[:2]),record['plans'][-1]['target']))
policy.recovery_anchor=np.asarray(x[:2]);policy.recovery_since=clock-15
# Recover the actual measured history, rather than restricting recovery to
# a single final target. Re-evaluate its refuge status using the saved GP.
positions=np.asarray(record['observations'])[:,1:3]
history_mu,history_cov=gp.belief.predict(jnp.asarray(positions))
policy.history=[(p,holding_force_bound(mu,float(np.sqrt(cov[0,0]))))
    for p,mu,cov in zip(positions,np.asarray(history_mu),np.asarray(history_cov))]
seed=8101;noise=np.random.default_rng(seed).normal(size=(5001,2))
tables=cache.build(gp.belief,jnp.array(len(gp.positions)),jax.random.PRNGKey(seed+1000),jax.random.PRNGKey(seed+271828))
truth=model.m.physical.load_wind(2,model.m.physical.SITE/'revised')
execute=model.execution_block(env,cfg,pcfg,basis,seconds=.2);alive=jnp.array(True)
samples=[];decisions=[];infeasible=0;contact=False;began=time.perf_counter()
for i in range(150):
    b=gp.belief
    if i%10==0:reacher.task_weights=acquisition.task_weights(x,q)[0]
    mean,cov=b.predict(x[None,:2]);local_std=float(jnp.sqrt(cov[0,0,0]))
    path,decision,_,_=reacher.plan(np.asarray(x),np.asarray(q),b,cache,tables,clock,np.asarray(mean[0]),local_std)
    ref=policy.reference(np.asarray(x),path);refs=jnp.tile(jnp.asarray(ref),(cfg.steps+1,1));aa=jnp.zeros((cfg.steps,2))
    end,rows=execute(x,q,refs,aa,jnp.array(0.),tables,truth,alive);jax.block_until_ready(end)
    xs,qs,us,nom,slack,correction,feasible,gaps,costs,living,hits=map(np.asarray,rows)
    infeasible+=int((~feasible).sum());samples.extend([[clock+(k+1)*.05,*z.tolist(),float(gaps[k])] for k,z in enumerate(xs)])
    decisions.append(dict(t=clock,path=path.tolist(),**decision));x,q,alive=end;clock=round(clock+.2,8)
    if not bool(alive):contact=True;break
    measured=np.asarray(truth.at(x[:2]))[0]+.1*noise[round(clock/.2)]
    gp.update(x[None,:2],jnp.asarray(measured)[None],train_kernel=(len(gp.positions)+1)%50==0)
    tables=cache.safety(gp.belief,tables,jax.random.PRNGKey(seed+271828));jax.block_until_ready(tables)
result=dict(seconds=(i+1)*.2,wall_seconds=time.perf_counter()-began,initial=initial.tolist(),final=np.asarray(x).tolist(),
    displacement_m=float(np.linalg.norm(np.asarray(x[:2])-initial[:2])),contact=contact,infeasible_commands=infeasible,
    regained_control=policy.retreat is None,directed_decisions=sum(d.get('recovery_graph',False) for d in decisions),
    minimum_clearance_m=min(s[-1] for s in samples),samples=samples,decisions=decisions)
(out/'result.json').write_text(json.dumps(result,allow_nan=False))
print(json.dumps({k:v for k,v in result.items() if k not in ('samples','decisions')}),flush=True)
