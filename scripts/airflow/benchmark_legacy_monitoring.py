"""Bounded old-pipeline timing before any mission is allowed to run."""
import argparse,json,time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import legacy_monitoring as legacy
import monitoring_mbd as m

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--proposals',type=int,default=28);p.add_argument('--worlds',type=int,default=4)
    p.add_argument('--validation',type=int,default=8);p.add_argument('--horizon',type=float,default=15)
    p.add_argument('--levels',type=int,default=4);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--float32',action='store_true')
    p.add_argument('--validation-batch',type=int,default=0)
    p.add_argument('--belief',type=Path,help='Optional saved full belief fixture; same reached state and clarity')
    a=p.parse_args()
    if a.float32:jax.config.update('jax_enable_x64',False)
    cfg=replace(legacy.Settings(),proposals=a.proposals,batch_size=a.proposals,worlds=a.worlds,validation_worlds=a.validation,horizon=a.horizon,iterations=a.levels,validation_batch_size=a.validation_batch)
    cfg.__post_init__();physical=legacy.physical_settings(cfg);env,grid=m.environment(physical)
    poam=m.estimator(physical,jax.random.PRNGKey(8202))
    raw=np.load('results/airflow/monitoring_coarse_review/s2-aware-plan-005.npz')
    structure=jax.tree_util.tree_structure(poam.planning_belief())
    belief_raw=np.load(a.belief) if a.belief else raw
    belief=jax.tree_util.tree_unflatten(structure,[jnp.asarray(belief_raw[k]) for k in sorted(belief_raw.files) if k.startswith('belief_leaf_')])
    x=jnp.asarray(raw['initial_state']);q=jnp.asarray(raw['initial_clarity'])
    sample=legacy.table_sampler(cfg);plan=legacy.optimizer(env,cfg,True)
    report={'config':asdict(cfg),'state':'saved S2 at 25 s','convention':'Synchronized end-to-end seed generation + current-belief field sampling + old MBD + fresh-field final validation. POAM construction reported separately. Compilation measured separately.','calls':[]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for i in range(a.repeats+1):
        started=time.perf_counter();seeds,previous,meta=legacy.seed_bank(x,q,env,cfg)
        seed_s=time.perf_counter()-started;start=time.perf_counter()
        mean,fields=sample(belief,jax.random.PRNGKey(4001),jax.random.PRNGKey(4002));jax.block_until_ready(fields)
        sampling_s=time.perf_counter()-start;start=time.perf_counter()
        result=plan(x,q,mean,fields[:cfg.construction_worlds],fields[cfg.construction_worlds:cfg.construction_worlds+cfg.worlds],fields[-cfg.validation_worlds:],previous,jax.random.PRNGKey(4003),jnp.array(1.),jnp.array(False),seeds)
        jax.block_until_ready(result);optimizer_s=time.perf_counter()-start
        row=dict(compilation_included=i==0,seed_s=seed_s,field_sampling_s=sampling_s,optimizer_validation_s=optimizer_s,total_s=time.perf_counter()-started,cost=float(result[1]),evaluations=int(result[3].evaluations),starts=len(seeds))
        report['calls'].append(row);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
        if i>0 and row['total_s']>5:break
    report['warm_max_s']=max(r['total_s'] for r in report['calls'][1:])
    report['budget_passed']=report['warm_max_s']<=1.
    a.output.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
