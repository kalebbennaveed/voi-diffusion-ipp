"""Bounded, synchronized full planning timings; no missions in this script."""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import legacy_monitoring as l
import monitoring_mbd as m
import active_sensing_monitoring as active

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('active','inducing','passive','thompson'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--worlds',type=int,default=2);p.add_argument('--validation',type=int,default=4);a=p.parse_args()
    directory=Path('results/airflow/monitoring_nosurvey_safe_300');protocol=json.loads((directory/'s1-aware-protocol.json').read_text())
    cfg=replace(l.Settings(**protocol['legacy_settings']),safety_margin=.12,construction_worlds=8,worlds=a.worlds,validation_worlds=a.validation)
    if a.mode=='thompson':cfg=replace(cfg,worlds=1,validation_worlds=1)
    env,_=m.environment(l.physical_settings(cfg));env=l.float32(env)
    raw=np.load(directory/'s1-aware-plan-000.npz');template=l.matched_prior(env)
    b=jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template),[jnp.asarray(raw[k]) for k in sorted(raw.files) if k.startswith('belief_leaf_')])
    b,x,q=l.float32((b,raw['initial_state'],raw['initial_clarity']));report=dict(mode=a.mode,horizon_s=cfg.horizon,search_worlds=cfg.worlds,validation_worlds=cfg.validation_worlds,calls=[])
    with jax.experimental.disable_x64():
        optimizer=active.optimizer(env,cfg,a.mode);sampler=active.field_sampler(cfg)
        for i in range(3):
            start=time.perf_counter();seeds,previous,metadata=l.seed_bank(x,q,env,cfg);geometry=time.perf_counter()-start;t=time.perf_counter()
            mean,construction,worlds,validation=sampler(b,jax.random.PRNGKey(9101),jax.random.PRNGKey(279929));jax.block_until_ready(validation);sampling=time.perf_counter()-t;t=time.perf_counter()
            if a.mode=='thompson':validation=worlds
            result=optimizer(b,x,q,mean,construction,worlds,validation,previous,jax.random.PRNGKey(10101),seeds);jax.block_until_ready(result)
            row=dict(compilation_included=i==0,geometry_s=geometry,field_sampling_s=sampling,optimization_validation_s=time.perf_counter()-t,total_s=time.perf_counter()-start,score=float(result.score),expected=float(result.expected),information=float(result.information[result.index]),budget=float(result.budget))
            report['calls'].append(row);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
            if i and row['total_s']>1:break
    report['budget_passed']=all(r['total_s']<=1 and np.isfinite(r['score']) for r in report['calls'][1:]);a.output.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
