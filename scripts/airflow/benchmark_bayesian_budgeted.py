"""Bounded task-directed Bayesian look-ahead timing gate."""
import json,time,argparse
from pathlib import Path
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import legacy_monitoring as l
import monitoring_mbd as m
from exploration_monitoring import BayesianSelector,diverse_prefixes,forecast

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 directory=Path('results/airflow/monitoring_nosurvey_safe_300');protocol=json.loads((directory/'s1-aware-protocol.json').read_text())
 cfg=replace(l.Settings(**protocol['legacy_settings']),worlds=1,validation_worlds=2,validation_batch_size=4)
 env,_=m.environment(l.physical_settings(cfg));env=l.float32(env);raw=np.load(directory/'s1-aware-plan-000.npz');template=l.matched_prior(env)
 b=jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template),[jnp.asarray(raw[k]) for k in sorted(raw.files) if k.startswith('belief_leaf_')]);b,x,q=l.float32((b,raw['initial_state'],raw['initial_clarity']))
 report=dict(horizon_s=cfg.horizon,decision_s=cfg.commit,root_candidates=3,observation_branches=1,conditional_worlds=1,continuation_choices=4,calls=[])
 with jax.experimental.disable_x64():
  optimizer=l.optimizer(env,cfg,True);sampler=l.table_sampler(cfg)
  selector=BayesianSelector(env,replace(cfg,safety_margin=.12,construction_worlds=8,prediction_filter='hard'),roots=3,branches=1,conditional_worlds=1)
  for i in range(3):
   start=time.perf_counter();seeds,previous,meta=l.seed_bank(x,q,env,cfg);mean,fields=sampler(b,jax.random.PRNGKey(9101),jax.random.PRNGKey(279929));jax.block_until_ready(fields)
   result=optimizer(x,q,mean,fields[:2],fields[2:3],fields[3:],previous,jax.random.PRNGKey(10101),jnp.array(1.),jnp.array(False),seeds);jax.block_until_ready(result);original=time.perf_counter()-start
   candidates,indices=diverse_prefixes(result[0],seeds,x,cfg,count=3);decision=selector.select(b,x,q,candidates,jax.random.PRNGKey(12001),jax.random.PRNGKey(279929));total=time.perf_counter()-start
   row=dict(compilation_included=i==0,original_mbd_s=original,lookahead=decision['timing'],total_s=total,selected=decision['index'],scores=decision['scores'].tolist(),prefix_contacts=decision['prefix_contacts'].tolist());report['calls'].append(row);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
   if i and total>1:break
 report['budget_passed']=all(r['total_s']<=1 for r in report['calls'][1:]);a.output.write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
