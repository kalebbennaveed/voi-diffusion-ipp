"""Independent larger-bank ranking check of one recorded MBD selection.

This does not change the mission or select replacement trajectories. It compares
exactly the recorded finalists, controller construction fields and initial state.
"""
import argparse,json,time
from pathlib import Path
import jax,jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr
from voi_diffusion_ipp.quarter.continuous import ControlSettings,rollout,aggregate_costs
from voi_diffusion_ipp.quarter.environment import initial_belief,environment,table_sampler
from voi_diffusion_ipp.quarter.diffusion import lower_tail


def main():
 p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--fields',type=int,default=128);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 d=json.loads(args.record.read_text());cfg=ControlSettings(**d['config']);plan=d['plans'][0];env=environment(d['layout'],cfg.environment_scale,cfg.inspection_scale,d['case'])
 with np.load(args.record.parent/plan['audit']) as f:a={k:f[k] for k in f.files}
 theta=jnp.array(a['finalists']);selected=int(a['chosen_index']);b=initial_belief(cfg)
 # This audit is intentionally the pre-observation planning call: no replayed
 # posterior or unknown assimilation history is introduced.
 assert plan['t']==0 and len(a['validation_fields'])==cfg.validation_worlds
 _,fields=table_sampler(args.fields,cfg.environment_scale)(b,jax.random.PRNGKey(196233))
 @jax.jit
 def components(theta,fields):
  def one_theta(th):
   def one_field(field):
    r=rollout(th,jnp.array(a['initial_state']),jnp.array(a['clarity']),jnp.array(a['mean']),jnp.array(a['construction']),field,env,cfg,True)
    return jnp.array([r.monitor,r.penalty])
   return jax.vmap(one_field)(fields)
  return jax.lax.map(one_theta,theta,batch_size=1)
 start=time.perf_counter();small=np.asarray(components(theta,jnp.array(a['validation_fields'])));print('recomputed saved selection bank',time.perf_counter()-start,flush=True)
 def scores(values):return np.asarray(jax.vmap(lambda v:lower_tail(aggregate_costs(v[:,0],v[:,1],cfg.penalty_aggregation),cfg.optimism))(jnp.array(values)))
 small_scores=scores(small)
 np.testing.assert_allclose(small_scores[selected],plan['predicted_optimistic'],atol=.003,rtol=1e-4)
 large=np.asarray(components(theta,fields));print('independent bank complete',time.perf_counter()-start,flush=True);large_scores=scores(large);best=int(np.argmin(large_scores))
 ranks=np.argsort(np.argsort(large_scores,kind='stable'),kind='stable');ties=np.abs(large_scores-large_scores[selected])<1e-4
 block_winners=[]
 for start in range(0,args.fields,8):
  block=scores(large[:,start:start+8]);win=int(np.argmin(block));block_winners.append(dict(index=win,selected=win==selected,large_bank_cost=float(large_scores[win])))
 out=dict(record=str(args.record),t=0,nominal_sequences=len(theta),saved_validation_fields=cfg.validation_worlds,independent_fields=args.fields,selected_index=selected,
  saved_selected_score=float(small_scores[selected]),larger_bank_selected_score=float(large_scores[selected]),larger_bank_best_index=best,larger_bank_best_score=float(large_scores[best]),
  selected_rank_one_based=int(1+np.count_nonzero(large_scores<large_scores[selected]-1e-4)),tied_sequences=int(ties.sum()),rank_correlation=float(spearmanr(small_scores,large_scores).statistic),
  selected_regret=float(large_scores[selected]-large_scores[best]),independent_eight_field_blocks=block_winners,small_scores=small_scores.tolist(),larger_scores=large_scores.tolist(),
  note='One pre-observation planning call, not a mission performance or generalization test. Same recorded construction fields and finalists. Independent prior fields; hidden physical wind unused. Larger bank is a numerical diagnostic, not ground truth.')
 args.output.write_text(json.dumps(out,indent=2));np.savez_compressed(args.output.with_suffix('.npz'),monitor=large[:,:,0],penalty=large[:,:,1],theta=np.asarray(theta));print(json.dumps({k:v for k,v in out.items() if k not in ('small_scores','larger_scores','independent_eight_field_blocks')}),flush=True)

if __name__=='__main__':main()
