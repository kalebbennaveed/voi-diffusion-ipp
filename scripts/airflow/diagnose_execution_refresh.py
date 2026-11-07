"""Bounded counterfactual replay of saved collision approaches; no MBD tuning."""
import json,time
from pathlib import Path
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import legacy_monitoring as l
import monitoring_mbd as m
from voi_diffusion_ipp.wind.cache import TableMean
from voi_diffusion_ipp.wind.gp import InnerFields

def main():
    directory=Path('results/airflow/monitoring_legacy_300');rows=[]
    for scenario,start in [(1,80),(2,265)]:
        protocol=json.loads((directory/f's{scenario}-baseline-protocol.json').read_text())
        cfg=l.Settings(**protocol['legacy_settings']);physical=l.physical_settings(cfg);env,_=m.environment(physical)
        truth=m.physical.load_wind(scenario);template=m.estimator(physical,jax.random.PRNGKey(8202)).planning_belief()
        initial=np.load(directory/f's{scenario}-baseline-plan-{start//5:03d}.npz')
        for label,margin,count,refresh in [('original',.06,2,False),('fresh_measurements',.06,2,True),('fresh_margin_samples',.12,8,True)]:
            begun=time.monotonic();pcfg=replace(physical,safety_margin=margin);execute=m.execution_kernel(env,pcfg)
            x=jnp.asarray(initial['initial_state']);q=jnp.asarray(initial['initial_clarity']);alive=True;hit=None;minimum=float(env.clearance(x));bad=0
            belief=jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template),[jnp.asarray(initial[k]) for k in sorted(initial.files) if k.startswith('belief_leaf_')])
            condition=jax.jit(lambda b,p,y:b.condition(p,y,relocate=True))
            @jax.jit
            def fields(b):
                c=b.sample_field(jax.random.PRNGKey(8111),count)
                return b.mean_at(l.POINTS).reshape(l.SHAPE),jax.vmap(lambda a:b.basis.evaluate(l.POINTS,a))(c).reshape((-1,*l.SHAPE))
            for step in range(200):
                t=start+step*.05;cycle=int(t//5);local=t-5*cycle
                if step%100==0:
                    raw=np.load(directory/f's{scenario}-baseline-plan-{cycle:03d}.npz')
                    refs,aa=l.old.nominal_reference(jnp.asarray(raw['theta'],jnp.float64),jnp.asarray(raw['initial_state']),cfg)
                    if not refresh:mean=jnp.asarray(raw['mean'],jnp.float64);draws=jnp.asarray(raw['construction'],jnp.float64)
                if refresh and step%4==0:
                    y=np.asarray(truth.at(x[:2]))[0]+.1*np.random.default_rng(8101+round(t*5)).normal(size=2)
                    position32=np.asarray(x)[None,:2].astype(np.float32)
                    with jax.experimental.disable_x64():
                        belief=condition(l.float32(belief),jnp.asarray(position32),jnp.asarray(y[None],jnp.float32))
                        mean,draws=fields(belief)
                    mean=mean.astype(jnp.float64);draws=draws.astype(jnp.float64)
                live=TableMean(l.BASIS,mean);inner=InnerFields(l.BASIS,draws)
                i=min(int(round(local/cfg.dt,8)),cfg.steps-1);off=local-i*cfg.dt;a=aa[i];r=refs[i];reference=jnp.r_[r[:2]+off*r[2:]+.5*off*off*a,r[2:]+off*a]
                nominal=m.tracking_control(x,reference,a,live,pcfg,True)
                end,u,info=execute(x,q,nominal,live,inner,truth,jnp.asarray(alive));x,q,survived,_,gap,contact=end
                minimum=min(minimum,float(gap));bad+=not bool(info.feasible)
                if not bool(survived):hit=t+float(contact);break
            row=dict(scenario=scenario,variant=label,start_s=start,contact_s=hit,min_clearance_m=minimum,infeasible_steps=bad,final_state=np.asarray(x).tolist(),wall_s=time.monotonic()-begun)
            rows.append(row);print(json.dumps(row),flush=True)
    Path('results/airflow/execution_refresh_diagnostic.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
