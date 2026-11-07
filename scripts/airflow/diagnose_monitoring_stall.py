"""Bounded saved-state diagnostics; no optimization or mission changes."""
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
from voi_diffusion_ipp.safety.flow_policy_value import filter_control

def main():
    root=Path('results/airflow/monitoring_survey_safe_300'); rows=[]
    for scenario in (1,2):
        protocol=json.loads((root/f's{scenario}-aware-protocol.json').read_text())
        cfg=l.Settings(**protocol['legacy_settings']);pcfg=replace(l.physical_settings(cfg),safety_margin=.12)
        env,_=m.environment(pcfg);truth=m.physical.load_wind(scenario)
        raw=np.load(root/f's{scenario}-aware-plan-012.npz')
        template=l.matched_prior(env)
        b=jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template),[jnp.asarray(raw[k]) for k in sorted(raw.files) if k.startswith('belief_leaf_')])
        x=jnp.asarray(raw['initial_state']);refs,aa=l.old.nominal_reference(jnp.asarray(raw['theta']),x,cfg)
        # A reference two seconds into the already selected plan shows what
        # the tracking controller requests after the robot has remained still.
        ref=refs[8];a=aa[8]
        @jax.jit
        def draw(bb):
            c=bb.sample_field(jax.random.PRNGKey(protocol['seed']+271828),8)
            return bb.mean_at(l.POINTS).reshape(l.SHAPE),jax.vmap(lambda z:bb.basis.evaluate(l.POINTS,z))(c).reshape((-1,*l.SHAPE))
        with jax.experimental.disable_x64():mean,draws=draw(l.float32(b));jax.block_until_ready(draws)
        true_array=truth.at(l.POINTS).reshape(l.SHAPE)
        variants=[('forecast_soft_2',raw['mean'],raw['construction'],cfg,False),
                  ('execution_hard_8',mean,draws,pcfg,True),
                  ('posterior_mean_only_hard',mean,jnp.tile(mean[None],(8,1,1,1)),pcfg,True),
                  ('known_wind_hard',true_array,jnp.tile(true_array[None],(8,1,1,1)),pcfg,True)]
        for name,mu,fields,c,hard in variants:
            started=time.perf_counter();belief=TableMean(l.BASIS,jnp.asarray(mu,jnp.float64));inner=InnerFields(l.BASIS,jnp.asarray(fields,jnp.float64))
            nominal=m.tracking_control(x,ref,a,belief,pcfg,True)
            if hard:
                run=jax.jit(lambda state,nom,bel,ff:filter_control(state,nom,bel,ff,env,c))
                u,info=run(x,nominal,belief,inner);jax.block_until_ready(u)
                slack=float(info.slack);feasible=bool(info.feasible)
            else:
                run=jax.jit(lambda state,nom,bel,ff:l.old.control(state,nom,bel,ff,env,c))
                u,ss=run(x,nominal,belief,inner);jax.block_until_ready(u);slack=float(ss);feasible=None
            air=m.physical.acceleration(np.asarray(x[2:]),np.asarray(truth.at(x[:2]))[0])
            row=dict(scenario=scenario,variant=name,state=np.asarray(x).tolist(),reference=np.asarray(ref).tolist(),nominal=np.asarray(nominal).tolist(),u=np.asarray(u).tolist(),actual_acceleration=(np.asarray(u)+air).tolist(),slack=slack,feasible=feasible,wall_s=time.perf_counter()-started)
            rows.append(row);print(json.dumps(row),flush=True)
    Path('results/airflow/monitoring_stall_diagnostic.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
