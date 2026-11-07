"""Replay physical execution, selected forecasts and the original MBD updates."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import legacy_monitoring as l
import monitoring_mbd as m
from run_monitoring_mbd import write
from voi_diffusion_ipp.quarter.diffusion import cosine_schedule
from voi_diffusion_ipp.quarter.multimode import mode_score_and_update

def audit(directory,site=m.physical.SITE):
    checks=[]
    for scenario in (1,2):
        for method in ('aware','baseline'):
            ident=f's{scenario}-{method}';record=json.loads((directory/f'{ident}.json').read_text());protocol=json.loads((directory/f'{ident}-protocol.json').read_text())
            cfg=l.Settings(**protocol['legacy_settings']);physical=l.physical_settings(cfg);env,_=m.environment(physical);truth=m.physical.load_wind(scenario,site)
            assert record['summary']['completed'];assert protocol['no_anticipated_updates'] and protocol['no_optimistic_objective']
            assert cfg.optimism==1 and cfg.mixture==0 and cfg.guidance_mode=='off'
            checked_sources=0
            for relative,digest in protocol['source_sha256'].items():
                paths=[m.physical.ROOT/relative,*directory.glob('source_snapshot*/'+relative)]
                assert any(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==digest for p in paths),(ident,relative)
                checked_sources+=1
            for history in protocol.get('source_history',[]):
                for relative,digest in history.items():
                    paths=[m.physical.ROOT/relative,*directory.glob('source_snapshot*/'+relative)]
                    assert any(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==digest for p in paths),(ident,'source history',relative)
            raw=np.load(directory/f'{ident}-execution.npz');a=raw['samples'];states=raw['states'];q=raw['clarity']
            assert np.all(np.isfinite(a)) and np.all(np.isfinite(q)) and q.min()>=0 and q.max()<=1
            replay=jax.jit(jax.vmap(lambda x,q,u,alive:m.advance(x,q,u,truth,env,physical,alive,interval=.05)))
            # Keep replay memory bounded for longer missions; every recorded step is checked.
            errors=dict(state=0.,clarity=0.,cost=0.);cost_increments=[]
            for start in range(0,len(states)-1,1200):
                stop=min(start+1200,len(states)-1)
                end=replay(jnp.asarray(states[start:stop]),jnp.asarray(q[start:stop]),jnp.asarray(a[start+1:stop+1,5:7]),jnp.asarray(a[start:stop,18]>0))
                errors['state']=max(errors['state'],float(np.max(np.abs(np.asarray(end[0])-states[start+1:stop+1]))))
                errors['clarity']=max(errors['clarity'],float(np.max(np.abs(np.asarray(end[1])-q[start+1:stop+1]))))
                cost_increments.append(np.asarray(end[3]))
            errors['cost']=float(np.max(np.abs(np.cumsum(np.concatenate(cost_increments))-a[1:,14])))
            assert max(errors.values())<1e-8,(ident,errors)
            assert np.max(np.abs(a[:,5:7]))<=physical.umax+1e-9
            warm=[p['wall_seconds'] for p in record['plans'] if not p['timing']['compilation_included']];assert max(warm)<=protocol['runtime_limit_s']
            equation_error=0.;forecast_error=0.;env32=l.float32(env)
            for i,plan in enumerate(record['plans']):
                data=np.load(directory/f'{ident}-plan-{i:03d}.npz')
                with jax.experimental.disable_x64():
                    alpha,abar=cosine_schedule(cfg.iterations,cfg.cosine_end)
                    for level in range(cfg.iterations):
                        j=cfg.iterations-1-level
                        # rho=0 and no likelihood guidance: plain expected-cost MBD.
                        costs=jnp.asarray(data['search_sample_costs'][level]);logs=-costs/cfg.temperature+jnp.asarray(data['search_log_importance_ratios'][level])
                        updated,score,weights,ok=mode_score_and_update(jnp.asarray(data['search_before'][level]),jnp.asarray(data['search_samples'][level]),logs,jnp.asarray(data['search_proposal_modes']),alpha[j],abar[j])
                        valid=np.asarray(ok)
                        if valid.any():equation_error=max(equation_error,float(np.max(np.abs(np.asarray(updated)[valid]-data['search_after'][level][valid]))))
                        np.testing.assert_allclose(weights,data['search_weights'][level],atol=3e-5,rtol=3e-5)
                if i in (0,len(record['plans'])-1):
                    th=l.float32(jnp.asarray(data['theta']));x0=l.float32(jnp.asarray(data['initial_state']));q0=l.float32(jnp.asarray(data['initial_clarity']));mean=jnp.asarray(data['mean']);inner=jnp.asarray(data['construction']);worlds=jnp.asarray(data['validation'])
                    if method=='baseline':mean=jnp.zeros_like(mean);inner=jnp.zeros_like(inner);worlds=jnp.zeros_like(worlds)
                    with jax.experimental.disable_x64():
                        predict=jax.jit(lambda th,x,q,mean,inner,worlds:jax.vmap(lambda f:l.rollout(th,x,q,mean,inner,f,env32,cfg,method=='aware'))(worlds))
                        result=predict(th,x0,q0,mean,inner,worlds)
                    forecast_error=max(forecast_error,float(np.max(np.abs(np.asarray(result.x)-np.asarray(plan['predicted_states'])))))
                    np.testing.assert_allclose(float(jnp.mean(result.monitor+result.penalty)),plan['expected_cost'],atol=2e-5,rtol=2e-5)
            assert equation_error<1e-4 and forecast_error<1e-5
            prefix_file=directory/('resume-prefix-'+ident+'-execution.npz')
            if prefix_file.exists():
                prefix=np.load(prefix_file)
                for key in ('samples','states','clarity','observation_rows'):np.testing.assert_array_equal(prefix[key],raw[key][:len(prefix[key])])
            for rejected_index,rejected in enumerate(protocol.get('rejected_planning_calls',[]),1):
                rejected_data=np.load(directory/f'{ident}-rejected-{rejected_index:03d}.npz')
                repeated=np.load(directory/f'{ident}-plan-{round(rejected["t"]/cfg.commit):03d}.npz')
                for key in ('theta','initial_state','initial_clarity','mean','construction','worlds','validation'):
                    np.testing.assert_array_equal(rejected_data[key],repeated[key])
            checks.append(dict(id=ident,seconds=float(a[-1,0]),source_files_verified=checked_sources,execution_replay_errors=errors,max_reverse_equation_error=equation_error,max_forecast_state_error=forecast_error,warm_plan_max_s=max(warm),rejected_planning_calls=protocol.get('rejected_planning_calls',[]),resume_prefix_verified=prefix_file.exists(),summary=record['summary']))
            print(json.dumps({k:v for k,v in checks[-1].items() if k!='summary'}),flush=True)
    result=dict(checks=checks,scope='Complete held-control robot/clarity/cost replay, full saved MBD score/reverse audit, first and last selected forecast replay, source hashes and all recorded warm-plan budgets. No claim of global safety or optimizer convergence.')
    write(directory/'audit.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--site',type=Path,default=m.physical.SITE);a=p.parse_args();audit(a.directory,a.site)
