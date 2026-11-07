"""Replay exported controls and clarity, and audit paired monitoring inputs."""
import argparse,json,hashlib
from dataclasses import replace
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import monitoring_mbd as m
from voi_diffusion_ipp.wind.gp import InnerFields,WorldField


def audit_selected_plans(directory,ident,record,cfg,env):
    """Recompute first/last selected forecasts from saved full posterior leaves."""
    aware=record['method']=='aware'
    template=m.estimator(cfg,jax.random.PRNGKey(0)).planning_belief()
    structure=jax.tree_util.tree_structure(template)
    if aware:
        evaluate=jax.jit(lambda theta,x,q,b,inner,outer:jax.vmap(
            lambda coeff:m.forecast(theta,x,q,b,inner,WorldField(outer.basis,coeff),env,cfg,True)
        )(outer.coefficients))
    else:
        evaluate=jax.jit(lambda theta,x,q,b,inner,outer:m.forecast(theta,x,q,None,None,m.StillAir(),env,cfg,False))
    checks=[]
    selected=sorted(set([0,len(record['plans'])-1]))
    for index in selected:
        plan=record['plans'][index]
        raw=np.load(directory/f'{ident}-plan-{index:03d}.npz')
        leaves=[jnp.asarray(raw[k]) for k in sorted(raw.files) if k.startswith('belief_leaf_')]
        belief=jax.tree_util.tree_unflatten(structure,leaves)
        inner=InnerFields(belief.basis,jnp.asarray(raw['inner_coefficients']))
        outer=InnerFields(belief.basis,jnp.asarray(raw['outer_coefficients']))
        result=evaluate(jnp.asarray(raw['theta']),jnp.asarray(raw['initial_state']),
                        jnp.asarray(raw['initial_clarity']),belief,inner,outer)
        xs=np.asarray(result.states);xs=xs if aware else xs[None]
        state_error=float(np.max(np.abs(xs-np.asarray(plan['predicted_states']))))
        assert state_error<=5.1e-7
        cost_error=None
        if plan['valid']:
            if cfg.reject_collisions:assert not np.any(np.asarray(result.collided))
            cost_error=abs(float(np.mean(np.asarray(result.cost)))-plan['expected_cost'])
            assert cost_error<1e-9
        checks.append(dict(cycle=index,time_s=plan['t'],valid=plan['valid'],
                           max_exported_forecast_state_error=state_error,expected_cost_error=cost_error))
    return checks



def audit_diffusion_history(directory,ident,index,cfg,seed):
    """Independently replay equations 6/8/9 and PRNG proposals, every chain."""
    from voi_diffusion_ipp.planning.mbd import schedule
    raw=np.load(directory/f'{ident}-plan-{index:03d}.npz');starts=raw['initials'];modes=len(starts)
    if not cfg.multimode:return None
    alpha,ab=map(np.asarray,schedule(cfg.diffusion_steps,cfg.beta_min,cfg.beta_max,cfg.diffusion_reference_steps))
    histories={k[8:]:raw[k] for k in raw.files if k.startswith('history_')}
    keys=jax.random.split(jax.random.PRNGKey(seed+2000+index),modes)
    largest=0.;valid=0;total=0
    for mode,key in enumerate(keys):
        key,initial_key=jax.random.split(key)
        y=np.sqrt(ab[-1])*starts[mode]+np.sqrt(1-ab[-1])*np.asarray(jax.random.normal(initial_key,starts[mode].shape))
        for level,i in enumerate(range(cfg.diffusion_steps-1,-1,-1)):
            before=histories['noisy_before'][mode,level];np.testing.assert_allclose(before,y,atol=2e-11)
            key,k=jax.random.split(key)
            candidates=before/np.sqrt(ab[i])+np.sqrt((1-ab[i])/ab[i])*np.asarray(jax.random.normal(k,(cfg.proposals//modes,starts.shape[1])))
            np.testing.assert_allclose(candidates,histories['candidates'][mode,level],atol=2e-11)
            costs=histories['costs'][mode,level];finite=np.isfinite(costs)
            total+=len(costs);valid+=int(finite.sum())
            assert np.all(~finite[np.any(np.abs(candidates)>cfg.mbd_latent_bound,axis=1)])
            weights=np.zeros_like(costs)
            if finite.any():weights[finite]=np.exp(-(costs[finite]-costs[finite].min())/cfg.temperature);weights/=weights.sum()
            np.testing.assert_allclose(weights,histories['weights'][mode,level],atol=2e-11)
            mean=weights@candidates if finite.any() else before/np.sqrt(ab[i])
            score=(-before+np.sqrt(ab[i])*mean)/(1-ab[i])
            np.testing.assert_allclose(score,histories['score'][mode,level],atol=2e-9)
            after=(before+(1-ab[i])*score)/np.sqrt(alpha[i]) if finite.any() else before
            error=float(np.max(np.abs(after-histories['noisy_after'][mode,level])));largest=max(largest,error)
            assert error<2e-10;y=histories['noisy_after'][mode,level]
    return dict(cycle=index,modes=modes,proposals=total,finite_proposals=valid,max_reverse_equation_error=largest)


def audit_executed_filter(directory,ident,record,raw,cfg,env):
    """Recompute tracking and full RPCBF at representative executed states.

    Include each replan start, its smallest-clearance step, and largest-slack
    step. Stored applied controls are never used as inputs to this check.
    """
    template=m.estimator(cfg,jax.random.PRNGKey(0)).planning_belief()
    structure=jax.tree_util.tree_structure(template)
    apply=jax.jit(lambda x,nom,b,inner:m.filter_control(x,nom,b,inner,env,cfg))
    rows=raw['samples'];states=raw['states'];checks=[]
    for cycle,plan in enumerate(record['plans']):
        data=np.load(directory/f'{ident}-plan-{cycle:03d}.npz')
        belief=jax.tree_util.tree_unflatten(structure,[jnp.asarray(data[k]) for k in sorted(data.files) if k.startswith('belief_leaf_')])
        inner=InnerFields(belief.basis,jnp.asarray(data['inner_coefficients']))
        refs,aa=m.nominal_reference(jnp.asarray(data['theta']),jnp.asarray(data['initial_state']),cfg)
        candidates=np.flatnonzero((rows[:-1,0]>=plan['t']-1e-8)&(rows[:-1,0]<plan['t']+cfg.commit_seconds-1e-8)&(rows[:-1,18]>0))
        if not len(candidates):continue
        selected=set([int(candidates[0]),int(candidates[np.argmin(rows[candidates+1,12])]),int(candidates[np.argmax(rows[candidates+1,9])])])
        for i in sorted(selected):
            local=round(float(rows[i,0]-plan['t']),8);index=min(int(round(local/cfg.dt,8)),cfg.steps-1);offset=local-index*cfg.dt
            ref=refs[index];a=aa[index]
            interpolated=jnp.r_[ref[:2]+offset*ref[2:]+.5*offset**2*a,ref[2:]+offset*a]
            nominal=m.tracking_control(jnp.asarray(states[i]),interpolated,a,belief,cfg,True) if cfg.reference_tracking else a
            u,info=apply(jnp.asarray(states[i]),nominal,belief,inner)
            nom_error=float(np.max(np.abs(np.asarray(nominal)-rows[i+1,7:9])))
            control_error=float(np.max(np.abs(np.asarray(u)-rows[i+1,5:7])))
            slack_error=abs(float(info.slack)-rows[i+1,9])
            assert nom_error<1e-8 and control_error<1e-7 and slack_error<1e-7
            checks.append(dict(time_s=float(rows[i,0]),max_nominal_error=nom_error,max_control_error=control_error,slack_error=slack_error))
    return checks


def audit(directory,site,only=None):
    rows=[]
    for scenario in (1,2):
        protocols=[]
        for method in ('aware','baseline'):
            ident=f's{scenario}-{method}'
            if only and ident not in only:continue
            record=json.loads((directory/(ident+'.json')).read_text())
            protocol=json.loads((directory/(ident+'-protocol.json')).read_text());protocols.append(protocol)
            cfg=m.Settings(**protocol['settings']);env,grid=m.environment(cfg)
            truth=m.physical.load_wind(scenario,site)
            raw=np.load(directory/(ident+'-execution.npz'))
            a=raw['samples'];x=raw['states'];q=raw['clarity'];obs=raw['observation_rows']
            assert record['summary']['completed'] and a[-1,0]==cfg.mission_seconds
            assert np.all(np.isfinite(a)) and np.all(np.isfinite(q))
            assert np.max(np.abs(a[:,5:7]))<=cfg.umax+1e-10
            assert q.min()>=0 and q.max()<=1 and np.all(np.diff(a[:,14])>=-1e-10)
            assert protocol['wind_sha256']==hashlib.sha256((site/f'data/scenario{scenario}/stationary-velocity.f32').read_bytes()).hexdigest()
            replay=jax.jit(jax.vmap(lambda state,clarity,u,alive:m.advance(state,clarity,u,truth,env,cfg,alive,interval=cfg.execution_dt)))
            end=replay(jnp.asarray(x[:-1]),jnp.asarray(q[:-1]),jnp.asarray(a[1:,5:7]),jnp.asarray(a[:-1,18]>0))
            # The saved row records the command applied during the preceding
            # interval. All intervals, including post-contact decay, are tested.
            state_error=float(np.max(np.abs(np.asarray(end[0])-x[1:])))
            clarity_error=float(np.max(np.abs(np.asarray(end[1])-q[1:])))
            cost_error=float(np.max(np.abs(np.cumsum(np.asarray(end[3]))-a[1:,14])))
            assert state_error<1e-10 and clarity_error<1e-10 and cost_error<1e-9
            fine_cfg=replace(cfg,physics_dt=.005)
            fine=jax.jit(jax.vmap(lambda state,clarity,u,alive:m.advance(state,clarity,u,truth,env,fine_cfg,alive,interval=cfg.execution_dt)))
            refined=fine(jnp.asarray(x[:-1]),jnp.asarray(q[:-1]),jnp.asarray(a[1:,5:7]),jnp.asarray(a[:-1,18]>0))
            # Local held-control numerical sensitivity, not a reoptimized run.
            local_position_difference=float(np.max(np.linalg.norm(np.asarray(refined[0])[:,:2]-x[1:,:2],axis=1)))
            local_clarity_difference=float(np.max(np.abs(np.asarray(refined[1])-q[1:])))
            targets=np.asarray(env.targets)
            deficit=np.maximum(targets-q,0).mean(axis=1)
            np.testing.assert_allclose(deficit,a[:,13],atol=1e-12)
            for i in range(2):np.testing.assert_allclose(q[:,grid['membership'][:,i]].mean(axis=1),a[:,15+i],atol=1e-12)
            noise=np.random.default_rng(protocol['seed']).normal(size=(round(cfg.mission_seconds/cfg.dt)+1,2))
            expected=np.asarray(truth.at(obs[:,1:3]))+cfg.observation_std*noise[np.rint(obs[:,0]/cfg.dt).astype(int)] if len(obs) else np.zeros((0,2))
            np.testing.assert_allclose(obs[:,3:5],expected,atol=1e-12)
            diffusion_checks=[]
            for index,plan in enumerate(record['plans']):
                finite_seeds=[v for v in plan['seed_costs'] if v is not None and np.isfinite(v)]
                if finite_seeds:assert plan['expected_cost']<=min(finite_seeds)+1e-9
                if cfg.multimode:
                    modes=len(np.load(directory/f'{ident}-plan-{index:03d}.npz')['initials'])
                    assert plan['evaluations']==cfg.diffusion_steps*cfg.proposals+3*modes
                    diffusion_checks.append(audit_diffusion_history(directory,ident,index,cfg,protocol['seed']))
                else:assert plan['evaluations']==cfg.diffusion_steps*cfg.proposals+2+10
                assert len(plan['predicted_states'])==(cfg.worlds if method=='aware' else 1)
            forecast_checks=audit_selected_plans(directory,ident,record,cfg,env)
            control_checks=audit_executed_filter(directory,ident,record,raw,cfg,env)
            rows.append(dict(id=ident,seconds=float(a[-1,0]),max_state_replay_error=state_error,
                max_clarity_replay_error=clarity_error,max_cost_replay_error=cost_error,
                max_local_position_refinement_m=local_position_difference,max_local_clarity_refinement=local_clarity_difference,
                observations=len(obs),plans=len(record['plans']),selected_forecast_replays=forecast_checks,executed_filter_replays=control_checks,diffusion_equation_audits=diffusion_checks,summary=record['summary']))
        if len(protocols)==2:
            assert protocols[0]['settings']==protocols[1]['settings']
            assert protocols[0]['wind_sha256']==protocols[1]['wind_sha256']
            assert protocols[0]['seed']==protocols[1]['seed']
    report=dict(checks='All exported intervals replayed from stored controls; independent paired-input, cost, region clarity and measurement checks.',
        refinement_scope='Local held-control .01 to .005 s integration check, starting at each saved state. This is not a reoptimized or re-filtered mission.',runs=rows)
    (directory/('audit-partial.json' if only else 'audit.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--site',type=Path,default=m.physical.SITE)
    p.add_argument('--only',nargs='+')
    args=p.parse_args();audit(args.directory,args.site,args.only)
