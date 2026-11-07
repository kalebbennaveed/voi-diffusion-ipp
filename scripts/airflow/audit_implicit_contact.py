"""Retrospective contact audit. True wind is compared AFTER controller evaluation.

No truth is passed to POAM, the controller, or the confidence calculation.
Reconstructs the actual 0.2-second posterior updates and 0.05-second commands.
"""
import gzip,json,signal,time
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import implicit_monitoring_v5 as im
import cluster_environment as ce


def main():
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('90 second audit limit')))
    signal.alarm(90)
    root=im.m.physical.ROOT
    source=root/'results/airflow/implicit_v5_20261104/missions300'
    rec=json.load(gzip.open(source/'s2-center-aware.json.gz'))
    cfg=im.settings();pcfg=replace(im.legacy.physical_settings(cfg),safety_margin=cfg.safety_margin)
    env,grid,spec=ce.environment(pcfg,root/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
    basis,points,shape=im.table_grid(env.bounds)
    b=im.initial_belief(env,points);mask=jnp.zeros(128,bool)
    sampler=im.field_builder(points,shape,cfg)
    safe_sampler=im.safety_builder(points,shape,cfg)
    obs=np.asarray(rec['observations']);samples=np.asarray(rec['samples']);cursor=0
    skey=jax.random.PRNGKey(8101+271828)
    @jax.jit
    def evaluate(x,nominal,tables,b):
        mean=im.TableMean(basis,tables.mean);fields=im.InnerFields(basis,tables.safety)
        grad,B,bound=im.constraints(x,mean,fields,tables.std,basis,env,cfg)
        u,info=im.control(x,nominal,mean,fields,tables.std,basis,env,cfg,True)
        mu=mean.mean_at(x[:2])[0]
        sigma=jnp.maximum(basis.evaluate(x[:2],tables.std)[0,0],0.)
        direct,cov=b.predict(x[None,:2])
        return u,info,grad,B,bound,mu,3*jnp.sqrt(sigma**2+.1**2),direct[0],jnp.sqrt(cov[0,0,0])
    # The hidden field is used only for the diagnostic comparisons below.
    truth=im.m.physical.load_wind(2,im.m.physical.SITE/'revised')
    rows=[];start=time.perf_counter();tables=None
    for t in np.arange(284.,289.025,.05):
        sample=samples[round(t/.05)];nxt=samples[round(t/.05)+1]
        update_t=round(np.floor((t+1e-7)/.2)*.2,8)
        changed=False
        while cursor<len(obs) and obs[cursor,0]<=update_t+1e-7:
            o=obs[cursor];b,mask=im.condition_supported(b,jnp.asarray(o[None,1:3]),jnp.asarray(o[None,3:5]),mask)
            cursor+=1;changed=True
        old_tables=tables
        if tables is None:
            tables=sampler(b,jnp.asarray(cursor),jax.random.PRNGKey(8101+1000+int(t//5)),skey)
        elif changed:tables=safe_sampler(b,tables,skey)
        result=evaluate(jnp.asarray(sample[1:5]),jnp.asarray(nxt[7:9]),tables,b)
        u,info,grad,B,bound,mu,R,direct,sigma=map(lambda x:jax.tree.map(np.asarray,x),result)
        previous_B=None
        if changed and old_tables is not None:
            # Same position and velocity, before/after the actual posterior
            # update: isolate the safety-set jump from physical motion.
            old_value=im.values(jnp.asarray(sample[1:5]),im.TableMean(basis,old_tables.mean),
                im.InnerFields(basis,old_tables.safety),env,cfg)
            previous_B=float(jnp.max(old_value))
        wind=np.asarray(truth.at(sample[1:3]))[0]
        acceleration=np.asarray(im.m.physical.acceleration(sample[3:5],wind,xp=np))
        actual_derivative=grad[:,:2]@sample[3:5]+grad[:,2:]@(u+acceleration)
        rows.append(dict(t=round(float(t),8),position=sample[1:3].tolist(),velocity=sample[3:5].tolist(),
            control=u.tolist(),recorded_control=nxt[5:7].tolist(),command_error=float(np.max(np.abs(u-nxt[5:7]))),
            clearance=float(sample[12]),feasible=bool(info.feasible),slack=float(info.slack),
            B=B.tolist(),max_B=float(B.max()),before_update_max_B_same_state=previous_B,
            estimated_mean=mu.tolist(),direct_gp_mean=direct.tolist(),
            direct_gp_std=float(sigma),radius=float(R),true_wind=wind.tolist(),
            truth_outside_disk=bool(np.linalg.norm(wind-mu)>R),distance_from_mean=float(np.linalg.norm(wind-mu)),
            true_acceleration=acceleration.tolist(),max_actual_cbf_residual=float(np.max(actual_derivative+cfg.cbf_alpha*B)),
            update_time=update_t,observations=cursor))
    out=dict(run='s2-center-aware v5',scope='retrospective audit only; no true wind in inference/control',
        rows=rows,seconds=time.perf_counter()-start,max_command_error=max(r['command_error'] for r in rows),
        first_truth_outside_disk=next((r['t'] for r in rows if r['truth_outside_disk']),None),
        first_positive_B=next((r['t'] for r in rows if r['max_B']>0),None),
        first_infeasible=next((r['t'] for r in rows if not r['feasible']),None))
    target=root/'output/diagnostics/implicit_v2_20261104/contact-v5.json'
    target.write_text(json.dumps(out,indent=2));signal.alarm(0)
    print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
    for r in rows[::8]:print(json.dumps(r))


if __name__=='__main__':main()
