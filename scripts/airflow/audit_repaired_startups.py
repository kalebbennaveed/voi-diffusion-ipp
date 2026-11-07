"""First filtered command at all six declared starts, not a mission certificate."""
import json,pickle,signal,time
import numpy as np
import jax
import jax.numpy as jnp
import rmse_monitoring as im
import cluster_environment as ce
from spatial_poam import SpatialPOAM
from waypoint_poam import condition
from uncertainty_waypoints import WaypointPolicy

ROOT=im.m.physical.ROOT;OUT=ROOT/'output/gp_safety_20261104'

def main():
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('90s startup check limit')));signal.alarm(90)
    cfg=im.settings();env,grid,_=ce.environment(im.legacy.physical_settings(cfg),ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
    basis,points,shape=im.table_grid(env.bounds,.1);build=im.field_builder(points,shape,cfg)
    with (OUT/'sampling/s1-dense-profile.pkl').open('rb') as f:example=pickle.load(f)
    initial=SpatialPOAM(env.cells,env.bounds,prior_support=example.belief.basis.support).belief
    pred=jax.jit(lambda b:b.predict(jnp.asarray(grid['all_cells'])))
    @jax.jit
    def command(x,ref,tables):
        mean=im.TableMean(basis,tables.mean);fields=im.InnerFields(basis,tables.safety)
        nominal=im.m.tracking_control(x,ref,jnp.zeros(2),mean,cfg,True)
        return im.control(x,nominal,mean,fields,tables.std,basis,env,cfg,True)
    results=[]
    for scenario in (1,2):
        truth=im.m.physical.load_wind(scenario,im.m.physical.SITE/'revised')
        with (OUT/f'sampling/s{scenario}-dense-profile.pkl').open('rb') as f:learned=pickle.load(f).belief
        for label,prior in [('no survey',initial),('full-coverage learned GP',learned)]:
            for case in ce.CASES:
                x,q=ce.initial(case,env,grid,im.legacy.physical_settings(cfg))
                y=truth.at(x[:2])+jnp.asarray(.1*np.random.default_rng(case['seed']).normal(size=(1,2)))
                b,_=condition(prior,x[None,:2],y,jnp.zeros(len(prior.mean),bool))
                tables=build(b,jnp.array(1),jax.random.PRNGKey(case['seed']+1000),jax.random.PRNGKey(case['seed']+271828))
                mu,cov=pred(b);policy=WaypointPolicy(env);path,decision=policy.plan(np.asarray(x),np.asarray(cov)[:,0,0],0,np.asarray(mu))
                ref=jnp.asarray(policy.reference(x,path));u,info=command(x,ref,tables);jax.block_until_ready(u)
                assert np.isfinite(u).all() and np.max(np.abs(u))<=1.6+1e-6
                if bool(info.feasible):assert float(info.slack)<1e-6
                row=dict(scenario=scenario,case=case['id'],belief=label,state=np.asarray(x).tolist(),control=np.asarray(u).tolist(),
                    feasible=bool(info.feasible),slack=float(info.slack),residual=float(info.residual))
                results.append(row);print(json.dumps(row),flush=True)
    (OUT/'startup-checks.json').write_text(json.dumps(dict(scope='Only the first filtered waypoint command; does not establish trajectory safety.',checks=results),indent=2))
    signal.alarm(0)

if __name__=='__main__':main()
