"""Pre-observation POAM calibration from actual recorded readings only."""
import gzip,json,signal,time
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import implicit_monitoring_v5 as im
import cluster_environment as ce

def main():
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60 second innovation audit limit')));signal.alarm(60)
    root=im.m.physical.ROOT;source=root/'results/airflow/implicit_v5_20261104/confirmed300'
    cfg=im.settings();pcfg=replace(im.legacy.physical_settings(cfg),safety_margin=cfg.safety_margin)
    env,_,_=ce.environment(pcfg,root/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
    _,points,_=im.table_grid(env.bounds);initial=im.initial_belief(env,points)
    @jax.jit
    def innovation(b,x,y):
        mean,cov=b.predict(x);residual=y[0]-mean[0]
        return residual,jnp.sum(residual**2)/(cov[0,0,0]+b.noise_std**2),jnp.sqrt(cov[0,0,0]+b.noise_std**2)
    rows=[];start=time.perf_counter()
    for scenario in [1,2]:
        ident=f's{scenario}-center-aware';r=json.load(gzip.open(source/(ident+'.json.gz')))
        b=initial;mask=jnp.zeros(128,bool);audit=[]
        for o in r['observations']:
            x=jnp.asarray([o[1:3]]);y=jnp.asarray([o[3:5]])
            e,z,s=innovation(b,x,y)
            audit.append(dict(t=o[0],innovation=np.asarray(e).tolist(),mahalanobis_squared=float(z),predictive_std=float(s)))
            b,mask=im.condition_supported(b,x,y,mask)
        z=np.asarray([a['mahalanobis_squared'] for a in audit])
        rows.append(dict(id=ident,observations=len(z),nominal_3sigma_disk_coverage=float(1-np.exp(-9/2)),
            measured_3sigma_disk_coverage=float(np.mean(z<=9)),outside_3sigma_count=int(np.sum(z>9)),
            nominal_99percent_threshold=float(-2*np.log(.01)),measured_99percent_coverage=float(np.mean(z<=-2*np.log(.01))),
            maximum_mahalanobis_squared=float(z.max()),innovations=audit))
    report=dict(scope='Every prediction precedes its actual measurement update. No true field or synthetic observations used. Gaussian coverage is nominal under a correctly specified model, not a safety guarantee.',runs=rows,seconds=time.perf_counter()-start)
    target=root/'output/diagnostics/implicit_v2_20261104/innovations-v5.json';target.write_text(json.dumps(report,indent=2));signal.alarm(0)
    for r in rows:print(json.dumps({k:v for k,v in r.items() if k!='innovations'}))

if __name__=='__main__':main()
