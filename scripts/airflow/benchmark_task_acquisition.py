"""Bounded, isolated task-acquisition benchmark on the real POAM dimensions."""
import argparse,json,pickle,time,signal
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import full_learning_monitoring as im
import cluster_environment as ce
from cached_poam_fields import CachedFields
from task_wind_acquisition import TaskWindAcquisition

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(exist_ok=True,parents=True)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120 second benchmark guard')));signal.alarm(120)
    root=im.m.physical.ROOT;cfg=im.settings();pcfg=im.legacy.physical_settings(cfg)
    env,grid,_=ce.environment(pcfg,root/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
    _,points,shape=im.table_grid(env.bounds,.1);cache=CachedFields(points,shape,cfg,grid['all_cells'])
    reports=[]
    for scenario in (1,2):
        with (root/f'output/gp_safety_20261104/sampling/s{scenario}-dense-profile.pkl').open('rb') as f:gp=pickle.load(f)
        b=gp.belief;mu,cov=cache.map(b);jax.block_until_ready(mu);mu=np.asarray(mu);var=np.asarray(cov)[:,0,0]
        for case in ce.CASES[:3]:
            x,q=ce.initial(case,env,grid,pcfg)
            for kind in ['mask','correlated','routes']:
                acquisition=TaskWindAcquisition(env,grid,cfg,kind)
                for repeat in range(3):
                    before=np.asarray(b.mean).copy();start=time.perf_counter()
                    priority,weights,info=acquisition.compute(np.asarray(x),np.asarray(q),b,cache,mu,var)
                    elapsed=time.perf_counter()-start
                    assert np.isfinite(priority).all() and np.min(priority)>=0
                    assert np.isclose(weights.sum(),1) and np.all(np.isfinite(weights))
                    assert np.array_equal(before,np.asarray(b.mean))
                    reports.append(dict(scenario=scenario,case=case['id'],kind=kind,repeat=repeat,
                        compilation_included=scenario==1 and case['id']=='center' and kind=='correlated' and repeat==0,
                        seconds=elapsed,task_vector_std_m_s=info['weighted_vector_std_m_s']))
    a.output.write_text(json.dumps(reports,indent=2))
    for kind in ['mask','correlated','routes']:
        times=[r['seconds'] for r in reports if r['kind']==kind and not r['compilation_included']]
        print(json.dumps(dict(method=kind,warm_median_s=float(np.median(times)),warm_max_s=max(times))),flush=True)
    signal.alarm(0)

if __name__=='__main__':main()
