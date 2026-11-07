"""Recover POAM's complete optimizer/RNG state from actual recorded measurements.

No robot is rerun, and no true field is queried. Refuse recovery unless the
posterior agrees with the separately saved original full-precision arrays.
"""
import argparse,gzip,json,time,signal
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import full_learning_monitoring as model
import cluster_environment as ce
from spatial_poam import SpatialPOAM,SpatialPOAMSettings
from profile_io import save_profile,load_profile

def main():
    p=argparse.ArgumentParser();p.add_argument('record',type=Path);p.add_argument('--wall-limit',type=int,default=1800);a=p.parse_args()
    with gzip.open(a.record,'rt') as f:r=json.load(f)
    ident=r['id'];destination=a.record.with_name(ident+'-final-profile.pkl.gz')
    if destination.exists():raise RuntimeError('Existing inference checkpoint preserved')
    proto=json.loads(a.record.with_name(ident+'-protocol.json').read_text())
    if r['summary']['contact'] is not None or proto['initial_measurements']!=0:raise ValueError('Expected contact-free cold-start trial')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Inference recovery budget exhausted')))
    signal.alarm(a.wall_limit);began=time.perf_counter()
    cfg=model.settings();pcfg=model.legacy.physical_settings(cfg)
    env,_,_=ce.environment(pcfg,proto['environment_spec'])
    _,extended,_=model.table_grid(env.bounds,.4)
    outside=np.any((np.asarray(extended)<np.asarray(env.bounds[0]))|(np.asarray(extended)>np.asarray(env.bounds[1])),axis=1)
    settings=dict(proto['gp_profile']);settings['lengths_m']=tuple(settings['lengths_m'])
    partial=a.record.with_name(ident+'-inference-replay.pkl.gz')
    gp=load_profile(partial) if partial.exists() else SpatialPOAM(env.cells,env.bounds,settings=SpatialPOAMSettings(**settings),prior_support=np.vstack((np.asarray(env.cells),np.asarray(extended)[outside])),
        training_seed=proto['initial_condition']['seed']+441,factorized_support=proto['factorized_support'])
    restored=len(gp.positions)
    np.testing.assert_array_equal(gp.positions,np.asarray(r['observations'])[:restored,1:3])
    np.testing.assert_array_equal(gp.measurements,np.asarray(r['observations'])[:restored,3:5])
    for k,(row,update) in enumerate(zip(r['observations'],r['updates'],strict=True)):
        if k<restored:continue
        gp.update(jnp.asarray([row[1:3]]),jnp.asarray([row[3:5]]),train_kernel=update['training']);jax.block_until_ready(gp.belief)
        if (k+1)%200==0:save_profile(partial,gp)
        if k%100==0:print(json.dumps(dict(observations=k+1,total=len(r['observations']),wall_s=time.perf_counter()-began)),flush=True)
    expected=np.load(a.record.with_name(ident+'-posterior.npz'));errors={}
    pairs=[('mean',gp.belief.mean),('covariance',gp.belief.covariance),('inverse',gp.belief.inverse),('centers',gp.belief.basis.centers)]
    pairs.extend(('param_'+key,value) for key,value in gp.belief.basis.parameters.items())
    for key,value in pairs:
        errors[key]=float(np.max(np.abs(np.asarray(value)-expected[key])))
        np.testing.assert_allclose(value,expected[key],atol=1e-10,rtol=1e-10)
    save_profile(destination,gp)
    audit=dict(source=str(a.record),actual_measurements=len(gp.positions),posterior_errors=errors,wall_s=time.perf_counter()-began,verified=True)
    a.record.with_name(ident+'-profile-recovery-audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True);signal.alarm(0)

if __name__=='__main__':main()
