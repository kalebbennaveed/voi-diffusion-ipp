"""Offline estimator/actuator diagnosis. Truth is used only in this audit."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from voi_diffusion_ipp.wind.poam import attentive_kernel
from package_repaired_validation import wind


def diagnose(path):
    r=json.load(gzip.open(path));s=np.asarray(r['samples']);obs=np.asarray(r['observations']);field=wind(r['scenario'])
    env=json.loads((path.parent/'environment.json').read_text());cells=np.asarray(env['cells'])
    final=np.load(path.with_name(path.name.replace('.json.gz','-posterior.npz')))
    params={k[6:]:jnp.asarray(final[k]) for k in final.files if k.startswith('param_')}
    centers=jnp.asarray(final['centers']);inv=jnp.asarray(final['inverse']);mean=jnp.asarray(final['mean']);cov=jnp.asarray(final['covariance'])
    lengths=jnp.asarray(final['lengths']);alpha=inv@mean;correction=inv@cov@inv.T-inv
    @jax.jit
    def moments(x):
        K=attentive_kernel(params,x,centers,lengths)
        return K@alpha,jnp.maximum(jnp.exp(params['log_amplitude'])+jnp.sum((K@correction)*K,axis=1),0)
    def score(points):
        mu,var=map(np.asarray,moments(jnp.asarray(points)));actual=np.array([field(z) for z in points]);error=np.linalg.norm(mu-actual,axis=1)
        return dict(count=len(points),vector_rmse_m_s=float(np.sqrt(np.mean(error**2))),
            safety_three_sigma_coverage=float(np.mean(error<=3*np.sqrt(var+.01))),max_vector_error_m_s=float(error.max()),
            predicted_vector_rms_std_m_s=float(np.sqrt(2*var.mean())))
    selected=obs[::max(1,len(obs)//300),1:3]
    failed=np.flatnonzero(s[:,11]==0);first=float(s[failed[0],0]) if len(failed) else None
    approach=[]
    if first is not None:
        for t in np.arange(max(0,first-2),min(s[-1,0],first+1)+.01,.2):
            row=s[np.argmin(abs(s[:,0]-t))];w=field(row[1:3]);relative=w-row[3:5];force=.1/.680*np.linalg.norm(relative)*relative
            approach.append(dict(t=float(row[0]),position=row[1:3].tolist(),wind=w.tolist(),
                wind_acceleration=force.tolist(),within_component_authority=bool(np.abs(force).max()<=1.6),
                qp_feasible=bool(row[11]),clearance=float(row[12])))
    predictions=r.get('observation_predictions',[])
    return dict(id=r['id'],summary=r['summary'],prior_sigma_m_s=1.8,
        fitted_amplitude_std_m_s=float(np.sqrt(np.exp(final['param_log_amplitude']))),
        final_fit_at_executed_measurement_locations=score(selected),final_whole_map=score(cells),
        premeasurement_innovation_three_sigma_coverage=float(np.mean([z['standardized_innovation']<=3 for z in predictions])) if predictions else None,
        first_infeasible_s=first,approach_to_infeasibility=approach)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('record',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=diagnose(a.record);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
