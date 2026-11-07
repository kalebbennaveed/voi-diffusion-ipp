"""Calibrate a candidate-path stopping threshold from an actual flown survey.

No observations are invented, and no truth is provided to MBD/RPCBF. This is
an offline, simulation-only stopping diagnostic, not independent evidence of
subsequent mission performance. A fresh mission must recheck its handoff.
"""
import argparse,gzip,json,pickle,time
from pathlib import Path
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import full_learning_monitoring as model
import cluster_environment as ce
from cached_poam_fields import CachedFields
from exact_poam_variance import ResolvedVarianceBasis
from geometric_task_mask import CandidatePathAcquisition
from handoff_validation import HandoffValidator
from profile_io import load_profile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--scenarios',default='1,2')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-validation-slack',action='store_true')
    parser.add_argument('--handoff-metric',choices=['position-and-cost','monitoring-cost'],default='position-and-cost')
    args=parser.parse_args();reports=[]
    args.output.mkdir(parents=True,exist_ok=False)
    for scenario in map(int,args.scenarios.split(',')):
        ident=f's{scenario}-center-pathmask'
        with gzip.open(args.source/f'{ident}.json.gz','rt') as f:r=json.load(f)
        protocol=json.loads((args.source/f'{ident}-protocol.json').read_text())
        if not r['summary']['completed'] or r['summary']['contact']:
            reports.append(dict(scenario=scenario,passed=False,reason='Survey did not finish without contact'))
            continue
        gp=load_profile(args.source/f'{ident}-final-profile.pkl')
        cfg=type(model.settings())(**protocol['settings'])
        pcfg=replace(model.legacy.physical_settings(cfg),mission_seconds=r['summary']['simulated_seconds'],
            safety_margin=cfg.safety_margin,inner_samples=cfg.construction_worlds)
        env,grid,spec=ce.environment(pcfg,model.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
        basis,points,shape=model.table_grid(env.bounds,protocol['cache_spacing_m'])
        if protocol['exact_safety_variance']:basis=ResolvedVarianceBasis(basis.lower,basis.spacing)
        cache=CachedFields(points,shape,cfg,grid['all_cells'],exact_variance=protocol['exact_safety_variance'])
        seed=protocol['initial_condition']['seed'];b=gp.belief
        x=jnp.asarray(r['samples'][-1][1:5]);q=jnp.asarray(r['clarity_frames'][-1]['q'])
        tables=cache.build(b,jnp.array(len(gp.positions)),jax.random.PRNGKey(seed+1000),jax.random.PRNGKey(seed+271828))
        truth=model.m.physical.load_wind(scenario,model.m.physical.SITE/'revised')
        mu,cov=cache.map(b)
        domain,metadata=CandidatePathAcquisition(env,grid,cfg,protocol['candidate_path_mask']['radius_m']).task_weights(x,q)
        precision=float(np.sqrt(2*np.dot(domain,np.asarray(cov)[:,0,0])))
        error=np.asarray(mu)-np.asarray(truth.at(jnp.asarray(grid['all_cells'])))
        task_rmse=float(np.sqrt(np.dot(domain,np.sum(error*error,axis=1))))
        check=HandoffValidator(env,cfg,pcfg,basis,spec,allow_validation_slack=args.allow_validation_slack,metric=args.handoff_metric).check(x,q,b,cache,tables,truth,seed,details=True)
        report=dict(scenario=scenario,survey_seconds=r['summary']['simulated_seconds'],
            actual_measurements=len(gp.positions),task_vector_std_m_s=precision,task_rmse_m_s=task_rmse,
            active_cells=metadata['active_cells'],free_cells=metadata['free_cells'],
            selected_threshold_m_s=float(np.ceil(precision*100)/100) if check['passed'] else None,
            **check)
        reports.append(report);print(json.dumps(report),flush=True)
        (args.output/f's{scenario}-handoff-calibration.json').write_text(json.dumps(report,indent=2))
    (args.output/'handoff-calibration.json').write_text(json.dumps(dict(reports=reports,
        selection='Round a passing actual-survey candidate-domain posterior vector RMS upward to 0.01 m/s; require fresh forward validation at online crossing.',
        limitation='Finite three-state forecast calibration, not a global accuracy or future safety guarantee. Does not use wind truth for acquisition, MBD or RPCBF.'),indent=2))


if __name__=='__main__':main()
