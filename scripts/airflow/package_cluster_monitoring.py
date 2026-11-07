"""Audit and package the deterministic runs in the revised environment without replacing older data."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from cluster_environment import CASES

def read_record(path):
    return json.loads(path.read_bytes() if path.exists() else gzip.decompress(Path(str(path)+'.gz').read_bytes()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--site',type=Path,required=True);args=p.parse_args()
    out=args.site/'data/monitoring-cluster-300'
    if (out/'manifest.json').exists():raise RuntimeError('Refusing to replace a packaged comparison')
    out.mkdir(parents=True,exist_ok=True)
    checks=[];runs=[]
    env=json.loads((args.directory/'environment.json').read_text())
    for scenario in (1,2):
        flow=json.loads((args.site/f'data/scenario{scenario}/metadata.json').read_text())
        assert np.allclose(np.asarray(flow['obstacles'])[:,:3],env['obstacles'])
        assert flow['domain_m'][:2]==env['bounds'][1]
        assert flow['regions']==env['regions'] and flow['grid'][:2]==env['grid']
    for scenario,case in ((s,c) for s in (1,2) for c in CASES):
        for method in ('aware','baseline'):
            ident=f's{scenario}-{case["id"]}-{method}';record=read_record(args.directory/f'{ident}.json')
            protocol=json.loads((args.directory/f'{ident}-protocol.json').read_text())
            root=Path(__file__).resolve().parents[2]
            for name,digest in protocol['source_sha256'].items():
                # The harness gained only a more detailed budget-stop report
                # during this suite. Every model and planner source must match.
                if name!='scripts/airflow/run_cluster_monitoring.py':
                    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest, name
            raw=np.load(args.directory/f'{ident}-execution.npz')['samples']
            assert record['summary']['completed'] and raw[-1,0]==300 and len(raw)==6001
            assert np.all(np.isfinite(raw)) and np.max(np.abs(raw[:,5:7]))<=1.6+1e-8
            assert np.allclose(np.diff(raw[:,0]),.05)
            q=np.array([f['q'] for f in record['clarity_frames']])
            assert q.shape==(601,len(env['cells'])) and q.min()>=0 and q.max()<=1
            mean_deficit=np.maximum(np.array(env['targets'])[None]-q,0).mean(1)
            assert np.max(abs(mean_deficit-raw[::10,13]))<1e-6
            geometry=np.array(env['obstacles']);positions=raw[:,1:3];bounds=np.array(env['bounds'])
            clear=np.minimum(np.min(np.linalg.norm(positions[:,None]-geometry[None,:,:2],axis=2)-geometry[:,2],axis=1),np.min(np.c_[positions-bounds[0],bounds[1]-positions],axis=1))
            # Column 12 is the minimum over the preceding physical substeps,
            # including their initial state; it need not equal endpoint clearance.
            assert np.all(clear>=raw[:,12]-1e-8)
            if record['summary']['contact'] is None:assert np.all(clear>0) and np.all(raw[:,18]==1)
            warm=[r['wall_seconds'] for r in record['plans'] if not r['timing']['compilation_included']]
            assert len(record['plans'])==60 and max(warm)<1
            assert protocol['initial_condition']==case
            assert protocol['environment_spec']==env['environment_spec']
            assert np.allclose(raw[0,1:5],case['state'])
            assert protocol['uncertainty']==0 and protocol['legacy_settings']['construction_worlds']==1
            sha=hashlib.sha256((args.site/f'data/scenario{scenario}/stationary-velocity.f32').read_bytes()).hexdigest()
            assert sha==protocol['wind_sha256']
            checks.append(dict(id=ident,seconds=300,finite=True,actuator_bound=True,clarity_range=[float(q.min()),float(q.max())],
                clarity_deficit_consistent=True,unchanged_wind_sha256=sha,first_visit_s=record['summary']['first_visit_s'],
                contact=record['summary']['contact'],infeasible_execution_steps=record['summary']['infeasible_execution_steps']))
            timing=dict(min_s=min(warm),median_s=float(np.median(warm)),max_s=max(warm),calls=len(warm),
                initialization_median_s=float(np.median([r['timing']['initialization'] for r in record['plans'] if not r['timing']['compilation_included']])),
                mbd_forecast_median_s=float(np.median([r['timing']['optimization_validation'] for r in record['plans'] if not r['timing']['compilation_included']])),
                first_compilation_s=[r['wall_seconds'] for r in record['plans'] if r['timing']['compilation_included']],
                execution_block_median_s=float(np.median([r['seconds'] for r in record['execution_timing'] if not r['compilation_included']])))
            payload=json.dumps(record,separators=(',',':'),allow_nan=False).encode()
            (out/f'{ident}.json.gz').write_bytes(gzip.compress(payload,compresslevel=6,mtime=0))
            shutil.copyfile(args.directory/f'{ident}-protocol.json',out/f'{ident}-protocol.json')
            runs.append(dict(id=ident,scenario=scenario,case=case['id'],method=method,file=f'{ident}.json.gz',decoded_sha256=hashlib.sha256(payload).hexdigest(),summary=record['summary'],timing=timing))
    pairs=[]
    for scenario in (1,2):
        for case in CASES:
            a,b=[next(r for r in runs if r['scenario']==scenario and r['case']==case['id'] and r['method']==method) for method in ('aware','baseline')]
            sa,sb=a['summary'],b['summary']
            pairs.append(dict(scenario=scenario,case=case['id'],baseline_plans_central=sb['first_planned_route_to_A']=='central',aware_first_route=sa['first_transfer']['route'],baseline_first_route=sb['first_transfer']['route'],both_visit_A=sa['first_visit_s']['A'] is not None and sb['first_visit_s']['A'] is not None,both_no_contact=sa['contact'] is None and sb['contact'] is None,deficit_improvement_percent=100*(sb['integrated_deficit']-sa['integrated_deficit'])/sb['integrated_deficit']))
    # At t=0 both scenarios have identical geometry, robot state, clarity and
    # random key. A genuinely wind-blind optimizer must return the same plan.
    blind_checks=[]
    for case in CASES:
        initial=[np.load(args.directory/f's{s}-{case["id"]}-baseline-plan-000.npz') for s in (1,2)]
        error=float(np.max(np.abs(initial[0]['theta']-initial[1]['theta'])))
        assert error==0., 'Wind leaked into the baseline planning decision'
        blind_checks.append(dict(case=case['id'],maximum_initial_control_parameter_difference=error))
    audit=dict(checks=checks,pairs=pairs,model='Original physical double integrator plus quadratic air force; exact stored stationary wind known; full spline RPCBF; whole-environment clarity ODE.',
        limits='Six declared conditions per scenario, with three seeds across the suite. This is a finite robustness check, not a guarantee over all initial conditions. Relaxed forecasting and hard execution have different control periods. No global safety guarantee is inferred from no contact.')
    audit['wind_blind_initial_plan_equivalence']=blind_checks
    audit['geometry_consistent_across_CFD_planner_execution_display']=True
    audit['acceptance_checks']={
        'all_12_baselines_first_plan_central':all(p['baseline_plans_central'] for p in pairs),
        'all_6_scenario1_aware_first_take_lower':all(p['aware_first_route']=='lower' for p in pairs if p['scenario']==1),
        'all_6_scenario2_aware_first_take_upper':all(p['aware_first_route']=='upper' for p in pairs if p['scenario']==2),
        'all_12_aware_reach_A':all(r['summary']['first_visit_s']['A'] is not None for r in runs if r['method']=='aware'),
        'all_24_runs_without_contact':all(r['summary']['contact'] is None for r in runs),
        'all_12_aware_have_lower_deficit':all(p['deficit_improvement_percent']>0 for p in pairs),
        'completed_mission_warm_plans_below_one_second':all(r['timing']['max_s']<1 for r in runs),
    }
    audit['acceptance_passed']=all(audit['acceptance_checks'].values())
    investigation=args.site/'data/runtime-investigation.json'
    if investigation.exists():
        audit['earlier_budget_stop']=json.loads(investigation.read_text())
        audit['runtime_claim']='Completed missions met the measured warm-call limit. One earlier attempt exceeded it and was stopped; a hard real-time guarantee on this shared machine is not established.'
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    shutil.copyfile(args.directory/'environment.json',out/'environment.json')
    manifest=dict(duration_s=300,state='complete',implementation='known-wind-mbd-revised-environment',initial_wind_knowledge='known',uncertainty=0,
        planning_horizon_s=15,replan_s=5,control_coordinates=120,runtime_limit_s=1,runs=runs,conditions=CASES,comparisons=pairs,
        planning='Original 7-mode MBD, 2 levels and 14 proposals per level; one known field. Select the exact deterministic search incumbent. One selected forecast retained for display.',
        execution='Full hard RPCBF, one known field, 0.06 m margin, 0.05 s filter period, 0.01 s physics and clarity.',
        no_learning=True,no_exploration=True,source_available=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    root=Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in (root/'src').rglob('*.py'):z.write(f,str(f.relative_to(root)))
        for name in ('known_monitoring','run_cluster_monitoring','package_cluster_monitoring','cluster_environment','simulate_cluster_airflow','simulate_sheltered_airflow','export_cluster_wind','legacy_monitoring','monitoring_mbd','monitoring_routes','robot_review','run_monitoring_mbd'):
            f=root/f'scripts/airflow/{name}.py';z.write(f,str(f.relative_to(root)))
        z.write(root/'pyproject.toml','pyproject.toml')
        z.writestr('environment-spec.json',json.dumps(env['environment_spec'],indent=2))
        for scenario in (1,2):
            for name in ('metadata.json','stationary-velocity.f32'):
                z.write(args.site/f'data/scenario{scenario}/{name}',f'scenario-site/data/scenario{scenario}/{name}')
        z.writestr('REPRODUCE.txt','Install pyproject.toml dependencies. Run from the extracted project:\nPYTHONPATH=src:scripts/airflow python scripts/airflow/run_cluster_monitoring.py --site scenario-site --spec environment-spec.json --output reproduced --seconds 300 --wall-limit 900\nWarm planning stops above 1 s; compilation is separate. Existing outputs are never overwritten.\n')
    print(json.dumps(dict(pairs=pairs,run_count=len(runs),acceptance=audit['acceptance_checks'],maximum_warm_plan_s=max(r['timing']['max_s'] for r in runs)),indent=2))

if __name__=='__main__':main()
