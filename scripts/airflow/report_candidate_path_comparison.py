"""Summarize all fresh matched runs without selecting away failures."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np


def distribution(values):
    values=np.asarray(values,float)
    return None if len(values)==0 else dict(count=len(values),median_s=float(np.median(values)),
        p95_s=float(np.quantile(values,.95)),max_s=float(values.max()))


def read_run(path):
    with gzip.open(path,'rt') as stream:r=json.load(stream)
    protocol=json.loads(path.with_name(path.name.replace('.json.gz','-protocol.json')).read_text())
    summary=r['summary'];warm=[p for p in r['plans'] if p['phase']=='monitoring' and not p['compilation_included']]
    pos=np.asarray(r['samples']);distance=np.linalg.norm(np.diff(pos[:,1:3],axis=0),axis=1)
    gate=summary['gate_time_s']
    phases=np.array([v[0] for v in r['samples'][1:]])
    environment=json.loads((path.parent/'environment.json').read_text())
    monitoring_visits={}
    for region in environment['regions']:
        lo,hi=np.asarray(region['bounds']);inside=np.all(pos[:,1:3]>=lo,axis=1)&np.all(pos[:,1:3]<=hi,axis=1)
        eligible=inside&(pos[:,0]>=gate) if gate is not None else np.zeros(len(pos),bool)
        monitoring_visits[region['label']]=float(pos[eligible,0][0]) if eligible.any() else None
    method=path.name.split('-')[-1].split('.')[0]
    innovations=np.asarray([p['standardized_innovation'] for p in r.get('observation_predictions',[])],float)
    learning=[p for p in r['plans'] if p['phase']=='learning' and not p['compilation_included']]
    stats=dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),protocol=str(path.with_name(path.name.replace('.json.gz','-protocol.json'))),
        scenario=r['scenario'],method=method,**summary,
        learning_distance_m=float(distance[phases<=gate].sum()) if gate is not None else float(distance.sum()),
        learning_measurements=sum(z[0]<=gate for z in r['observations']) if gate is not None else len(r['observations']),
        total_distance_m=float(distance.sum()),
        replay_verification=r.get('replay_verification'),
        wall_time_scope='Restoration plus continuation; saved-prefix MBD request timings retain their original measurements.' if r.get('replay_verification') else 'Original uninterrupted process through this recording.',
        acquisition_domain=dict(kind='uniform whole free space' if method=='baseline1' else 'clarity-valued geometric path tubes',
            median_free_space_fraction=float(np.median([f['active_cells']/f['free_cells'] for f in r.get('task_frames',[]) if f.get('free_cells')])) if any(f.get('free_cells') for f in r.get('task_frames',[])) else 1. if method=='baseline1' else None,
            interpretation='Nonzero acquisition weight, not a claim that the robot visits every cell. GP correlation can value measurements outside the weighted domain.'),
        first_monitoring_visit_s=monitoring_visits,
        monitoring_cost_history=[[float(z[0]),float(z[14])] for i,z in enumerate(pos)
            if abs(z[0]-round(z[0]))<1e-7 or i==len(pos)-1],
        timing={k:distribution([p['timing'][k] for p in warm if k in p['timing']])
            for k in ('posterior','safety_belief_refresh','fields','geometry','optimization')},
        search=distribution([p['validation']['search_s'] for p in warm]),
        validation=distribution([p['validation']['validation_s'] for p in warm]),
        request=distribution([sum(v for k,v in p['timing'].items() if k not in ('posterior','safety_belief_refresh')) for p in warm]),
        update_and_request=distribution([p['wall_seconds'] for p in warm]),
        learning_request=distribution([sum(v for k,v in p['timing'].items() if k not in ('posterior','safety_belief_refresh')) for p in learning]),
        learning_update_and_request=distribution([p['wall_seconds'] for p in learning]),
        candidate_path_mask_construction=distribution([f['domain_seconds'] for f in r.get('task_frames',[]) if 'domain_seconds' in f]),
        timing_scope='Learning requests exclude the separately reported periodic path-mask construction. MBD request includes fields, geometry, search and validation; update-and-request also includes its preceding POAM update and safety refresh. Simulation-only handoff checks, map diagnostics and serialization are separate from these request times.',
        recorded_work_totals_s={key:float(sum(x['seconds'] for x in r.get(source,[]))) for key,source in
            [('poam_updates','updates'),('safety_field_refresh','safety_field_timing'),('execution','execution_timing')]},
        predictive_innovation=dict(count=len(innovations),outside_three_sigma_disk=int(np.count_nonzero(innovations>3)),
            fraction_outside_three_sigma_disk=float(np.mean(innovations>3)),maximum=float(innovations.max()),
            definition='Norm of the actual noisy reading minus the pre-update posterior mean, divided by sqrt(component latent variance + sensor noise variance). A diagnostic of local predictive calibration; no independent-data or safety guarantee.') if len(innovations) else None,
        relaxed_selected_plans=sum(not p['validation'].get('safety_feasible',[True]*len(p['validation']['feasible']))[p['validation']['selected']]
            for p in r['plans'] if p['phase']=='monitoring'),
        hold_fallbacks=sum(bool(p.get('execution_fallback')) for p in r['plans'] if p['phase']=='monitoring'))
    return stats


def paired_protocol(baseline,proposed):
    b=json.loads(Path(baseline['protocol']).read_text());p=json.loads(Path(proposed['protocol']).read_text())
    critical={'full_learning_monitoring.py','rmse_monitoring.py','efficient_learning_monitoring.py',
        'implicit_monitoring_v2.py','implicit_monitoring_v3.py','legacy_monitoring.py','monitoring_routes.py',
        'guarded_waypoints.py','reachable_learning.py','cached_poam_fields.py','exact_poam_variance.py',
        'spatial_poam.py','observed_poam.py','geometric_task_mask.py'}
    keys=[k for k in b['source_sha256'] if k.startswith('src/') or Path(k).name in critical]
    changed=[k for k in b['source_sha256'] if b['source_sha256'][k]!=p['source_sha256'].get(k)]
    physical=lambda v:{k:x for k,x in v['physical_settings'].items() if k!='mission_seconds'}
    checks=dict(settings=b['settings']==p['settings'],physical_model=physical(b)==physical(p),
        initial_condition=b['initial_condition']==p['initial_condition'],gp_profile=b['gp_profile']==p['gp_profile'],
        selection_rule=b['allow_validation_slack']==p['allow_validation_slack'],handoff_metric=b['handoff_metric']==p['handoff_metric'],
        numerical_sources=all(b['source_sha256'][k]==p['source_sha256'].get(k) for k in keys))
    return dict(checks=checks,matched=all(checks.values()),source_files_checked=len(keys),
        changed_source_files=changed,
        reviewed_driver_changes='Exact matched-duration scheduling and its legacy capacity allocation; diagnostic labels; compact recording buffers that serialize byte-identically; optional verified replay of saved decisions after storage interruption. Replay starts from the original initial condition and regenerates actual measurements through the unchanged model. No physical timestep, inference, actuator limit, controller or monitoring objective changes.',
        intended_duration_matched=baseline['planned_seconds']==proposed['planned_seconds'],
        note='Only legacy allocation capacity may be padded to a five-second block. The intended experiment horizon and actual physics steps are separately recorded.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('output/path_mask_20261105'))
    args=parser.parse_args();runs=[]
    for folder in ('safety-aware-baseline','safety-aware-proposed','safety-aware-baseline-s2','safety-aware-proposed-s2'):
        if folder=='safety-aware-proposed-s2' and (args.root/(folder+'-recovered')/'summary.json').exists():folder+='-recovered'
        runs.extend(read_run(p) for p in sorted((args.root/folder).glob('s*-center-*.json.gz')))
    comparisons=[]
    for s in (1,2):
        members=[r for r in runs if r['scenario']==s]
        if len(members)!=2:continue
        baseline=next(r for r in members if r['method']=='baseline1');proposed=next(r for r in members if r['method']=='pathmask')
        protocol=paired_protocol(baseline,proposed)
        matched=all(r['completed'] for r in members) and abs(baseline['simulated_seconds']-proposed['simulated_seconds'])<1e-8
        common=min(r['simulated_seconds'] for r in members)
        common_cost=[float(np.interp(common,np.array(r['monitoring_cost_history'])[:,0],np.array(r['monitoring_cost_history'])[:,1])) for r in [baseline,proposed]]
        comparisons.append(dict(scenario=s,complete_matched_pair=matched,
            protocol_audit=protocol,
            learning_time_saved_s=baseline['survey_duration_s']-proposed['survey_duration_s'] if all(r['gate_reached'] for r in members) else None,
            common_recorded_time_s=common,common_time_cost_baseline=common_cost[0],common_time_cost_proposed=common_cost[1],
            common_time_cost_reduction_fraction=1-common_cost[1]/common_cost[0] if common_cost[0]>0 else None,
            common_time_cost_scope='Censored comparison over the recorded overlap only; never a completed mission result if either trial failed.',
            monitoring_cost_reduction_fraction=1-proposed['integrated_deficit']/baseline['integrated_deficit'] if matched else None,
            contacts=sum(r['contact'] is not None for r in members),
            both_visit_A=all(r['first_visit_s']['A'] is not None for r in members),
            claimed_success=matched and protocol['matched'] and all(r['contact'] is None for r in members)
                and proposed['survey_duration_s']<baseline['survey_duration_s']
                and proposed['integrated_deficit']<baseline['integrated_deficit']
                and all(r['first_visit_s']['A'] is not None for r in members)))
    report=dict(runs=runs,comparisons=comparisons,
        counts=dict(recorded=len(runs),completed=sum(r['completed'] for r in runs),contacts=sum(r['contact'] is not None for r in runs),
            stopped_without_contact=sum(not r['completed'] and r['stop_reason'] is not None and r['contact'] is None for r in runs)),
        scope='All fresh declared center-in-B trials; no run is removed based on outcome. Earlier diagnostics remain separately archived.',
        comparison='Same initial state, physical model, observation-noise seed, POAM, full uncertainty, MBD and execution filter. Baseline survey + 300 s versus equal proposed total time; all survey monitoring cost counts.',
        limitations=['Different global RMSE and task-posterior stopping domains; both must pass the same finite task-cost validation.',
            'A successful example is not a collision-free guarantee or evidence across all initial conditions.',
            'Finite wind sampling and possible model misspecification remain; infeasible commands and collisions are reported.',
            'Scenario 2 proposed uses a compact in-memory recorder verified to serialize identical values. This changes no numerical model, but whole-run wall times are not an isolated algorithm benchmark.',
            'Warm request timing excludes compilation; update-and-request includes the preceding POAM update and safety-field refresh.'])
    temporary=args.root/'safety-aware-results.json.tmp'
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False));temporary.replace(args.root/'safety-aware-results.json')
    print(json.dumps(dict(counts=report['counts'],comparisons=comparisons)))


if __name__=='__main__':main()
