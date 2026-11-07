"""Read-only comparison; distinguish completed missions from failed diagnostics."""
import argparse,gzip,json,hashlib,zipfile
from pathlib import Path
import numpy as np

def stats(values):
    values=np.asarray(values,float)
    return dict(count=len(values),median=float(np.median(values)),p95=float(np.percentile(values,95)),max=float(values.max())) if len(values) else None

def evaluate(folder):
    env=json.loads((folder/'environment.json').read_text());reports=[]
    lo,hi=np.asarray(next(r for r in env['regions'] if r['label']=='B')['bounds'])
    with zipfile.ZipFile(folder/'executed-source.zip') as archive:
        source_hashes={name:hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
    for path in sorted(folder.glob('s*-center-*.json.gz')):
        r=json.load(gzip.open(path));sm=r['summary'];s=np.asarray(r['samples'])
        protocol=json.loads(path.with_name(path.name.replace('.json.gz','-protocol.json')).read_text())
        assert protocol['source_sha256']==source_hashes
        first=sm['first_visit_s']['A'];returned=None
        if first is not None:
            inside=(s[:,0]>first)&np.all(s[:,1:3]>=lo,axis=1)&np.all(s[:,1:3]<=hi,axis=1)
            if inside.any():returned=float(s[np.flatnonzero(inside)[0],0])
        plans=[p for p in r['plans'] if p['phase']=='monitoring'];warm=[p for p in plans if not p['compilation_included']]
        # A monitoring probe can dispatch a learning reference at the same
        # timestamp. The final record includes that probe in its wall time;
        # count one actual decision cycle, not both log entries.
        cycles={}
        for plan in r['plans']:cycles.setdefault(plan['t'],[]).append(plan)
        warm_cycles=[entries[-1] for entries in cycles.values()
                     if not any(p['compilation_included'] for p in entries)]
        dispatch=np.asarray([p['t'] for p in r['plans']]);phases=np.asarray([p['phase']=='monitoring' for p in r['plans']])
        plan_indices=np.clip(np.searchsorted(dispatch,s[1:,0]-.025,side='right')-1,0,len(phases)-1)
        executed_monitoring=float(np.sum(phases[plan_indices])*.05)
        accepted=np.asarray([p['phase']=='monitoring' and 'execution_fallback' not in p
                             and not p.get('validation',{}).get('no_feasible',False) for p in r['plans']])
        executed_accepted=float(np.sum(accepted[plan_indices])*.05)
        sensing=[p for p in r['plans'] if p['phase']=='learning' and p.get('timing') and not p['compilation_included']]
        gate=sm['gate_time_s'];frame=next((b for b in r['belief_frames'] if gate is not None and b['t']==gate and b.get('task_vector_std_m_s') is not None),None)
        innovations=[z['standardized_innovation'] for z in r.get('observation_predictions',[])]
        finite=[p for p in plans if p['initial_cost'] is not None and p['optimized_training_cost'] is not None and np.isfinite(p['initial_cost']) and np.isfinite(p['optimized_training_cost'])]
        reports.append(dict(id=r['id'],scenario=r['scenario'],method=r['method'],source_snapshot_verified=True,
            complete_300=sm['completed'] and sm['simulated_seconds']==300,simulated_seconds=sm['simulated_seconds'],
            contact=sm['contact'],first_A_s=first,return_B_s=returned,mbd_start_s=gate,mbd_calls=len(plans),
            monitoring_seconds=executed_monitoring,
            accepted_mbd_execution_seconds=executed_accepted,
            filtered_hold_fallback_seconds=executed_monitoring-executed_accepted,
            targeted_information_fallback_requests=sum(bool(p.get('validation',{}).get('request_targeted_learning',False)) and p.get('execution_mode')=='targeted wind learning' for p in plans),
            full_mission_monitoring_cost=sm['integrated_deficit'] if sm['completed'] else None,
            partial_monitoring_cost=sm['integrated_deficit'] if not sm['completed'] else None,
            whole_map_wind_rmse_m_s=sm['final_rmse_m_s'],task_wind_rmse_m_s=sm.get('final_task_rmse_m_s'),
            task_uncertainty_at_handoff=frame.get('task_vector_std_m_s') if frame else None,
            actual_task_error_at_handoff=frame.get('task_rmse_m_s') if frame else None,
            handoff_kind=sm.get('gate_kind'),
            route_uncertainty_at_handoff=frame.get('route_vector_std_m_s') if frame else None,
            actual_route_error_at_handoff=frame.get('route_rmse_m_s') if frame else None,
            final_route_uncertainty_m_s=sm.get('final_route_vector_std_m_s'),
            final_route_error_m_s=sm.get('final_route_rmse_m_s'),
            infeasible_execution_steps=sm['infeasible_execution_steps'],minimum_clearance_m=sm['min_clearance_m'],
            warm_trajectory_request_s=stats([sum(p['timing'][k] for k in ['fields','geometry','optimization']) for p in warm]),
            warm_full_update_and_plan_s=stats([p['wall_seconds'] for p in warm]),
            warm_actual_decision_cycle_s=stats([p['wall_seconds'] for p in warm_cycles]),
            warm_actual_decision_deadline_misses=sum(p['wall_seconds']>1 for p in warm_cycles),
            warm_mbd_module_s={k:stats([p['timing'][k] for p in warm])
                               for k in ('fields','geometry','optimization','posterior','safety_belief_refresh')},
            warm_mbd_search_s=stats([p['validation']['search_s'] for p in warm if 'search_s' in p.get('validation',{})]),
            warm_mbd_validation_s=stats([p['validation']['validation_s'] for p in warm if 'validation_s' in p.get('validation',{})]),
            warm_sensing_trajectory_request_s=stats([sum(v for k,v in p['timing'].items() if k not in ('posterior','safety_belief_refresh')) for p in sensing]),
            warm_sensing_update_and_plan_s=stats([p['wall_seconds'] for p in sensing]),
            sensing_hold_decisions=sum(p.get('selected_candidate')==0 for p in sensing),
            sensing_unvalidated_decisions=sum(p.get('no_feasible',False) for p in sensing),
            acquisition_s=stats([f['seconds'] for f in r.get('task_frames',[])[1:]]),
            kernel_training_s=stats([u['seconds'] for u in r['updates'] if u['training'] and not u['compilation_included']]),
            conditioning_s=stats([u['seconds'] for u in r['updates'] if not u['training'] and not u['compilation_included']]),
            premeasurement_three_sigma_innovation_coverage=float(np.mean(np.array(innovations)<=3)) if innovations else None,
            plans_improved_over_initial_training_cost=sum(p['optimized_training_cost']<p['initial_cost']-1e-6 for p in finite),
            finite_training_comparisons=len(finite)))
    return reports

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
    rows=evaluate(a.folder);(a.folder/'comparison.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    print(json.dumps(rows,indent=2,allow_nan=False))
