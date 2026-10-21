"""Summarize all RRT development trials without dropping unsuccessful runs."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'output/european_quarter'
TRIALS={
    'rrt_mbd_runtime_300':'300 s mission; 45 s horizon; 0.2 s planning step; 6 × 16 MBD',
    'rrt_mbd_common_300':'300 s mission; 45 s horizon; common construction draws',
    'rrt_mbd_mission':'30 s; smaller clearance penalty margin',
    'rrt_mbd_clearance':'30 s; 0.6 m clearance penalty margin',
    'rrt_mbd_visits':'45 s; up to three RRT site visits',
    'rrt_mbd_common_construction':'45 s; common construction draws',
    'rrt_mbd_soft_screen':'45 s; softer filter, smaller MBD budget',
    'rrt_mbd_wind34':'30 s; stronger 3.4 m/s wind',
}


def main():
    rows=[]
    for folder,description in TRIALS.items():
        for file in sorted((RESULTS/folder).glob('two_*.json')):
            d=json.loads(file.read_text())
            if d.get('plan_only') or d.get('config',{}).get('initialization')!='rrt':continue
            complete=bool(d.get('completed'))
            times=[p.get('total_planning_seconds',p['seconds']) for i,p in enumerate(d['plans']) if i>=2 and not p.get('compilation_included') and not d['plans'][i-1].get('compilation_included')]
            row=dict(trial=folder,description=description,case=d['case'],method=d['method'],seed=d['seed'],
                status=d['status'],completed=complete,success=d['success'] if complete else None,
                physical_seconds=d['completed_seconds'],mission_seconds=d['mission_seconds'],
                mission_cost=d['cost'] if complete else None,partial_cost=d['cost'] if not complete else None,
                minimum_clearance=d['min_clearance'],failure=d['failure'],
                median_planning_seconds=float(np.median(times)) if times else None,
                config=d['config'],source=str(file.relative_to(ROOT)))
            rows.append(row)
    for row in rows:
        baseline=next((b for b in rows if b['trial']==row['trial'] and b['case']==row['case'] and b['seed']==row['seed'] and b['method']=='blind'),None)
        row['reduction_vs_blind_percent']=None
        if row['completed'] and baseline and baseline['completed']:
            assert row['config']==baseline['config']
            assert row['mission_seconds']==baseline['mission_seconds']
            row['reduction_vs_blind_percent']=100*(baseline['mission_cost']-row['mission_cost'])/baseline['mission_cost']
    description=('Development tuning on seed 8101. These are not independent confirmation trials. '
                 'Incomplete mission costs are excluded from percentages. Failed missions include the unsensed remainder. '
                 'Planning computation is outside the physical simulation clock. No real-time or safety guarantee is claimed.')
    (RESULTS/'rrt_study.json').write_text(json.dumps(dict(description=description,runs=rows),indent=2,allow_nan=False))
    lines=['# RRT-guided MBD development evidence','',description,'',
           'RRT supplies geometry-based initial trajectories. MBD optimizes every nominal acceleration interval; '
           'the nonlinear quadrotor executes the selected trajectory. All methods use the same RRT procedure and budget within each trial.','',
           '| Trial | Wind | Method | Status | Physical time (s) | Mission deficit integral | Reduction vs blind | Min. clearance (m) | Median planning (s) |',
           '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        cost='—' if r['mission_cost'] is None else f"{r['mission_cost']:.4f}"
        reduction='—' if r['reduction_vs_blind_percent'] is None else f"{r['reduction_vs_blind_percent']:+.2f}%"
        timing='—' if r['median_planning_seconds'] is None else f"{r['median_planning_seconds']:.2f}"
        lines.append(f"| {r['description']} | {r['case']} | {r['method']} | {r['status']} | {r['physical_seconds']:.2f} | {cost} | {reduction} | {r['minimum_clearance']:.3f} | {timing} |")
    lines.extend(['','## Reproduction','',
                  'The accompanying `rrt_study.json` gives every source path and complete configuration. '
                  'Each source recording contains source-code checksums, per-call RRT paths, sampled control vectors, '
                  'MBD traces and links to the 100 Hz execution records.','',
                  'Run `scripts/audit_rrt_quarter.py --results <trial directory>` to reconstruct the replay and reported cost. '
                  'Run `scripts/report_european_quarter.py --results <trial directory>` for measured passage crossings and sensing time.'])
    (RESULTS/'RRT_STUDY.md').write_text('\n'.join(lines)+'\n')
    print(f'{len(rows)} runs documented in {RESULTS / "RRT_STUDY.md"}')


if __name__=='__main__':main()
