"""Publish the full fresh experiment, retaining all methods and environments."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from rerun_quarter_comparisons import CASES,METHODS

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=ROOT/'output/european_quarter/rrt_mbd_execution_300')
    parser.add_argument('--study',default='execution300')
    args=parser.parse_args();results=args.results.resolve()
    destination=ROOT/'docs/european_quarter_preview/data/trials'/args.study
    rows=[];reference=None;timing=[]
    for layout,case in CASES:
        records=[]
        for method in METHODS:
            path=results/f'{layout}_{case}_{method}_8101.json'
            d=json.loads(path.read_text())
            assert d['completed'] and not d.get('plan_only') and d['mission_seconds']==300
            if reference is None:reference=d
            common=lambda c:{k:v for k,v in c.items() if k!='execution_margin'}
            assert common(d['config'])==common(reference['config']) and d['source_hashes']==reference['source_hashes']
            assert d['vehicle']==reference['vehicle']
            records.append(d)
            warmed=[p for i,p in enumerate(d['plans']) if i>=2 and not p['compilation_included']]
            timing.append(dict(layout=layout,case=case,method=method,
                first_plan_seconds=d['plans'][0]['total_planning_seconds'],
                median_plan_seconds=float(np.median([p['total_planning_seconds'] for p in warmed])) if warmed else None,
                median_execution_seconds_per_commit=float(np.median([p['execution_seconds'] for p in warmed])) if warmed else None,
                wall_seconds=d['wall_seconds']))
        assert all(r['environment']==records[0]['environment'] and r['config']==records[0]['config'] for r in records)
        for d in records:
            row={k:d[k] for k in ('layout','case','method','seed','success','failure','cost','completed_seconds','min_clearance')}
            row.update(observations=len(d['observations']),file=f'{layout}_{case}_{d["method"]}_8101.json')
            for method in ('blind','blind_filter'):
                baseline=next(r for r in records if r['method']==method)
                row[f'reduction_vs_{method}_percent']=100*(1-d['cost']/baseline['cost']) if d['success'] and baseline['success'] else None
            rows.append(row)
        subprocess.run([sys.executable,str(ROOT/'scripts/report_mission_duration.py'),'--results',str(results),'--layout',layout,'--case',case],check=True,stdout=subprocess.DEVNULL)
    successes=sum(row['success'] for row in rows)
    notice=f'Fresh 5-minute runs · all 5 environments × 3 methods · {successes}/15 flights completed without detected contact · seed 8101. Open All comparisons for every result.'
    subprocess.run([sys.executable,str(ROOT/'scripts/build_european_quarter_data.py'),'--results',str(results),
        '--destination',str(destination),'--asset-prefix',f'trials/{args.study}/','--development','--complete-comparison',
        '--minimum-duration','300','--scale',str(reference['config']['environment_scale']),
        '--inspection-scale',str(reference['config']['inspection_scale']),'--notice',notice],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/diagnose_quarter_cases.py'),'--results',str(results)],check=True)
    shutil.copy2(results/'analysis.json',destination/'analysis.json')
    comparison=dict(description=notice,source=str(results),runs=rows,
        changes='Planning remains soft. Proposed and filtered blind use the same hard execution projection when feasible, with explicitly logged minimum-slack fallback. All three update tracking at 20 Hz; nonlinear physics remains 100 Hz. Execution margin: 0.55 m in two-site environments, 0.70 m in the four-site environment. Settings are identical across all methods within each environment.',
        interpretation='One matched development seed. No claims of statistical superiority; negative reductions mean proposed performs worse.')
    (results/'scenario_comparison.json').write_text(json.dumps(comparison,indent=2,allow_nan=False))
    (results/'runtime_benchmark.json').write_text(json.dumps(dict(description='Measured within these new missions; first two planner calls excluded from warmed medians. Some environment reruns overlap, so these latencies are not isolated hardware benchmarks. Physics time is separate from planning and simulation time.',runs=timing),indent=2,allow_nan=False))
    for path in [results/'scenario_comparison.json',results/'runtime_benchmark.json',results/'experiment.json',*results.glob('mission_duration_*.json')]:shutil.copy2(path,destination/path.name)
    manifest_path=destination/'manifest.json';manifest=json.loads(manifest_path.read_text())
    assert len(manifest['runs'])==15
    manifest.update(timing='runtime_benchmark.json',comparison='scenario_comparison.json')
    manifest_path.write_text(json.dumps(manifest,indent=2))
    root_manifest=ROOT/'docs/european_quarter_preview/data/manifest.json';catalog=json.loads(root_manifest.read_text())
    catalog['default_study']=args.study
    studies=[s for s in catalog.get('studies',[]) if s['id']!=args.study]
    for study in studies:
        if study['id']=='runtime300':study['title']='Archive · original 5 min runs (soft execution filter)'
    catalog['studies']=[dict(id=args.study,title='Fresh 5 min · all 5 environments · corrected execution filter',manifest=f'trials/{args.study}/manifest.json',source=results.name),*studies]
    root_manifest.write_text(json.dumps(catalog,indent=2))
    old_manifest=root_manifest.parent/'trials/runtime300/manifest.json'
    if old_manifest.exists():
        old=json.loads(old_manifest.read_text())
        old['notice']='Archived original recordings before the execution-filter correction. A fresh five-minute comparison now includes all five environments and all three methods; open the latest comparison below.'
        old_manifest.write_text(json.dumps(old,indent=2))
    # Verifiable hashes refer to the exact recordings copied into the viewer.
    hashes={row['file']:hashlib.sha256((destination/row['file']).read_bytes()).hexdigest() for row in rows}
    (destination/'recording_hashes.json').write_text(json.dumps(hashes,indent=2))
    readme=ROOT/'docs/european_quarter_preview/README.txt'
    divider='\nARCHIVED EXPERIMENT NOTES\n\n'
    historical=readme.read_text().split(divider,1)[-1]
    table=['Environment                 Proposed       Blind     Blind + filter']
    for layout,case in [('two','forward'),('two','reverse'),('two','crosswind'),('two','commitment'),('multi','crosswind')]:
        entries={r['method']:r for r in rows if r['layout']==layout and r['case']==case}
        table.append(f'{layout+" / "+case:26s}'+''.join(f'{entries[m]["cost"]:12.4f}' for m in METHODS))
    current=f'''CURRENT EXPERIMENT — FRESH FIVE-MINUTE COMPARISONS

{notice}

Use the Environment selector or All comparisons to open any of the five cases.
The main view and each spatial map independently select Proposed, Baseline, or
Both. The Baseline selector switches between blind planning and blind planning
with the execution filter. Values and trajectories are recorded simulation data.

Execution correction:
- Planning retains its soft constraint filter and penalized slack.
- Proposed and filtered blind share the same execution filter: exact 2D
  projection onto the sampled constraints when feasible. Infeasible constraints
  use the existing minimum-slack fallback; slack remains in the audit logs.
- Tracking and filtering update every 0.05 s, independently of the 0.2 s planner
  discretization. Nonlinear rigid-body and motor dynamics integrate at 0.01 s.
- Execution uses a 0.55 m distance margin in the two-site environments and
  0.70 m in the four-site environment; all methods within an environment share
  this setting. The physical collision radius is 0.275 m. The margin reserves
  space for tracking error. This is an
  empirical controller correction, not a safety guarantee under unknown wind.
- Blind without filter uses the same faster tracking clock without projection.
  No controller accesses the true wind field; it uses its observed posterior.

Every method restarts from t=0. Within each environment, geometry, wind, initial state, vehicle, seed,
planning horizon (45 s), replanning interval (5 s), cosine schedule, temperature,
6 x 16 diffusion budget, and training/validation field counts remain matched.
The planner predicts with its soft filter; actual execution uses the stricter
filter. Their mismatch is not hidden in the performance measure.

Time-integrated mean clarity deficit (lower is better):
{chr(10).join(table)}

All comparisons, including negative percentage improvements, are displayed.
The full cost includes early monitoring losses. There is no added information
reward in the reported metric. These are individual development runs, not a
statistical demonstration that proposed is better in every environment.

Reproduce: scripts/rerun_quarter_comparisons.py, then scripts/rerun_quarter_four_site.py
Assemble: scripts/assemble_quarter_comparisons.py (copies the recorded files unchanged)
Audit: scripts/audit_rrt_quarter.py --results {results.relative_to(ROOT)}
Publish: scripts/publish_quarter_comparisons.py
Data: data/trials/{args.study}/scenario_comparison.json
Evidence: analysis.json, execution_integrity.json, recording_hashes.json, and
runtime_benchmark.json in the same directory. Raw 100 Hz records remain in
{results.relative_to(ROOT)}.
Planning wall time is outside the simulated mission clock. Some reruns overlap;
the reported timings are not isolated hardware benchmarks.
The earlier four-site filtered-blind attempt contacted a wall at 264.44 s. It
remains in rrt_mbd_execution_300_v1, and the new four-site runs all use 0.70 m.
'''
    readme.write_text(current+divider+historical)
    subprocess.run([sys.executable,str(ROOT/'scripts/build_european_quarter_preview.py')],check=True)

if __name__=='__main__':main()
