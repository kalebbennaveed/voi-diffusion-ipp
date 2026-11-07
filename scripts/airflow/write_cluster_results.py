"""Populate the revised demo summary from audited recordings, not handpicked runs."""
import argparse,json,re
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);a=p.parse_args()
d=a.site/'data/monitoring-cluster-300';m=json.loads((d/'manifest.json').read_text());audit=json.loads((d/'audit.json').read_text());runs=m['runs'];pairs=m['comparisons']
assert len(runs)==24 and len(pairs)==12, 'Do not publish an incomplete or selected subset'
assert audit['acceptance_passed'], audit['acceptance_checks']
blind=sum(x['baseline_plans_central'] for x in pairs);contact=sum(x['summary']['contact'] is not None for x in runs)
aware=[r for r in runs if r['method']=='aware'];base=[r for r in runs if r['method']=='baseline'];visit=lambda rr:sum(r['summary']['first_visit_s']['A'] is not None for r in rr)
rows=[]
for s in (1,2):
 rr=[r for r in aware if r['scenario']==s];expected='lower' if s==1 else 'upper';chosen=sum(r['summary']['first_transfer']['route']==expected for r in rr);times=[r['summary']['first_visit_s']['A'] for r in rr if r['summary']['first_visit_s']['A'] is not None];gains=[r['deficit_improvement_percent'] for r in pairs if r['scenario']==s]
 rows.append(f'<tr><th>Scenario {s}</th><td>{expected.capitalize()} · {chosen}/6</td><td>{min(times):.1f}–{max(times):.1f} s</td><td>{min(gains):.1f}–{max(gains):.1f}%</td></tr>')
content=f'<strong>24 recorded missions · six initial conditions per scenario · 300 s each</strong><br>The wind-blind baseline first plans through the middle in <strong>{blind}/12</strong> comparisons. Disturbance-aware MBD reaches A in <strong>{visit(aware)}/12</strong> runs; the baseline reaches it in <strong>{visit(base)}/12</strong>. Recorded contacts: <strong>{contact}/24</strong>.<table class="monitor-all-results"><thead><tr><th>Environment</th><th>Aware first route to A</th><th>Aware first A visit</th><th>Deficit reduction vs baseline</th></tr></thead><tbody>{"".join(rows)}</tbody></table><span>These are the six declared conditions, not a guarantee for every starting state. The table below each comparison distinguishes the baseline’s planned route from its eventual executed route.</span>'
med={method:float(np.median([r['timing']['median_s'] for r in runs if r['method']==method])) for method in ('aware','baseline')};maximum=max(r['timing']['max_s'] for r in runs)
budget=f'15 s horizon · 120 free acceleration coordinates · 43 MBD search evaluations plus one retained forecast. Across completed missions, median warm planning is {med["aware"]*1000:.0f} ms (aware) / {med["baseline"]*1000:.0f} ms (baseline); maximum {maximum:.3f} s. First-call compilation is separate. An earlier attempt hit 1.497 s and was stopped by the budget guard; five repetitions of that exact call took 0.187–0.198 s. The original overrun’s cause was not isolated. <a href="data/runtime-investigation.json">Timing investigation ↗</a>'
f=a.site/'index.html';html=f.read_text();html=re.sub(r'<p class="revision-results" id="revision-results">.*?</p>',f'<div class="revision-results" id="revision-results">{content}</div>',html,flags=re.S);html=html.replace('<p id="revision-budget"></p>',f'<p id="revision-budget">{budget}</p>');f.write_text(html)
print(json.dumps(dict(baseline_central=blind,aware_visits_A=visit(aware),baseline_visits_A=visit(base),contacts=contact,median_s=med,maximum_completed_plan_s=maximum),indent=2))
