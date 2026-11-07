"""Append the completed retest; verify every pre-existing demo file is preserved."""
import argparse,json,shutil,hashlib,zipfile
from pathlib import Path
from audit_implicit_monitoring_v2 import audit


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--site',type=Path,required=True);a=p.parse_args()
    report=audit(a.directory);runs=report['runs']
    assert len(runs)==4 and all(r['summary']['completed'] and r['summary']['simulated_seconds']==300 for r in runs)
    dest=a.site/'data/retest-monitoring-300'
    assert not dest.exists(),'Preserve previously published results; choose another dataset name.'
    dest.mkdir(parents=True)
    for r in runs:
        for name in [r['file'],r['id']+'-protocol.json']:shutil.copyfile(a.directory/name,dest/name)
    for name in ['environment.json','audit.json']:shutil.copyfile(a.directory/name,dest/name)
    protocol=json.loads((a.directory/(runs[0]['id']+'-protocol.json')).read_text())
    manifest=dict(duration_s=300,planning_horizon_s=protocol['settings']['horizon'],runs=runs,method_version=protocol['version'])
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    root=Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(dest/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        names=set(protocol['source_sha256']) | {str(p.relative_to(root)) for p in (root/'scripts/airflow').glob('*.py')}
        source_index={}
        for name in sorted(names):
            sha=hashlib.sha256((root/name).read_bytes()).hexdigest()
            if name in protocol['source_sha256']:assert sha==protocol['source_sha256'][name]
            source_index[name]=dict(sha256=sha,recorded_before_run=name in protocol['source_sha256'])
            z.write(root/name,name)
        z.writestr('source-index.json',json.dumps(source_index,indent=2))
        for name in ['scripts/airflow/audit_implicit_monitoring_v2.py','tests/test_implicit_monitoring_v2.py','pyproject.toml','results/airflow/cluster_revision_20261103/v2/environment-spec.json']:
            if name not in names:z.write(root/name,name)
        for scenario in [1,2]:
            for name in ['metadata.json','stationary-velocity.f32']:z.write(a.site/f'data/scenario{scenario}'/name,f'airflow-site/revised/data/scenario{scenario}/{name}')
        z.writestr('REPRODUCE.txt','Set ROBOT_REVIEW_SITE to the extracted airflow-site directory. Run from the source root:\npython scripts/airflow/run_implicit_monitoring_v4.py --output results/new-retest --seconds 300 --scenarios 1,2 --methods aware,baseline --cases center --wall-limit 360 --plan-limit 1\nFirst JIT compilation is separate. Existing output directories are never replaced. CPU timings depend on hardware and system load.\n')
    js=(a.site/'implicit-review.js').read_text().replace('implicit','retest')
    js=js.replace('Measured belief + relaxed RPCBF in planning','Measured belief + relaxed MBD + hard finalist validation')
    js=js.replace('A not reached', "${s.first_visit_s.A===null?'A not reached':`A reached at ${s.first_visit_s.A.toFixed(1)} s`}")
    js=js.replace('No initial survey','No survey · immediate local measurements')
    js=js.replace('Two predicted trajectories','Two independently validated trajectories')
    js=js.replace('Two planning + two safety fields','2 training + 2 validation + 2 safety fields')
    js=js.replace("['optimization_validation','MBD + selected forecasts']", "['mbd_search','MBD search'],['execution_validation','Execution-resolution validation'],['optimization_validation','Search + validation + dispatch']")
    js=js.replace('· ${plan.selected_origin}.<br>Performance', '· ${plan.selected_origin}.<br>Finalist screen ${plan.validation.no_feasible?\'FAILED — fallback, not certified\':\'passed on the two validation fields\'}.<br>Performance')
    (a.site/'retest-review.js').write_text(js)
    (a.site/'retest-review.css').write_text((a.site/'implicit-review.css').read_text().replace('implicit','retest'))
    reaches=sum(r['summary']['first_visit_s']['A'] is not None for r in runs)
    contacts=sum(r['summary']['contact'] is not None for r in runs)
    infeasible=sum(r['summary']['infeasible_execution_steps'] for r in runs)
    maximum=max(r['timing']['total']['max_s'] for r in runs)
    section=f'''<!-- retest-review:start -->
<section id="retest-review" class="retest-review" aria-labelledby="retest-title">
<p class="eyebrow">Retest · original recordings above are preserved</p>
<h2 id="retest-title">Execution-consistent validation and immediate wind measurements</h2>
<p class="retest-intro">Same robot, stationary wind, clarity dynamics, geometric initializations, and independent MBD chains. No exploration reward or hypothetical wind measurements.</p>
<div class="retest-result"><strong>Four completed 300 s runs: {contacts} contacts · {reaches}/4 reach Region A.</strong><p>{'The Region A objective remains unmet; this is not a successful route-switching result.' if reaches<4 else 'Region A is reached in all four recorded runs.'} {infeasible} execution steps required the minimum-slack fallback. No global safety guarantee is claimed. Maximum measured warm planning time: {maximum:.3f} s on CPU.</p></div>
<div class="retest-flow"><span>Actual wind measurement<br><b>POAM update every 0.2 s</b></span><span>6 paths + hold<br><b>43 MBD evaluations</b></span><span>Independent wind fields<br><b>Hard-controller finalist checks</b></span><span>Unchanged confidence disk<br><b>Certified tighter force bound</b></span></div>
<p class="retest-note">15 s horizon · 5 s replanning · two training fields and two independent validation fields. The first plan assumes zero performance wind. One actual local reading is acquired before the first command; there is no survey. Planning beliefs remain frozen within each rollout.</p>
<div id="retest-comparisons"></div>
<details class="retest-method"><summary>Implementation changes, measured runtime, and test evidence</summary>
<p>The relaxed RPCBF remains inside MBD search. Up to two shortlisted trajectories are reevaluated with the hard execution filter at 0.05 s control and 0.01 s physical/clarity integration. If both fail, the hold fallback is evaluated and any failure is reported. Predicted contact is rejected during search. The unchanged monitoring objective and existing clearance/speed terms select feasible finalists.</p>
<p>The tighter directional drag bound encloses the same full 3-sigma confidence disk. Unobserved wind uncertainty is retained for safety. Sensing calculations are batched separately from motion, with numerical equivalence checked against the original coupled integrator, including contact. All seven independent chains retain the original 43 evaluations.</p>
<div id="retest-runtime-table"></div>
<p>Warm timing includes geometric initialization, field construction, search, finalist validation, and the latest actual posterior update. First compilation is reported separately in the recordings. Earlier deadline failures are retained in the diagnostic record; they are not counted as completed missions.</p>
<p><a href="data/retest-monitoring-300/audit.json">Recording audit</a> · <a href="data/retest-monitoring-300/manifest.json">Run manifest and timings</a> · <a href="data/retest-monitoring-300/source.zip">Reproduction source</a></p>
</details></section>
<!-- retest-review:end -->'''
    index=a.site/'index.html';original=index.read_text()
    assert '<!-- retest-review:start -->' not in original
    changed=original.replace('<footer>',section+'\n<footer>',1)
    changed=changed.replace('</head>','<!-- retest-assets:start --><link rel="stylesheet" href="retest-review.css"><!-- retest-assets:end --></head>',1)
    changed=changed.replace('</body>','<!-- retest-scripts:start --><script src="retest-review.js" defer></script><!-- retest-scripts:end --></body>',1)
    index.write_text(changed)
    print(json.dumps(dict(runs=4,contacts=contacts,A_reached=reaches,infeasible_steps=infeasible,max_warm_s=maximum,url='http://127.0.0.1:8786/revised/#retest-review')))

if __name__=='__main__':main()
