"""Publish audited legacy runs, preserving the existing animation sections."""
import argparse,json,shutil,zipfile,io,gzip,hashlib
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def package(directory,site,name,compress_records=False):
    audit=json.loads((directory/'audit.json').read_text());assert len(audit['checks'])==4
    out=site/'data'/name;out.mkdir(parents=True,exist_ok=True)
    runs=[];times=[];first=[];survey=False;durations=set();rejected=[]
    for scenario in (1,2):
        for method in ('aware','baseline'):
            ident=f's{scenario}-{method}';record=json.loads((directory/f'{ident}.json').read_text());protocol=json.loads((directory/f'{ident}-protocol.json').read_text())
            summary=record['summary']
            rejected.extend(dict(id=ident,**call) for call in protocol.get('rejected_planning_calls',[]))
            assert summary['completed'] and summary['simulated_seconds']==summary['planned_seconds']
            durations.add(summary['simulated_seconds'])
            l=protocol['legacy_settings'];settings=protocol['settings']|dict(diffusion_steps=l['iterations'],proposals=l['proposals'],horizon_seconds=l['horizon'],commit_seconds=l['commit'],multimode=True,legacy_safety_costs=True,planning_clearance=l['planning_clearance'],mode_count=l['mode_count'],validation_worlds=l['validation_worlds'])
            for i,p in enumerate(record['plans']):
                if 'selected_origin' not in p:
                    raw=np.load(directory/f'{ident}-plan-{i:03d}.npz');same=np.all(raw['search_finalists']==raw['theta'][None],axis=1);ix=np.flatnonzero(same)[0]
                    p['selected_origin']='initialization' if raw['search_finalist_indices'][ix]<len(raw['search_initial_parameters']) else 'diffusion'
                (first if p['timing']['compilation_included'] else times).append(p['wall_seconds'])
            payload=json.dumps(record,separators=(',',':')).encode()
            filename=ident+('.json.gz' if compress_records else '.json')
            encoded=gzip.compress(payload,mtime=0) if compress_records else payload
            if compress_records:assert gzip.decompress(encoded)==payload
            (out/filename).write_bytes(encoded)
            # Replace only this generated viewer record, retaining all source data.
            if compress_records:(out/f'{ident}.json').unlink(missing_ok=True)
            shutil.copyfile(directory/f'{ident}-protocol.json',out/f'{ident}-protocol.json')
            runs.append(dict(id=ident,scenario=scenario,method=method,file=filename,decoded_sha256=hashlib.sha256(payload).hexdigest(),summary=record['summary']))
            survey='initial_survey' in protocol
    for file in ('environment.json','audit.json'):shutil.copyfile(directory/file,out/file)
    if (directory/'belief-audit.json').exists():shutil.copyfile(directory/'belief-audit.json',out/'belief-audit.json')
    assert len(durations)==1, 'All four runs must cover the same completed duration'
    duration=durations.pop()
    assert all(check['seconds']==duration for check in audit['checks'])
    manifest=dict(duration_s=duration,state='complete',implementation='legacy-mbd',planner_settings=settings,legacy_settings=l,initial_wind_knowledge='survey' if survey else 'cold-start',execution_belief=protocol.get('execution_belief'),prior_configuration=protocol.get('prior_configuration'),runtime=dict(warm_min_s=min(times),warm_max_s=max(times),warm_calls=len(times),first_call_min_s=min(first),first_call_max_s=max(first),limit_s=1.),runs=runs,verification='audit.json',source_available=True)
    manifest['runtime']['rejected_calls']=rejected
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in (ROOT/'src').rglob('*.py'):z.write(p,str(p.relative_to(ROOT)))
        for name in ('legacy_monitoring.py','run_legacy_monitoring.py','benchmark_legacy_monitoring.py','prepare_legacy_wind_survey.py','audit_legacy_monitoring.py','audit_belief_comparison.py','package_legacy_monitoring.py','monitoring_mbd.py','monitoring_routes.py','run_monitoring_mbd.py','robot_review.py'):
            p=ROOT/'scripts/airflow'/name;z.write(p,str(p.relative_to(ROOT)))
        for p in directory.glob('source_snapshot*/**/*.py'):z.write(p,str(p.relative_to(directory)))
        for p in (ROOT/'tests/quarter').glob('test_*.py'):z.write(p,str(p.relative_to(ROOT)))
        z.write(ROOT/'tests/test_legacy_monitoring.py','tests/test_legacy_monitoring.py');z.write(ROOT/'pyproject.toml','pyproject.toml')
        # Viewer records are served next to this archive; avoid duplicating the
        # large heat-map/forecast arrays inside the source-and-inputs download.
        viewer_files={r['file'] for r in runs}
        for p in out.glob('*.json'):
            if p.name not in viewer_files:z.write(p,'results/'+p.name)
        z.writestr('VIEWER-RECORDS.txt', 'Full recorded trajectories, clarity frames and forecast traces are available beside source.zip at:\n'+''.join(r['file']+'\n' for r in runs)+'The reproduction command regenerates these files and the full execution audit data.\n')
        for r in runs:
            raw=np.load(directory/f"{r['id']}-execution.npz");buf=io.BytesIO();np.savez_compressed(buf,samples=raw['samples'],observation_rows=raw['observation_rows']);z.writestr(f"results/{r['id']}-controls-observations.npz",buf.getvalue())
        for scenario in (1,2):
            for name in ('stationary-velocity.f32','metadata.json'):z.write(site/f'data/scenario{scenario}/{name}',f'scenario-site/data/scenario{scenario}/{name}')
            if survey:z.write(ROOT/f'results/airflow/legacy_survey_s{scenario}.npz',f'initial-belief/legacy_survey_s{scenario}.npz')
        z.write(site/'app.js','scenario-site/app.js')
        execution=protocol.get('execution_belief')
        extras=('--matched-prior ' if protocol.get('prior_configuration','').startswith('survey-matched') else '')
        extras+=f"--proposal-batch {l['batch_size']} "
        if execution:extras+=f"--posterior-update-seconds {execution['posterior_update_seconds']} --execution-margin {execution['safety_margin_m']} --execution-fields {execution['construction_fields']} "
        z.writestr('REPRODUCE.txt',f"Install pyproject.toml dependencies and set PYTHONPATH=src:scripts/airflow.\nFor S=1,2 and M=aware,baseline:\npython scripts/airflow/run_legacy_monitoring.py --scenario S --method M --seconds {duration:g} --horizon {l['horizon']} --levels {l['iterations']} --proposals {l['proposals']} --worlds {l['worlds']} --validation {l['validation_worlds']} --validation-batch {l['validation_batch_size']} --site scenario-site --output reproduced "+extras+(' --survey initial-belief/legacy_survey_sS.npz' if survey else '')+"\nReplace S and M with their values. The runtime guard stops a warm plan above 1 s; first-call compilation is separate. Use an external process timeout as well.\npython scripts/airflow/audit_legacy_monitoring.py --directory reproduced --site scenario-site\nSource snapshots preserve exact earlier driver versions.\nAfter inspecting a planning-budget stop, --resume continues from its exact checkpoint and retains the rejected timing.\npython scripts/airflow/audit_belief_comparison.py --directory reproduced --site scenario-site --survey-directory initial-belief\n")
    print(json.dumps(dict(directory=str(out),runtime=manifest['runtime'],archive_bytes=(out/'source.zip').stat().st_size)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--site',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--compress-records',action='store_true');a=p.parse_args();package(a.directory,a.site,a.name,a.compress_records)
