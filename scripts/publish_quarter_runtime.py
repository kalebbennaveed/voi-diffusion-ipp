"""Publish all completed matched cases without omitting failed flights."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'output/european_quarter/rrt_mbd_runtime_300'
DATA=ROOT/'docs/european_quarter_preview/data'
DESTINATION=DATA/'trials/runtime300'
CASES=[('two','forward'),('two','reverse'),('two','crosswind'),
       ('two','commitment'),('multi','crosswind')]


def main():
    rows=[]
    for layout,case in CASES:
        paths=[RESULTS/f'{layout}_{case}_{method}_8101.json' for method in ('proposed','blind','blind_filter')]
        if not all(path.exists() for path in paths):continue
        records=[json.loads(path.read_text()) for path in paths]
        if not all(r.get('completed') and not r.get('plan_only') for r in records):continue
        reference=records[0]
        assert all(r['config']==reference['config'] and r['mission_seconds']==300 for r in records)
        for r,path in zip(records,paths):
            assert r['environment']==reference['environment']
            rows.append(dict(layout=layout,case=case,method=r['method'],success=r['success'],
                completed_seconds=r['completed_seconds'],cost=r['cost'],failure=r['failure'],
                comparison_eligible=bool(r['success'] and records[1]['success']),
                reduction_vs_blind_percent=100*(1-r['cost']/records[1]['cost']) if r['success'] and records[1]['success'] else None,
                observations=len(r['observations']),source=str(path.relative_to(ROOT))))
        subprocess.run([sys.executable,str(ROOT/'scripts/report_mission_duration.py'),
                        '--results',str(RESULTS),'--layout',layout,'--case',case],check=True)
    notice=(f'Five-minute comparison · {len(rows)//3} recorded environments · seed 8101. '
            'Choose any of the five recordings from the Environment menu above. '
            'Contact-terminated flights are diagnostic only and excluded from performance percentages. Individual development runs.')
    subprocess.run([sys.executable,str(ROOT/'scripts/build_european_quarter_data.py'),
        '--results',str(RESULTS),'--destination',str(DESTINATION),'--asset-prefix','trials/runtime300/',
        '--development','--complete-comparison','--minimum-duration','300','--scale','.35',
        '--inspection-scale','.55','--notice',notice],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/diagnose_quarter_cases.py')],check=True)
    shutil.copy2(RESULTS/'analysis.json',DESTINATION/'analysis.json')
    summary=dict(description=notice,seed=8101,runs=rows)
    (RESULTS/'scenario_comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    shutil.copy2(RESULTS/'scenario_comparison.json',DESTINATION/'scenario_comparison.json')
    for path in RESULTS.glob('mission_duration_*.json'):shutil.copy2(path,DESTINATION/path.name)
    manifest_path=DESTINATION/'manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest.update(timing='runtime_benchmark.json',comparison='scenario_comparison.json')
    manifest_path.write_text(json.dumps(manifest,indent=2))
    root_manifest=DATA/'manifest.json';catalog=json.loads(root_manifest.read_text())
    for study in catalog.get('studies',[]):
        if study['id']=='runtime300':study['title']='5 min · all recorded environments · 6 × 16 MBD'
    root_manifest.write_text(json.dumps(catalog,indent=2))
    subprocess.run([sys.executable,str(ROOT/'scripts/report_rrt_study.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/build_european_quarter_preview.py')],check=True)


if __name__=='__main__':main()
