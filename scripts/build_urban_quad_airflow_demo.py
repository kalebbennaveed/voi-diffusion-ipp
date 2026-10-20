"""Publish only completed, finite-planning mission records to the local replay."""
import json
import shutil
from pathlib import Path
import numpy as np
from voi_diffusion_ipp.experiments.urban_airflow import REVISION,wind_velocity

root=Path('output/all_methods_demo/urban-quad-v2')
root.mkdir(parents=True,exist_ok=True)
shutil.copytree('docs/urban_quad_v2',root,dirs_exist_ok=True)
(root/'data').mkdir(exist_ok=True)
runs=[]
for path in sorted(Path('output/urban_quad_airflow').glob('*_8101.json')):
    data=json.loads(path.read_text())
    if data.get('model_revision')!=REVISION or data['mission_seconds']!=360 or any(p['predicted_cost'] is None for p in data['plans']):
        print('Exclude incomplete/invalid development run:',path.name);continue
    if data['config']['dt']*(data['config']['stage_steps']+data['config']['future_steps'])!=(240 if data['case']=='commitment' else 120):
        print('Exclude superseded prediction horizon:',path.name);continue
    expected=np.asarray(wind_velocity(np.asarray(data['environment']['grid']),data['case']))
    if not np.allclose(expected,data['environment']['wind'],atol=1e-6):
        print('Exclude superseded wind field:',path.name);continue
    shutil.copy2(path,root/'data'/path.name)
    runs.append(dict(case=data['case'],method=data['method'],file='data/'+path.name))
(root/'data/manifest.json').write_text(json.dumps(dict(runs=runs),indent=2))
for extra in ['validation.json','passage_validation.json','airspeed_validation.json']:
    source=Path('output/urban_quad_airflow')/extra
    if source.exists():shutil.copy2(source,root/'data'/extra)
print('Local replay:',root,';',len(runs),'verified planning records')

# Update the existing local entry only after all twelve current records exist.
# Preserve the old complete renderer/data as a separately accessible archive.
if len(runs)==12:
    old=Path('output/all_methods_demo/urban-quad')
    archive=Path('output/all_methods_demo/urban-quad-v1')
    if old.exists() and not archive.exists():shutil.copytree(old,archive)
    old.mkdir(exist_ok=True)
    (old/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Updated urban quadrotor experiment</title><p><a href="../urban-quad-v2/">Open the updated urban quadrotor experiment</a> · <a href="../urban-quad-v1/">Previous version</a></p><script>location.replace("../urban-quad-v2/"+location.search+location.hash)</script>')
