"""Publish only completed, finite-planning mission records to the local replay."""
import json
import shutil
from pathlib import Path

root=Path('output/all_methods_demo/urban-quad')
root.mkdir(parents=True,exist_ok=True)
shutil.copytree('docs/urban_quad_demo',root,dirs_exist_ok=True)
(root/'data').mkdir(exist_ok=True)
runs=[]
for path in sorted(Path('output/urban_quad').glob('*_8101.json')):
    data=json.loads(path.read_text())
    if data.get('model_revision')!='matched_limits_v3' or data['mission_seconds']!=120 or any(p['predicted_cost'] is None for p in data['plans']):
        print('Exclude incomplete/invalid development run:',path.name);continue
    shutil.copy2(path,root/'data'/path.name)
    runs.append(dict(case=data['case'],method=data['method'],file='data/'+path.name))
(root/'data/manifest.json').write_text(json.dumps(dict(runs=runs),indent=2))
print('Local replay:',root,';',len(runs),'verified planning records')
