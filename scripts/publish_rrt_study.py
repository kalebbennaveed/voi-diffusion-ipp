"""Publish only complete, matched RRT comparisons to the local replay."""
import json
import subprocess
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/european_quarter_preview/data'
STUDIES=[
    ('runtime300','5 min · 0.2 s step · 6 × 16 MBD','rrt_mbd_runtime_300'),
    ('five-minute','5 min mission · 45 s horizon','rrt_mbd_common_300'),
    ('horizon45','45 s horizon · 3.1 m/s wind','rrt_mbd_visits'),
    ('common','45 s · shared filter samples','rrt_mbd_common_construction'),
    ('soft','45 s · softer filter, smaller MBD budget','rrt_mbd_soft_screen'),
    ('wind34','30 s horizon · 3.4 m/s wind','rrt_mbd_wind34'),
]


def publish(folder,destination,prefix,notice):
    subprocess.run([sys.executable,str(ROOT/'scripts/build_european_quarter_data.py'),
                    '--results','output/european_quarter/'+folder,'--development','--complete-comparison',
                    '--scale','.35','--inspection-scale','.55','--destination',destination,
                    '--asset-prefix',prefix,'--notice',notice],check=True,cwd=ROOT)


def main():
    publish('rrt_mbd_clearance','docs/european_quarter_preview/data','',
            'RRT-guided MBD · 120 s development trials, seed 8101. Forward and reversed wind are recorded. '
            'Forward: 6.0% less deficit than blind planning; reversed: 2.4%. The filtered baseline is better in forward wind.')
    catalog=[]
    for identifier,title,folder in STUDIES:
        paths=[ROOT/'output/european_quarter'/folder/f'two_forward_{method}_8101.json' for method in ('proposed','blind','blind_filter')]
        if not all(p.exists() for p in paths):continue
        records=[json.loads(p.read_text()) for p in paths]
        if not all(r.get('completed') and not r.get('plan_only') for r in records):continue
        proposed,blind,filtered=records
        assert all(r['config']==proposed['config'] and r['mission_seconds']==proposed['mission_seconds'] for r in records)
        outcomes=[]
        for case,label in [('forward','Forward wind'),('reverse','Reversed wind')]:
            case_paths=[ROOT/'output/european_quarter'/folder/f'two_{case}_{method}_8101.json' for method in ('proposed','blind','blind_filter')]
            if not all(p.exists() for p in case_paths):continue
            case_runs=[json.loads(p.read_text()) for p in case_paths]
            if not all(r.get('completed') and not r.get('plan_only') for r in case_runs):continue
            pp,bb,ff=case_runs
            reduction=100*(bb['cost']-pp['cost'])/bb['cost']
            outcomes.append(f'{label}: proposed has {abs(reduction):.1f}% {"less" if reduction>=0 else "more"} deficit than blind planning.')
            failures=[f'{r["method"]} ended at {r["completed_seconds"]:.1f} s' for r in case_runs if not r['success']]
            outcomes.append(('Early termination retained in cost: '+', '.join(failures)+'.') if failures else 'All three flights completed without detected contact.')
        notice=(f'RRT-guided MBD · {title}. {proposed["mission_seconds"]:g} s development trials, seed 8101. '+' '.join(outcomes))
        prefix='trials/'+identifier+'/'
        publish(folder,'docs/european_quarter_preview/data/'+prefix,prefix,notice)
        if identifier=='runtime300':
            destination=DATA/prefix
            shutil.copy2(ROOT/'output/european_quarter/runtime_budget_review/comparison.json',destination/'runtime_benchmark.json')
            local_manifest=destination/'manifest.json'
            contents=json.loads(local_manifest.read_text());contents['timing']='runtime_benchmark.json'
            local_manifest.write_text(json.dumps(contents,indent=2))
        catalog.append(dict(id=identifier,title=title,manifest=prefix+'manifest.json',source=folder))
    path=DATA/'manifest.json';manifest=json.loads(path.read_text());manifest['studies']=catalog
    path.write_text(json.dumps(manifest,indent=2))
    subprocess.run([sys.executable,str(ROOT/'scripts/report_rrt_study.py')],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(ROOT/'scripts/build_european_quarter_preview.py')],check=True,cwd=ROOT)
    print(f'Published base forward/reverse comparison and {len(catalog)} additional complete experiments.')


if __name__=='__main__':main()
