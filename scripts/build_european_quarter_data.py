"""Publish recorded quarter runs and reproducible initial spatial maps locally."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from voi_diffusion_ipp.quarter.environment import Settings,metadata,initial_belief,wind
from voi_diffusion_ipp.quarter.continuous import ControlSettings

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--results',default='output/european_quarter/full_control_v1');p.add_argument('--development',action='store_true');p.add_argument('--complete-comparison',action='store_true');p.add_argument('--minimum-duration',type=float,default=120.);p.add_argument('--scale',type=float,default=.6);p.add_argument('--inspection-scale',type=float,default=1.);p.add_argument('--notice',help='Explicit evidence/status statement displayed above the recording');p.add_argument('--destination',default='docs/european_quarter_preview/data');p.add_argument('--asset-prefix',default='');args=p.parse_args()
out=root/args.destination;out.mkdir(parents=True,exist_ok=True)
cfg=ControlSettings(environment_scale=args.scale,inspection_scale=args.inspection_scale);b=initial_belief(cfg)
for layout in ['two','multi']:
    for case in ['forward','reverse','crosswind','commitment']:
        env=metadata(layout,args.scale,args.inspection_scale,case);bounds=np.array(env['bounds'])
        xx,zz=np.meshgrid(np.linspace(bounds[0,0],bounds[1,0],49),np.linspace(bounds[0,1],bounds[1,1],41));grid=np.c_[xx.ravel(),zz.ravel()].astype(np.float32)
        mu,cov=b.predict(jnp.array(grid));std=np.sqrt(np.maximum(np.asarray(cov)[:,0,0],0))
        env.update(grid=grid.tolist(),grid_shape=[41,49],wind=np.asarray(wind(jnp.array(grid),case,scale=args.scale)).tolist())
        q=np.full(len(env['cells']),.35);q[:42]=.65
        data=dict(environment=env,q=q.tolist(),mean=np.asarray(mu).tolist(),std=std.tolist(),position=[-28*args.scale,8*args.scale,3.2])
        (out/f'initial_{layout}_{case}.json').write_text(json.dumps(data,separators=(',',':')))
        if case=='forward':(out/f'initial_{layout}.json').write_text(json.dumps(data,separators=(',',':')))

manifest=[]
for file in sorted((root/args.results).glob('*.json')):
    data=json.loads(file.read_text())
    if not data.get('completed') or data.get('plan_only'):continue
    if data.get('revision')!='quarter-full-control-mbd-v1':continue
    if data.get('config',{}).get('environment_scale',1.)!=args.scale:continue
    if data.get('config',{}).get('inspection_scale',1.)!=args.inspection_scale:continue
    if data.get('mission_seconds',0)<args.minimum_duration:continue
    if data.get('success') and abs(data.get('completed_seconds',0)-data['mission_seconds'])>.02:continue
    if not data.get('plans') or len(data.get('frames',[]))<2:continue
    shutil.copy2(file,out/file.name)
    manifest.append(dict(file=file.name,layout=data['layout'],case=data['case'],method=data['method'],seed=data['seed'],development=args.development))
if args.complete_comparison:
    groups={(r['layout'],r['case'],r['seed']) for r in manifest}
    ready={g for g in groups if {'proposed','blind','blind_filter'} <= {r['method'] for r in manifest if (r['layout'],r['case'],r['seed'])==g}}
    manifest=[r for r in manifest if (r['layout'],r['case'],r['seed']) in ready]
    for group in ready:
        records=[json.loads((out/r['file']).read_text()) for r in manifest if (r['layout'],r['case'],r['seed'])==group]
        reference=records[0]
        for record in records[1:]:
            if record['config']!=reference['config'] or record['mission_seconds']!=reference['mission_seconds']:
                raise ValueError(f'Comparison requires identical experiment settings: {group}')
if manifest:
    subprocess.run([sys.executable,str(root/'scripts/report_european_quarter.py'),'--results',str(root/args.results)],check=True)
    shutil.copy2(root/args.results/'analysis.json',out/'analysis.json')
    if all(json.loads((out/r['file']).read_text()).get('config',{}).get('initialization')=='rrt' for r in manifest):
        subprocess.run([sys.executable,str(root/'scripts/audit_rrt_quarter.py'),'--results',str(root/args.results)],check=True)
        shutil.copy2(root/args.results/'execution_integrity.json',out/'execution_integrity.json')
notice=args.notice or ('Development recordings: MBD trajectory optimization.' if manifest else 'MBD trajectory optimization: numerical checks are in progress. No validated inter-region mission is available yet.')
(out/'manifest.json').write_text(json.dumps(dict(asset_prefix=args.asset_prefix,runs=manifest,analysis='analysis.json' if manifest else None,archived=False,notice=notice),indent=2))
print(f'{len(manifest)} completed records published from {args.results}')
