"""Package recorded CFD fields and report, rather than hide, grid sensitivity."""
import json,hashlib,shutil,zipfile
from pathlib import Path
import numpy as np
root=Path('voi-diffusion-ipp/results/airflow');web=Path('sites/belief-adaptive-monitoring/dist/data');web.mkdir(exist_ok=True)
check=json.loads((root/'solver_verification.json').read_text())
rows=[]
for scenario in (1,2):
    stats=[]
    for n in (64,128):
        p=root/f'scenario{scenario}_n{n}';m=json.loads((p/'metadata.json').read_text());v=np.fromfile(p/'velocity.f32',dtype='<f4').reshape(m['data_shape'])
        xx,yy=np.meshgrid((np.arange(m['grid'][0])+.5)*m['dx_m'],(np.arange(m['grid'][1])+.5)*m['dx_m'])
        valid=np.ones_like(xx,dtype=bool)
        for x,y,r,h in m['obstacles']:valid&=(xx-x)**2+(yy-y)**2>r*r
        dx,dy=xx-4.5425,yy-1.8975
        along=-.671*dx-.741*dy;across=.741*dx-.671*dy
        masks={'central_passage':(abs(along)<.65)&(abs(across)<.18)&valid,
               'lower_bypass':(xx>4.7)&(xx<6.1)&(yy>.06)&(yy<.26)&valid,
               'upper_bypass':(xx>1.3)&(xx<3.0)&(yy>4.38)&(yy<4.57)&valid,
               'whole_fluid_slice':valid}
        selected=np.array(m['times_s'])>=2
        speed=np.linalg.norm(v[selected],axis=-1)
        metrics={name:float(speed[:,mask].mean()) for name,mask in masks.items()}
        stats.append({'grid':m['grid'],'slice_height_m':m['slice_height_m'],'mean_speed_m_s_t2_to_t4':metrics})
    changes={k:abs(stats[1]['mean_speed_m_s_t2_to_t4'][k]-stats[0]['mean_speed_m_s_t2_to_t4'][k])/max(stats[1]['mean_speed_m_s_t2_to_t4'][k],.1) for k in stats[0]['mean_speed_m_s_t2_to_t4']}
    rows.append({'scenario':scenario,'meshes':stats,'relative_mesh_changes':changes})
    dest=web/f'scenario{scenario}';dest.mkdir(exist_ok=True)
    for file in ['metadata.json','velocity.f32']:shutil.copy2(root/f'scenario{scenario}_n128'/file,dest/file)
max_change=max(max(r['relative_mesh_changes'].values()) for r in rows)
verdict={'analytic_solver_check':check,'scene_grid_comparison':rows,'comparison_note':'Time average over 2–4 s; horizontal planes at respective nearest cell-center heights (0.8125 and 0.78125 m). This is a sensitivity check, not a rigorous grid-convergence study or turbulence validation.',
    'full_scene_grid_convergence_established':False,'maximum_reported_mean_speed_change':max_change,
    'summary':f"Analytic Taylor–Green check passed: 1.96% → 0.49% relative velocity error with refinement. Scene passage/whole-slice mean speeds changed by up to {100*max_change:.1f}% between the 64- and 128-cell-wide grids. Full scene grid convergence and wall-resolved turbulence are not established."}
(web/'validation.json').write_text(json.dumps(verdict,indent=2));(root/'validation.json').write_text(json.dumps(verdict,indent=2))
source=Path('voi-diffusion-ipp/scripts/airflow')
with zipfile.ZipFile(web/'airflow-source.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(source.glob('*.py')):archive.write(file,file.name)
    archive.writestr('requirements.txt','lbmpy==1.4.1\npystencils==1.4\nnumpy==2.4.6\nscipy==1.17.1\n')
    archive.writestr('REPRODUCE.txt','Run from workspace root using Python 3.11.15 and the pinned packages. Set PYSTENCILS_CACHE_DIR and PYSTENCILS_CONFIG to writable paths. Configure the compiler as clang++ -O3 -DNDEBUG -fPIC -std=c++11, no OpenMP.\nFor each scenario 1 and 2: python simulate_sketch_airflow.py --nx 128 --scenario N --seconds 4 --sample .04 --output OUTPUT_DIR\nA 64-cell run uses the same arguments with --nx 64.\nThe renderer reads the recorded f32 data; it does not run or change the CFD solver.\n')
    archive.writestr('source-sha256.json',json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob('*.py')},indent=2))
print(json.dumps(verdict,indent=2))
