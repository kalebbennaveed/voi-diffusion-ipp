"""Export the unchanged vector averaging rule for the new 3D geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from export_stationary_wind import mean_between

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--site',type=Path,required=True)
    a=p.parse_args();reports=[]
    for s in (1,2):
        folder=a.directory/f'cfd-s{s}';meta=json.loads((folder/'metadata.json').read_text());data=(folder/'velocity.f32').read_bytes()
        assert hashlib.sha256(data).hexdigest()==meta['velocity_sha256']
        raw=np.frombuffer(data,dtype='<f4').reshape(meta['data_shape']);mean=mean_between(meta,raw,16.,32.).astype('<f4')
        halves=[mean_between(meta,raw,start,end) for start,end in [(16.,24.),(24.,32.)]]
        nx,ny,_=meta['grid'];dx=meta['dx_m'];xx,yy=np.meshgrid((np.arange(nx)+.5)*dx,(np.arange(ny)+.5)*dx)
        speed=np.linalg.norm(mean[...,:2],axis=-1);regions={}
        for r in meta['regions']:
            lo,hi=r['bounds'];mask=(xx>=lo[0])&(xx<=hi[0])&(yy>=lo[1])&(yy<=hi[1]);z=speed[mask]
            regions[r['label']]=dict(mean_m_s=float(z.mean()),p95_m_s=float(np.quantile(z,.95)),max_m_s=float(z.max()))
        report=dict(scenario=s,regions=regions,sampling_vector_relative_L2=float(np.linalg.norm(halves[1]-halves[0])/np.linalg.norm(halves[1])),
                    max_density_change=max(d['max_relative_density_change'] for d in meta['diagnostics']),max_lattice_mach=max(d['max_lattice_mach'] for d in meta['diagnostics']))
        for key in ['times_s','data_shape','data_file','data_axes','velocity_sha256']:meta.pop(key)
        meta.update(time_dependence='none',data_shape=list(mean.shape),data_file='stationary-velocity.f32',data_axes=['y from south','x from west','u/v/w'],
            field_definition='Fixed vector mean over 16–32 s from the full 3D CFD solve',averaging_interval_s=[16.,32.],
            velocity_sha256=hashlib.sha256(mean.tobytes()).hexdigest(),monitoring_region_metrics=regions,
            validation_status='Provisional numerical environment; full-scene grid convergence and experimental accuracy not established.')
        out=a.site/f'data/scenario{s}';out.mkdir(parents=True,exist_ok=True)
        (out/'stationary-velocity.f32').write_bytes(mean.tobytes());(out/'metadata.json').write_text(json.dumps(meta,indent=2));reports.append(report)
    (a.site/'data/wind-audit.json').write_text(json.dumps(dict(scenarios=reports,interior_field_modification='None'),indent=2))
    print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
