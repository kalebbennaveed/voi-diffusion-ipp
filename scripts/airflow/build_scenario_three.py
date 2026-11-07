"""Same 3-D CFD and stationary-mean definition, new physical port placement."""
import argparse,hashlib,json,signal
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from functools import partial
import simulate_sheltered_airflow as solver
from export_stationary_wind import mean_between

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--spec',type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.spec.read_text())
    if a.output.exists() and any(a.output.iterdir()):raise RuntimeError('Preserve existing CFD data')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded 1800-second CFD budget')));signal.alarm(1800)
    solver.WIDTH,solver.HEIGHT=spec['bounds'][1];solver.DEPTH=spec['depth']
    solver.CYLINDERS=[[*o,spec['obstacle_height']] for o in spec['obstacles']]
    solver.build_flow=partial(solver.build_flow,lower_direction_override=[-.8,0.,-1.])
    args=SimpleNamespace(nx=64,scenario=2,seconds=32.,sample=.1,slice_height=.75,
        central_peak=0.,lower_peak=18.,port_z_min=.25,port_z_max=1.25,
        central_z_min=1.5,central_z_max=2.,central_x=6.,floor_outlet=True,
        floor_box=[6.9,7.9,.2,1.3],south_outlet=False,output=str(a.output/'transient'))
    solver.run(args)
    src=a.output/'transient';m=json.loads((src/'metadata.json').read_text())
    values=np.fromfile(src/'velocity.f32',dtype='<f4').reshape(m['data_shape'])
    mean=mean_between(m,values,16,32).astype('<f4')
    first=mean_between(m,values,16,24);second=mean_between(m,values,24,32)
    xx,yy=np.meshgrid((np.arange(m['grid'][0])+.5)*m['dx_m'],(np.arange(m['grid'][1])+.5)*m['dx_m'])
    free=np.ones(xx.shape,bool)
    for x,y,r in spec['obstacles']:free&=(xx-x)**2+(yy-y)**2>r*r
    speed=np.linalg.norm(mean[...,:2],axis=-1)
    def stats(mask):
        z=speed[mask&free];return dict(mean_m_s=float(z.mean()),max_m_s=float(z.max()),p95_m_s=float(np.quantile(z,.95)))
    regions={r['label']:stats((xx>=r['bounds'][0][0])&(xx<=r['bounds'][1][0])&(yy>=r['bounds'][0][1])&(yy<=r['bounds'][1][1])) for r in spec['regions']}
    out=a.output/'data/scenario3';out.mkdir(parents=True)
    metadata=dict(m,time_dependence='none',scenario=3,environment_spec=spec,regions=spec['regions'],
        field_definition='Fixed 16–32 s vector mean of the existing 3-D D3Q19 MRT / Smagorinsky CFD model',
        data_file='stationary-velocity.f32',data_shape=list(mean.shape),data_axes=['y from south','x from west','u/v/w'],
        averaging_interval_s=[16,32],environment_revision=spec['revision'],
        velocity_sha256=hashlib.sha256(mean.tobytes()).hexdigest(),monitoring_region_metrics=regions,
        monitoring_band_metrics=stats(yy>=2.8),lower_right_metrics=stats((xx>=4.8)&(yy<2.8)),
        sampling_vector_relative_L2=float(np.linalg.norm(first-second)/max(np.linalg.norm(second),1e-12)),
        wind_source='East lower inlet: 18 m/s peak, y=0.35–1.15 m, z=0.25–1.25 m, direction (-0.8,0,-1). No north inflow. Adjacent floor pressure outlet x=6.9–7.9 m, y=0.2–1.3 m.',
        boundary_conditions='No-slip perimeter, roof, floor and circular column, except declared east inlet and floor pressure outlet.',
        prior_information='No wind-support region or true field supplied to POAM, acquisition, handoff or controller.',
        stationary_model_limit='Numerical stationary mean-flow scenario; not a measured field or grid-convergence certificate. Calm region is a CFD result, not an imposed velocity mask.')
    metadata.pop('times_s',None)
    metadata['openings']['lower_inlet']['direction']=[-.8,0.,-1.]
    (out/metadata['data_file']).write_bytes(mean.tobytes());(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(dict(event='stationary_complete',regions=regions,monitoring=metadata['monitoring_band_metrics'],high_wind=metadata['lower_right_metrics'],wall_seconds=m['wall_seconds'])),flush=True)
if __name__=='__main__':main()
