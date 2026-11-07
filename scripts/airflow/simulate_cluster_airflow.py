"""New sketch geometry with the same full 3D solver and wall construction."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import signal
import numpy as np
import simulate_sheltered_airflow as solver

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--scenario',type=int,choices=(1,2),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--wall-limit',type=int,default=1000)
    a=p.parse_args();spec=json.loads(a.spec.read_text());s=a.scenario-1
    if (a.output/'metadata.json').exists():raise RuntimeError('Refusing to overwrite an existing CFD run')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CFD wall-time limit')));signal.alarm(a.wall_limit)
    solver.WIDTH,solver.HEIGHT=spec['bounds'][1];solver.DEPTH=spec['depth']
    solver.CYLINDERS=[[*o,spec['obstacle_height']] for o in spec['obstacles']]
    inlet=np.array([spec['central_inlet_x'][s],solver.HEIGHT,1.75]);direction=np.array(spec['central_target'])-inlet
    lower_box=spec.get('lower_floor_box',[None,None])[s]
    solver.build_flow=partial(solver.build_flow,central_direction_override=direction,additional_floor_box=lower_box)
    args=argparse.Namespace(nx=64,scenario=a.scenario,seconds=32.,sample=.1,slice_height=.75,
        central_peak=spec['central_peak'][s],lower_peak=spec['lower_peak'][s],port_z_min=.25,port_z_max=1.25,
        central_z_min=1.5,central_z_max=2.,central_x=inlet[0],floor_outlet=True,floor_box=spec['floor_box'],
        south_outlet=spec['south_outlet'][s],output=str(a.output))
    solver.run(args)
    meta=json.loads((a.output/'metadata.json').read_text())
    meta.update(environment_revision=spec['revision'],regions=spec['regions'],environment_spec=spec,
        environment_spec_sha256=hashlib.sha256(a.spec.read_bytes()).hexdigest())
    meta['openings']['central_inlet']['direction']=direction.tolist()
    if lower_box is not None:
        meta['openings']['lower_floor_outlet']=dict(side='floor',x_m=lower_box[:2],y_m=lower_box[2:],z_m=0.)
        meta['boundary_conditions']+=f' Additional floor pressure opening x={lower_box[:2]}, y={lower_box[2:]} below the lower passage.'
    meta['wind_source']=f"Elevated north inlet centered at {inlet.tolist()} m, peak {args.central_peak:g} m/s, aimed toward {spec['central_target']} m, direction {direction.tolist()} normalized. Compact sin-squared port profile. East lower inlet peak {args.lower_peak:g} m/s in Scenario {a.scenario}. Full 3D CFD determines interior wind; no velocity masking, scaling, or route-dependent force."
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2));signal.alarm(0)

if __name__=='__main__':main()
