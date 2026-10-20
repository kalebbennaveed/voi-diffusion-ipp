"""Controlled nonlinear-plant test: identical reference speed, three winds."""
import json
from pathlib import Path
import numpy as np
from voi_diffusion_ipp.quadrotor.config import VehicleConfig,ControllerConfig
from voi_diffusion_ipp.quadrotor.controller import GeometricController
from voi_diffusion_ipp.quadrotor.dynamics import Quadrotor
from voi_diffusion_ipp.quadrotor.types import State
from voi_diffusion_ipp.quadrotor.reference import flat_reference
from voi_diffusion_ipp.quadrotor.aerodynamics import acceleration,infer_wind

vehicle=VehicleConfig();limit=2.6
controller=GeometricController(vehicle,ControllerConfig(max_horizontal_accel=limit,limit_total_horizontal_force=True))
results={}
for name,w in [('calm',0.),('downwind',3.8),('upwind',-3.8)]:
    wind=np.array([w,0.,0.]);plant=Quadrotor(vehicle,wind,np.zeros(3))
    state=State(np.array([0.,0.,3.]),np.zeros(3),np.eye(3),np.zeros(3),np.full(4,np.sqrt(vehicle.mass*vehicle.gravity/(4*vehicle.thrust_coefficient))))
    frames=[]
    for k in range(3000):
        t=k*.01
        # This is a controlled actuator-authority test, not an inspection policy.
        v=2.;p=np.array([v*t,0.,3.]);vv=np.array([v,0,0])
        ref=flat_reference(p,vv,np.zeros(3),np.zeros(3),np.zeros(3),compensation=-acceleration(state.velocity,wind))
        command=controller(state,ref);state=plant.step(state,command,.01)
        if k%10==0:frames.append(dict(t=t,p=state.position.tolist(),v=state.velocity.tolist(),tilt_deg=float(np.degrees(np.arccos(np.clip(state.rotation[2,2],-1,1))))))
    speed=float(np.mean([f['v'][0] for f in frames if f['t']>=25]))
    results[name]=dict(mean_final_ground_speed=speed,wind_speed=w,requested_ground_speed=2.,frames=frames)
    print(name,round(speed,3),'m/s',flush=True)
assert results['upwind']['mean_final_ground_speed']>0
assert results['upwind']['mean_final_ground_speed']<.35*results['calm']['mean_final_ground_speed']
assert abs(results['downwind']['mean_final_ground_speed']-2)<.05
expected=np.sqrt(vehicle.mass*limit/vehicle.linear_drag)-3.8
assert abs(results['upwind']['mean_final_ground_speed']-expected)<.06
results['explanation']=dict(upwind_terminal_speed_theory=expected,thrust_acceleration_bound=limit,
    drag_coefficient=vehicle.linear_drag,mass=vehicle.mass,known_wind_in_this_diagnostic=True)
out=Path('output/urban_quad_airflow');out.mkdir(exist_ok=True,parents=True)
(out/'airspeed_validation.json').write_text(json.dumps(results,indent=2))
