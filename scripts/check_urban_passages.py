"""Known-wind controller diagnostic; not a result for any planning method."""
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from voi_diffusion_ipp.experiments.urban_airflow import wind_velocity,environment,REVISION
from voi_diffusion_ipp.quadrotor.config import VehicleConfig
from voi_diffusion_ipp.quadrotor.execution import executor
from voi_diffusion_ipp.quadrotor.reference import QuinticReference
from voi_diffusion_ipp.quadrotor.types import State
from voi_diffusion_ipp.wind.cache import TableMean,TableBasis
from run_urban_quad_airflow import clearance

vehicle=VehicleConfig();summary={};gridx,gridy=np.meshgrid(np.arange(-2,43,.25),np.arange(-2,31,.25));grid=np.c_[gridx.ravel(),gridy.ravel()]
for name,case,y,start,end in [('A_downwind','forward',9.,11.,31.),('A_upwind','forward',9.,31.,11.),('A_adverse','crosswind',9.,11.,31.),('C_adverse','crosswind',1.3,11.,31.),('B_calm','crosswind',25.,11.,31.),('A_strong_upwind','commitment',9.,31.,11.)]:
    p0=np.array([start,y,3.]);p1=np.array([end,y,3.]);zero=np.zeros(3)
    reference=QuinticReference(p0,zero,zero,p1,zero,zero,18.)
    state=State(p0,zero,np.eye(3),zero,np.full(4,np.sqrt(vehicle.mass*vehicle.gravity/(4*vehicle.thrust_coefficient))))
    polynomials=[]
    segments=120 if case=='commitment' else 40
    for k in range(segments):
        a=min(2*k,18.);b=min(2*k+2,18.)
        pp,vv,aa,_,_=reference.at(a);pe,ve,ae,_,_=reference.at(b)
        polynomials.append(QuinticReference(pp,vv,aa,pe,ve,ae,2.).coefficients)
    field=wind_velocity(jnp.array(grid),case).reshape(*gridx.shape,2)
    mean=TableMean(TableBasis(jnp.array([-2.,-2.]),jnp.array([.25,.25])),field)
    execute=executor(lambda p:wind_velocity(p,case),2.,segments,max_horizontal_accel=2.6,wind_velocity=True)
    _,logs=execute(jnp.array(state.pack(),jnp.float32),jnp.array(np.array(polynomials),jnp.float32),mean)
    states=np.asarray(logs[0]);position=states[:,:3];velocity=states[:,3:6]
    arrival=np.flatnonzero(np.linalg.norm(position-p1,axis=1)<.5)
    core=(position[:,0]>16)&(position[:,0]<26)&(np.sign(end-start)*velocity[:,0]>.05)
    speed=float(np.median(np.abs(velocity[core,0]))) if core.any() else None
    rect=np.asarray(environment(case).rectangles);gap=min(clearance(p,rect)-.2 for p in position)
    row=dict(case=case,passage=name,known_wind_diagnostic=True,arrival_seconds=float((arrival[0]+1)*.01) if len(arrival) else None,min_clearance=float(gap),median_core_ground_speed=speed,final_position=position[-1].tolist())
    summary[name]=row;print(name,row,flush=True)
    assert arrival.size and gap>0,(name,row)
assert summary['A_upwind']['median_core_ground_speed']<.6*summary['B_calm']['median_core_ground_speed']
out=Path('output/urban_quad_airflow/passage_validation.json');out.write_text(json.dumps(dict(revision=REVISION,results=summary),indent=2))
