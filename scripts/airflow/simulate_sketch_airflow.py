"""3-D, time-dependent air flow for the user's three-cylinder sketch.

D3Q19 MRT lattice Boltzmann + Smagorinsky LES, using lbmpy 1.4.1.
This is a numerical low-Mach CFD model, not DNS or measured/approximation-free air.
Geometry uses analytic cylinder/link intersections for Bouzidi no-slip boundaries.
No route labels, robot controller, or monitoring cost enter the flow solver.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('PYSTENCILS_CACHE_DIR','/private/tmp/airflow-pystencils')
os.environ.setdefault('PYSTENCILS_CONFIG','/private/tmp/airflow-pystencils.json')
import numpy as np
import pystencils as ps
from lbmpy import LBMConfig
from lbmpy.enums import Stencil, Method, SubgridScaleModel
from lbmpy.lbstep import LatticeBoltzmannStep
from lbmpy.boundaries import NoSlip, NoSlipLinearBouzidi, UBB, FixedDensity
from pystencils.slicing import slice_from_direction

WIDTH, HEIGHT, DEPTH = 8., 4.625, 2.
CYLINDERS = [(2.16,3.3125,.98,1.5),(3.73,2.6325,.855,1.5),(5.355,1.1625,.855,1.5)]
RHO = 1.204
MU = 1.458e-6*293.15**1.5/(293.15+110.4)
NU = MU/RHO


def build_flow(nx, scenario, speed_scale=1., smag=.16):
    dx=WIDTH/nx
    ny,nz=round(HEIGHT/dx),round(DEPTH/dx)
    if abs(ny*dx-HEIGHT)>1e-10 or abs(nz*dx-DEPTH)>1e-10:
        raise ValueError('Use nx multiples of 64 to preserve cubic cells and physical bounds.')
    # Same numerical lattice speed and physical conditions at every resolution.
    dt=.045*dx/(8.*speed_scale)
    omega=1/(.5+3*NU*dt/dx**2)
    cfg=LBMConfig(stencil=Stencil.D3Q19,method=Method.MRT,relaxation_rate=omega,
                  compressible=True,zero_centered=True,
                  subgrid_scale_model=(SubgridScaleModel.SMAGORINSKY,smag))
    start=time.perf_counter()
    flow=LatticeBoltzmannStep(domain_size=(nx,ny,nz),lbm_config=cfg,
        config=ps.CreateKernelConfig(cpu_openmp=int(os.environ.get('AIRFLOW_THREADS','1')) if int(os.environ.get('AIRFLOW_THREADS','1'))>1 else False,cpu_vectorize_info=None),
        name=f'wind{scenario}',compute_velocity_in_every_step=False,
        compute_density_in_every_step=False)
    bh=flow.boundary_handling
    wall=NoSlip('Floor, roof, and side walls')
    for direction in ['N','S','T','B']:
        bh.set_boundary(wall,slice_from_direction(direction,3))
    # External incoming wind profile, not an imposed interior velocity field.
    # The vertical sine profile vanishes at floor/roof. Mean direction SW.
    direction=np.array([-1.,-.60,0.]);direction/=np.linalg.norm(direction)
    def physical_inlet(y,z):
        central=5.*np.exp(-.5*((y-3.65)/.40)**2)
        lower=(8.*np.exp(-.5*((y-.83)/.28)**2)) if scenario==2 else 0.
        vertical=np.maximum(0.,np.sin(np.pi*z/DEPTH))
        return (central+lower)*vertical*speed_scale
    def inlet(data,**kwargs):
        # Direct global coordinates: block.offset already includes ghost layers.
        y=(data.index_array['y']+data.offset[1]+.5+.5*data.link_offsets()[:,1])*dx
        z=(data.index_array['z']+data.offset[2]+.5+.5*data.link_offsets()[:,2])*dx
        speed=physical_inlet(y,z)*dt/dx
        for i in range(3):data[f'vel_{i}']=speed*direction[i]
    bh.set_boundary(UBB(inlet,dim=3,name='External incoming wind'),slice_from_direction('E',3))
    bh.set_boundary(FixedDensity(1.,name='Pressure outlet'),slice_from_direction('W',3))
    q_stats=[]
    def inside(x,y,z):
        x,y,z=x*dx,y*dx,z*dx
        mask=np.zeros(np.broadcast_shapes(x.shape,y.shape,z.shape),dtype=bool)
        for cx,cy,r,height in CYLINDERS:
            mask|=((x-cx)**2+(y-cy)**2<r*r)&(z<height)&(z>=0)
        return mask
    def wall_distances(data,**kwargs):
        p=np.stack([(data.index_array[c]+data.offset[i]+.5)*dx for i,c in enumerate('xyz')],axis=1)
        v=data.link_offsets()*dx
        hit=np.full(len(p),np.inf)
        for cx,cy,r,height in CYLINDERS:
            rx,ry=p[:,0]-cx,p[:,1]-cy
            aa=v[:,0]**2+v[:,1]**2;bb=2*(rx*v[:,0]+ry*v[:,1]);cc=rx*rx+ry*ry-r*r
            disc=bb*bb-4*aa*cc
            q=np.divide(-bb-np.sqrt(np.maximum(disc,0)),2*aa,out=np.full(len(p),np.inf),where=aa>0)
            z=p[:,2]+q*v[:,2]
            valid=(aa>0)&(disc>=0)&(q>=-1e-9)&(q<=1+1e-9)&(z>=-1e-9)&(z<=height+1e-9)
            hit=np.minimum(hit,np.where(valid,q,np.inf))
            qtop=np.divide(height-p[:,2],v[:,2],out=np.full(len(p),-1.),where=v[:,2]!=0)
            xt=p[:,0]+qtop*v[:,0]-cx;yt=p[:,1]+qtop*v[:,1]-cy
            valid=(v[:,2]!=0)&(qtop>=0)&(qtop<=1)&(xt*xt+yt*yt<=r*r+1e-9)
            hit=np.minimum(hit,np.where(valid,qtop,np.inf))
        if not np.isfinite(hit).all():
            print({'bad_p':p[~np.isfinite(hit)][:8].tolist(),'bad_v':v[~np.isfinite(hit)][:8].tolist(),'total_links':len(p),'offset':data.offset,'ghost':data.ghost_layers},flush=True)
            raise RuntimeError(f'Unresolved analytic wall links: {np.sum(~np.isfinite(hit))}')
        # Bouzidi's q < 1/2 branch needs a second fluid node away from the wall.
        # Use the library-documented no-slip fallback at junctions where that
        # node is unavailable. Record every fallback explicitly: these links
        # use halfway bounce-back, not the interpolated cylinder distance.
        upstream=p-v
        upstream_solid=inside(*(upstream[:,i]/dx for i in range(3)))
        upstream_outside=np.any((upstream<0)|(upstream>np.array([WIDTH,HEIGHT,DEPTH])),axis=1)
        invalid=(hit<.5)&(upstream_solid|upstream_outside)
        data['q']=np.where(invalid,-1.,np.clip(hit,1e-10,1.))
        q_stats.append({'links':len(hit),'min_q':float(hit.min()),'max_q':float(hit.max()),'unresolved':0,'junction_halfway_no_slip_links':int(np.sum(invalid))})
    bh.set_boundary(NoSlipLinearBouzidi(name='Exact cylinder surface intersections',init_wall_distance=wall_distances),mask_callback=inside)
    flow.set_pdf_fields_from_macroscopic_values()
    print(json.dumps({'event':'built','shape':[nx,ny,nz],'dx_m':dx,'dt_s':dt,'nu_m2_s':NU,'omega':omega,'seconds':time.perf_counter()-start}),flush=True)
    return flow,dx,dt,inside,q_stats,physical_inlet


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    flow,dx,dt,inside,q_stats,inlet=build_flow(args.nx,args.scenario)
    nx,ny,nz=flow.domain_size
    grid=np.meshgrid(np.arange(nx)+.5,np.arange(ny)+.5,np.arange(nz)+.5,indexing='ij')
    solid=inside(*grid)
    z_float=args.slice_height/dx-.5;z_lo=int(np.floor(z_float));z_weight=z_float-z_lo;z_actual=args.slice_height
    if not (0<=z_lo<nz-1):raise ValueError('Slice height must lie between interior cell centers.')
    sample_steps=max(1,round(args.sample/dt));total_steps=round(args.seconds/dt)
    fields=[];times=[];diagnostics=[];start=time.perf_counter()
    actual_steps=0
    while actual_steps<total_steps:
        n=min(sample_steps,total_steps-actual_steps);flow.run(n);actual_steps+=n
        velocity=np.asarray(flow.velocity_slice((slice(None),)*3,masked=False))*dx/dt
        density=np.asarray(flow.density_slice((slice(None),)*3,masked=False))
        velocity[solid]=0
        speed=np.linalg.norm(velocity,axis=-1)
        if not np.isfinite(velocity).all() or not np.isfinite(density).all():raise RuntimeError('Nonfinite CFD solution')
        max_lattice=float(speed.max()*dt/dx)
        rho_delta=float(np.max(np.abs(density[~solid]-1)))
        if max_lattice>.20 or rho_delta>.08:raise RuntimeError(f'CFD stability gate failed: u_lattice={max_lattice}, density_delta={rho_delta}')
        plane=((1-z_weight)*velocity[:,:,z_lo,:]+z_weight*velocity[:,:,z_lo+1,:]).transpose(1,0,2).copy().astype('<f4')
        fields.append(plane);times.append(actual_steps*dt)
        if len(times)%10==0 or actual_steps==total_steps:
            diag={'time_s':times[-1],'max_speed_m_s':float(speed.max()),'max_lattice_mach':max_lattice*np.sqrt(3),'max_relative_density_change':rho_delta,'mean_speed_m_s':float(speed[~solid].mean()),'wall_seconds':time.perf_counter()-start}
            diagnostics.append(diag);print(json.dumps(diag),flush=True)
    values=np.stack(fields)
    (out/'velocity.f32').write_bytes(values.tobytes())
    np.savez_compressed(out/'final_volume.npz',velocity=velocity,density=density,solid=solid,dx=dx,dt=dt)
    metadata={'scenario':args.scenario,'solver':'lbmpy 1.4.1 / D3Q19 MRT / Smagorinsky LES Cs=0.16',
        'equations':'3D isothermal low-Mach filtered Navier-Stokes; molecular plus Smagorinsky subgrid viscosity',
        'units':'m, s, m/s','domain_m':[WIDTH,HEIGHT,DEPTH],'obstacles':CYLINDERS,
        'grid':[nx,ny,nz],'dx_m':dx,'dt_s':dt,'density_kg_m3':RHO,'temperature_K':293.15,'dynamic_viscosity_Pa_s':MU,'kinematic_viscosity_m2_s':NU,
        'wind_source':'Prescribed incoming external wind at east boundary; central 5 m/s peak and scenario 2 lower 8 m/s peak. Direction (-1,-0.6,0), normalized. Gaussian inlet profiles in y only; sine profile in z. Interior velocities computed by CFD.',
        'boundary_conditions':'No-slip floor, roof, north/south boundaries, and circular cylinders with analytic link intersections. Library-documented halfway no-slip fallback at junctions lacking a second fluid node (counts recorded). Pressure outlet west; prescribed velocity inlet east.',
        'initial_condition':'Air initially at rest; imposed inlet starts at t=0',
        'slice_height_m':z_actual,'requested_slice_height_m':args.slice_height,'slice_interpolation':'Linear interpolation between neighboring vertical CFD cell centers','times_s':times,
        'data_shape':list(values.shape),'data_file':'velocity.f32','data_dtype':'little-endian float32','data_axes':['time','y from south','x from west','u/v/w'],
        'diagnostics':diagnostics,'curved_boundary_checks':q_stats,'wall_seconds':time.perf_counter()-start,
        'velocity_sha256':hashlib.sha256(values.tobytes()).hexdigest(),
        'limitations':['LES models unresolved eddies; not DNS or an approximation-free or measured wind field.','Finite spatial/time discretization; near-wall turbulence is not wall-resolved at these grids.','Quiet zones and route impassability are design targets, not imposed solution properties.','A horizontal velocity slice is exported for animated streamlets; these are not three-dimensional particle trajectories.','Air velocity is not assigned directly to the robot additive-acceleration disturbance. Robot coupling remains separate.']}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({'event':'complete','output':str(out),'frames':len(times),'seconds':metadata['wall_seconds']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--nx',type=int,default=128);parser.add_argument('--scenario',type=int,choices=[1,2],required=True);parser.add_argument('--seconds',type=float,default=20.);parser.add_argument('--sample',type=float,default=.2);parser.add_argument('--slice-height',type=float,default=.75);parser.add_argument('--output',required=True)
    run(parser.parse_args())
