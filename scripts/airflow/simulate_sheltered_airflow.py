"""Sheltered monitoring-region redesign using the original 3-D airflow solver.

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


def build_flow(nx, scenario, central_peak=24., lower_peak=18., port_z=(.25,1.25), central_z=(1.5,2.), smag=.16, central_x=6.8, floor_outlet=False, floor_box=(3.,4.45,.2,1.55), south_outlet=False, central_direction_override=None, additional_floor_box=None,lower_direction_override=None):
    dx=WIDTH/nx
    ny,nz=round(HEIGHT/dx),round(DEPTH/dx)
    if abs(ny*dx-HEIGHT)>1e-10 or abs(nz*dx-DEPTH)>1e-10:
        raise ValueError('Use nx multiples of 64 to preserve cubic cells and physical bounds.')
    # Same numerical lattice speed and physical conditions at every resolution.
    dt=.045*dx/max(8.,central_peak,lower_peak if scenario==2 else 0.)
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
    for direction in ['N','S','T','B','W','E']:
        bh.set_boundary(wall,slice_from_direction(direction,3))
    # Elevated north inflow aims through the central gap and toward the south
    # outlet. It enters above the monitored plane, clearing the lower approach.
    central_direction=np.array([-2.2575+(6.8-central_x),-2.7275,-1.] if central_direction_override is None else central_direction_override,dtype=float);central_direction/=np.linalg.norm(central_direction)
    inlet_left,inlet_right=(6.4,7.2) if central_x==6.8 else (central_x-.4,central_x+.4)
    lower_direction=np.array([-1.,-.60,0.] if lower_direction_override is None else lower_direction_override,dtype=float);lower_direction/=np.linalg.norm(lower_direction)
    def port_profile(coordinate,lo,hi):
        phase=np.clip((coordinate-lo)/(hi-lo),0.,1.)
        return np.where((coordinate>lo)&(coordinate<hi),np.sin(np.pi*phase)**2,0.)
    def central_inlet(data,**kwargs):
        x=(data.index_array['x']+data.offset[0]+.5+.5*data.link_offsets()[:,0])*dx
        z=(data.index_array['z']+data.offset[2]+.5+.5*data.link_offsets()[:,2])*dx
        speed=central_peak*port_profile(x,inlet_left,inlet_right)*port_profile(z,*central_z)*dt/dx
        for i in range(3):data[f'vel_{i}']=speed*central_direction[i]
    bh.set_boundary(UBB(central_inlet,dim=3,name='Elevated north central wind inlet'),slice_from_direction('N',3),
                    mask_callback=lambda x,y,z:(x*dx>inlet_left)&(x*dx<inlet_right)&(z*dx>central_z[0])&(z*dx<central_z[1]))
    if scenario==2:
        def lower_inlet(data,**kwargs):
            y=(data.index_array['y']+data.offset[1]+.5+.5*data.link_offsets()[:,1])*dx
            z=(data.index_array['z']+data.offset[2]+.5+.5*data.link_offsets()[:,2])*dx
            speed=lower_peak*port_profile(y,.35,1.15)*port_profile(z,*port_z)*dt/dx
            for i in range(3):data[f'vel_{i}']=speed*lower_direction[i]
        bh.set_boundary(UBB(lower_inlet,dim=3,name='East lower wind inlet'),slice_from_direction('E',3),
                        mask_callback=lambda x,y,z:(y*dx>.35)&(y*dx<1.15)&(z*dx>port_z[0])&(z*dx<port_z[1]))
    # Physical outlet opening on the south perimeter, entirely outside B.
    # All other perimeter surfaces are solid; no wind values are clipped in ROIs.
    if floor_outlet:
        bh.set_boundary(FixedDensity(1.,name='Floor pressure outlet outside B'),slice_from_direction('B',3),
                        mask_callback=lambda x,y,z:(x*dx>floor_box[0])&(x*dx<floor_box[1])&(y*dx>floor_box[2])&(y*dx<floor_box[3]))
    if not floor_outlet or south_outlet:
        bh.set_boundary(FixedDensity(1.,name='South pressure outlet outside B'),slice_from_direction('S',3),
                        mask_callback=lambda x,y,z:(x*dx>2.95)&(x*dx<4.45))
    if additional_floor_box is not None:
        box=additional_floor_box
        bh.set_boundary(FixedDensity(1.,name='Lower floor pressure outlet outside B'),slice_from_direction('B',3),
                        mask_callback=lambda x,y,z:(x*dx>box[0])&(x*dx<box[1])&(y*dx>box[2])&(y*dx<box[3]))
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
    return flow,dx,dt,inside,q_stats


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    flow,dx,dt,inside,q_stats=build_flow(args.nx,args.scenario,args.central_peak,args.lower_peak,(args.port_z_min,args.port_z_max),(args.central_z_min,args.central_z_max),central_x=args.central_x,floor_outlet=args.floor_outlet,floor_box=args.floor_box,south_outlet=args.south_outlet)
    nx,ny,nz=flow.domain_size
    grid=np.meshgrid(np.arange(nx)+.5,np.arange(ny)+.5,np.arange(nz)+.5,indexing='ij')
    solid=inside(*grid)
    z_float=args.slice_height/dx-.5;z_lo=int(np.floor(z_float));z_weight=z_float-z_lo;z_actual=args.slice_height
    if not (0<=z_lo<nz-1):raise ValueError('Slice height must lie between interior cell centers.')
    sample_steps=max(1,round(args.sample/dt));total_steps=round(args.seconds/dt)
    fields=[];times=[];diagnostics=[];start=time.perf_counter();preview_written=False
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
        if not preview_written and times[-1]>=16.:
            # Read-only midpoint screening: saving this does not change the flow.
            np.savez_compressed(out/'midpoint.npz',fields=np.stack(fields),times=times,dx=dx,dt=dt)
            preview_written=True
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
        'wind_source':f'Elevated north central opening x={args.central_x-.4:g}–{args.central_x+.4:g} m, z={args.central_z_min:g}–{args.central_z_max:g} m, peak {args.central_peak:g} m/s, direction ({4.5425-args.central_x:g},-2.7275,-1) normalized. It aims from ({args.central_x:g},4.625,1.75) through the central gap (4.5425,1.8975,0.75), toward the south outlet outside B. Scenario 2 also has an east lower opening y=0.35–1.15 m, z={args.port_z_min:g}–{args.port_z_max:g} m, peak {args.lower_peak:g} m/s, direction (-1,-0.6,0) normalized. Compact sin^2 profiles across each opening. Interior velocity is solved by CFD; no ROI masking or scaling.',
        'boundary_conditions':'No-slip floor, roof, and perimeter except the elevated north central inlet, Scenario 2 east lower inlet, and south pressure-outlet opening x=2.95–4.45 m outside B. The same circular columns use analytic surface intersections, with recorded halfway fallback at stencil junctions.',
        'initial_condition':'Air initially at rest; imposed inlets start at t=0','environment_revision':'directional-sheltered-ports-v8' if args.central_x==6.8 else 'directional-sheltered-ports-v9',
        'openings':{'central_inlet':{'side':'north','x_m':[args.central_x-.4,args.central_x+.4],'z_m':[args.central_z_min,args.central_z_max],'peak_m_s':args.central_peak,'direction':[-2.2575+(6.8-args.central_x),-2.7275,-1.]},'lower_inlet':{'side':'east','y_m':[.35,1.15],'z_m':[args.port_z_min,args.port_z_max],'peak_m_s':args.lower_peak if args.scenario==2 else 0.,'direction':[-1.,-.6,0.]},'outlet':{'side':'south','x_m':[2.95,4.45]}},
        'slice_height_m':z_actual,'requested_slice_height_m':args.slice_height,'slice_interpolation':'Linear interpolation between neighboring vertical CFD cell centers','times_s':times,
        'data_shape':list(values.shape),'data_file':'velocity.f32','data_dtype':'little-endian float32','data_axes':['time','y from south','x from west','u/v/w'],
        'diagnostics':diagnostics,'curved_boundary_checks':q_stats,'wall_seconds':time.perf_counter()-start,
        'velocity_sha256':hashlib.sha256(values.tobytes()).hexdigest(),
        'limitations':['LES models unresolved eddies; not DNS or an approximation-free or measured wind field.','Finite spatial/time discretization; near-wall turbulence is not wall-resolved at these grids.','Quiet zones and route impassability are design targets, not imposed solution properties.','A horizontal velocity slice is exported for animated streamlets; these are not three-dimensional particle trajectories.','Air velocity is not assigned directly to the robot additive-acceleration disturbance. Robot coupling remains separate.']}
    if args.floor_outlet:
        metadata['environment_revision']='directional-sheltered-ports-v11' if list(args.floor_box)!=[3.,4.45,.2,1.55] or args.south_outlet else 'directional-sheltered-ports-v10'
        metadata['openings']['outlet']={'side':'floor','x_m':list(args.floor_box[:2]),'y_m':list(args.floor_box[2:]),'z_m':0.}
        if args.south_outlet:metadata['openings']['lower_outlet']={'side':'south','x_m':[2.95,4.45]}
        metadata['wind_source']=metadata['wind_source'].replace('toward the south outlet outside B','toward the floor outlet outside B')
        metadata['boundary_conditions']=f'No-slip roof, floor and perimeter except the elevated north inlet, Scenario 2 east lower inlet, and floor pressure outlet x={args.floor_box[0]:g}–{args.floor_box[1]:g} m, y={args.floor_box[2]:g}–{args.floor_box[3]:g} m at z=0, outside B and below the robot plane. '+('Additional south outlet x=2.95–4.45 m for the lower flow.' if args.south_outlet else 'The south perimeter is closed.')+' Same analytic circular-column boundaries and recorded stencil-junction fallback.'
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({'event':'complete','output':str(out),'frames':len(times),'seconds':metadata['wall_seconds']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--nx',type=int,default=128);parser.add_argument('--scenario',type=int,choices=[1,2],required=True);parser.add_argument('--seconds',type=float,default=24.);parser.add_argument('--sample',type=float,default=.2);parser.add_argument('--slice-height',type=float,default=.75);parser.add_argument('--output',required=True)
    parser.add_argument('--central-peak',type=float,default=24.);parser.add_argument('--lower-peak',type=float,default=18.)
    parser.add_argument('--port-z-min',type=float,default=.25);parser.add_argument('--port-z-max',type=float,default=1.25)
    parser.add_argument('--central-x',type=float,default=6.8)
    parser.add_argument('--floor-outlet',action='store_true')
    parser.add_argument('--floor-box',type=float,nargs=4,default=[3.,4.45,.2,1.55])
    parser.add_argument('--south-outlet',action='store_true')
    parser.add_argument('--central-z-min',type=float,default=1.5);parser.add_argument('--central-z-max',type=float,default=2.)
    run(parser.parse_args())
