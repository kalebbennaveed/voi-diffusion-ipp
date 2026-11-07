"""Reproducible planar robot review in the unchanged stationary CFD fields.

No planner/estimator is introduced. The exact known-wind diagnostic uses the
repository's quadratic air law and full flow-policy RPCBF without modification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import NamedTuple
import numpy as np
from scipy.interpolate import CubicSpline
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.environment import Environment
from voi_diffusion_ipp.quadrotor.aerodynamics import acceleration, MASS, DRAG
from voi_diffusion_ipp.safety.flow_policy_value import filter_control
from voi_diffusion_ipp.wind.oracle import KnownWind, KnownBasis

ROOT = Path(__file__).resolve().parents[2]
SITE = Path(os.environ.get('ROBOT_REVIEW_SITE', ROOT.parent / 'sites/belief-adaptive-monitoring/dist'))
BOUNDS = np.array([[0., 0.], [8., 4.625]])
OBSTACLES = np.array([[2.16, 3.3125, .98], [3.73, 2.6325, .855], [5.355, 1.1625, .855]])
CFG = Config(backup_steps=8, backup_dt=.15, backup_gain=2., inner_samples=3,
             cbf_alpha=1., umax=1.6, safety_margin=.06, spline='full', linear_drag=0.)
ENV = Environment(jnp.zeros((1, 2)), jnp.zeros(1), jnp.array(OBSTACLES), jnp.array(BOUNDS),
                  jnp.zeros(4), jnp.zeros(4), jnp.zeros(2))
DT = .05
KP, KV = 1.2, 1.8

class FixedWind(NamedTuple):
    grid: jax.Array  # (ny,nx,2), SI horizontal velocity, exact displayed data
    dx: float
    def at(self, locations):
        p = jnp.atleast_2d(locations)
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError("Fixed wind is spatial-only: supply (x, y), without time")
        ny, nx, _ = self.grid.shape
        z = jnp.clip(p / self.dx - .5, jnp.zeros(2), jnp.array([nx-1, ny-1]))
        ij = jnp.minimum(jnp.floor(z).astype(int), jnp.array([nx-2, ny-2]))
        f = z - ij
        i, j = ij[:, 0], ij[:, 1]
        fx, fy = f[:, 0, None], f[:, 1, None]
        return (1-fy)*((1-fx)*self.grid[j,i]+fx*self.grid[j,i+1]) + fy*((1-fx)*self.grid[j+1,i]+fx*self.grid[j+1,i+1])

def load_wind(scenario, site=SITE):
    root = site / f'data/scenario{scenario}'
    meta = json.loads((root/'metadata.json').read_text())
    if meta.get('time_dependence') != 'none':
        raise ValueError('Robot wind must be a fixed spatial field with time_dependence=none')
    nx, ny, _ = meta['grid']
    raw = np.fromfile(root/meta['data_file'], dtype='<f4').reshape(ny, nx, 3)
    return FixedWind(jnp.array(raw[:, :, :2], dtype=jnp.float64), meta['dx_m'])

def clearance(p):
    p = np.atleast_2d(p)
    circles = np.linalg.norm(p[:,None,:]-OBSTACLES[None,:,:2], axis=-1)-OBSTACLES[:,2]
    return np.min(np.c_[circles, p-BOUNDS[0], BOUNDS[1]-p], axis=1)

def route_curve(name, site=SITE):
    # Read the route guides already visible on the page. Smooth geometric joins
    # into a C2 reference, then retime using analytical derivatives, without wind.
    text = (site/'app.js').read_text()
    path = re.search(r"\b"+name+r":'(M [^']+)'", text)[1]
    a = np.array([float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',path)])
    start = a[:2]; pieces=[]
    for segment in a[2:].reshape(-1,3,2):
        b,c,d = segment
        s=np.linspace(0,1,101,endpoint=False)[:,None]
        pieces.append((1-s)**3*start+3*(1-s)**2*s*b+3*(1-s)*s**2*c+s**3*d)
        start=d
    points=np.vstack([*pieces,start]); points[:,0]*=.01; points[:,1]=4.625-points[:,1]*.01
    dist=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    knots=np.linspace(0,1,41)
    p=np.column_stack([np.interp(knots*dist[-1],dist,points[:,k]) for k in range(2)])
    curve=CubicSpline(knots,p)
    return curve

class Reference:
    def __init__(self, route, direction, site=SITE):
        self.curve=route_curve(route,site)
        self.reverse=direction=='B-A'
        self.duration=1.
        _,v,a=self.at(np.linspace(0,1,10001))
        required=max(np.max(np.linalg.norm(v,axis=-1))/1., np.sqrt(np.max(np.abs(a))/.8))
        self.duration=np.ceil(required/DT)*DT
        p,v,a=self.at(np.linspace(0,self.duration,10001))
        self.checks=dict(duration_s=float(self.duration), min_clearance_m=float(clearance(p).min()),
                         max_speed_m_s=float(np.linalg.norm(v,axis=-1).max()), max_accel_component_m_s2=float(np.abs(a).max()))
        assert self.checks['min_clearance_m'] >= CFG.safety_margin
        assert self.checks['max_speed_m_s']<=1.00001 and self.checks['max_accel_component_m_s2']<=.80001
    def at(self,t):
        z=np.clip(np.asarray(t)/self.duration,0,1)
        q=10*z**3-15*z**4+6*z**5
        qd=(30*z**2-60*z**3+30*z**4)/self.duration
        qdd=(60*z-180*z**2+120*z**3)/self.duration**2
        if self.reverse: q,qd,qdd=1-q,-qd,-qdd
        p=self.curve(q);v=self.curve(q,1)*qd[...,None]
        a=self.curve(q,2)*qd[...,None]**2+self.curve(q,1)*qdd[...,None]
        return p,v,a

@jax.jit
def filtered_control(x, nominal, field):
    belief=KnownWind(KnownBasis(field))
    inner=belief.inner_fields(jnp.zeros((3,1,2)))
    return filter_control(x,nominal,belief,inner,ENV,CFG)

def derivative(x,u,field):
    return jnp.r_[x[2:], u+acceleration(x[2:],field.at(x[:2])[0],xp=jnp)]

def rk4(x,u,field,h):
    a=derivative(x,u,field);b=derivative(x+h*a/2,u,field)
    c=derivative(x+h*b/2,u,field);d=derivative(x+h*c,u,field)
    return x+h*(a+2*b+2*c+d)/6

@jax.jit
def microstep(x,u,field,h):
    return rk4(x,u,field,h)

# Swept point contact along each short RK4 displacement. The substep refinement
# check quantifies this discrete event localization; no bounce/reset is invented.
def first_contact(a,b):
    delta=b-a; candidates=[]
    for axis in range(2):
        if delta[axis] < 0 and b[axis] <= BOUNDS[0,axis]: candidates.append((BOUNDS[0,axis]-a[axis])/delta[axis])
        if delta[axis] > 0 and b[axis] >= BOUNDS[1,axis]: candidates.append((BOUNDS[1,axis]-a[axis])/delta[axis])
    aa=float(delta@delta)
    if aa>1e-25:
        for cx,cy,r in OBSTACLES:
            z=a-np.array([cx,cy]);bb=2*float(z@delta);cc=float(z@z-r*r)
            disc=bb*bb-4*aa*cc
            if disc>=0:
                roots=[(-bb-np.sqrt(disc))/(2*aa),(-bb+np.sqrt(disc))/(2*aa)]
                candidates.extend(s for s in roots if 0<=s<=1)
    valid=[s for s in candidates if 0<=s<=1]
    return min(valid) if valid else None

def passage_crossings(samples):
    """Classify actual crossings of the line through columns 2 and 3.

    That line separates the A and B sides. Its three free intervals identify
    the upper bypass, central gap, and lower bypass without a path-error cutoff.
    """
    a=np.asarray(samples);c=OBSTACLES[1,:2];delta=OBSTACLES[2,:2]-c
    distance=np.linalg.norm(delta);normal=delta/distance
    toward_a=np.array([-normal[1],normal[0]])
    signed=(a[:,1:3]-c)@toward_a;events=[]
    for i in np.flatnonzero(signed[:-1]*signed[1:]<0):
        f=-signed[i]/(signed[i+1]-signed[i]);p=(1-f)*a[i,1:3]+f*a[i+1,1:3]
        coordinate=float((p-c)@normal)
        route='upper' if coordinate < -OBSTACLES[1,2] else 'lower' if coordinate > distance+OBSTACLES[2,2] else 'central' if OBSTACLES[1,2] < coordinate < distance-OBSTACLES[2,2] else 'solid interval'
        event=dict(time_s=float((1-f)*a[i,0]+f*a[i+1,0]),position_m=p.tolist(),route=route,direction='B-A' if signed[i+1]>signed[i] else 'A-B')
        if a.shape[1]>=5:
            velocity=(1-f)*a[i,3:5]+f*a[i+1,3:5]
            event.update(speed_m_s=float(np.linalg.norm(velocity)),forward_speed_m_s=float(abs(velocity@toward_a)))
        events.append(event)
    return events

def central_transit(samples, direction):
    """Time to make 1.2 m of forward progress through the central throat.

    Fixed entry/exit planes are 0.6 m either side of the existing passage gate.
    Include pauses/backtracking after entry. A detour or unfinished traversal
    is not reported as slow successful passage.
    """
    a=np.asarray(samples);c=OBSTACLES[1,:2];delta=OBSTACLES[2,:2]-c
    normal=delta/np.linalg.norm(delta);toward_a=np.array([-normal[1],normal[0]])
    progress=(a[:,1:3]-c)@toward_a*(1 if direction=='B-A' else -1)
    times=[]
    for threshold in (-.6,.6):
        matches=np.flatnonzero((progress[:-1]<threshold)&(progress[1:]>=threshold))
        matches=[i for i in matches if not times or a[i+1,0]>times[0]]
        if not matches:return dict(entry_time_s=times[0] if times else None,exit_time_s=None,duration_s=None,mean_forward_speed_m_s=None)
        i=matches[0];f=(threshold-progress[i])/(progress[i+1]-progress[i])
        times.append(float(a[i,0]+f*(a[i+1,0]-a[i,0])))
    events=[e for e in passage_crossings(a) if times[0]<=e['time_s']<=times[1] and e['direction']==direction]
    if not events or any(e['route']!='central' for e in events):return None
    duration=times[1]-times[0]
    return dict(entry_time_s=times[0],exit_time_s=times[1],duration_s=duration,mean_forward_speed_m_s=1.2/duration)

def simulate(ref,field,filtered=False,substeps=5):
    x=np.r_[ref.at(0)[0],0.,0.]
    horizon=round(ref.duration+10,8);count=int(round(horizon/DT))
    samples=[];contact=None;minimum=float(clearance(x[:2])[0]);n_infeasible=0
    max_residual=0.;max_slack=0.;max_u=0.;initial_value=None
    for k in range(count+1):
        t=k*DT;p,v,a=ref.at(t)
        nominal=np.clip(a+KP*(p-x[:2])+KV*(v-x[2:]),-CFG.umax,CFG.umax)
        feasible=True;slack=0.;residual=0.;corr=0.
        if filtered:
            u,info=filtered_control(jnp.array(x),jnp.array(nominal),field)
            u=np.array(u);feasible=bool(info.feasible);slack=float(info.slack);residual=float(info.residual);corr=float(info.correction)
            if initial_value is None:initial_value=np.array(info.value).tolist()
            n_infeasible+=int(not feasible);max_slack=max(max_slack,slack);max_residual=max(max_residual,residual)
        else:u=nominal
        wind=np.array(field.at(x[:2])[0]);external=acceleration(x[2:],wind)
        max_u=max(max_u,float(np.max(np.abs(u))))
        # t, px,py,vx,vy, windx,windy, ax_air,ay_air, ux,uy, error,clearance,correction,feasible,slack,residual
        samples.append([t,*x,*wind,*external,*u,float(np.linalg.norm(x[:2]-p)),float(clearance(x[:2])[0]),corr,int(feasible),slack,residual])
        if k==count:break
        for sub in range(substeps):
            h=DT/substeps;xn=np.array(microstep(jnp.array(x),jnp.array(u),field,h))
            hit=first_contact(x[:2],xn[:2])
            if hit is not None:
                x=x+hit*(xn-x);ct=t+sub*h+hit*h
                contact={'time_s':float(ct),'position_m':x[:2].tolist()}
                samples.append([ct,*x,*np.array(field.at(x[:2])[0]),*acceleration(x[2:],np.array(field.at(x[:2])[0])),*u,float(np.linalg.norm(x[:2]-ref.at(ct)[0])),0.,corr,int(feasible),slack,residual])
                minimum=0.;break
            x=xn;minimum=min(minimum,float(clearance(x[:2])[0]))
        if contact:break
    endpoint=ref.at(horizon)[0];error=float(np.linalg.norm(x[:2]-endpoint))
    in_target=(4.98<=x[0]<=8 and 2.295<=x[1]<=4.625) if ref.reverse else (0<=x[0]<=2.9 and 0<=x[1]<=1.755)
    status='collision' if contact else ('target region reached' if in_target else 'outside target region')
    return dict(samples=np.round(samples,7).tolist(),summary=dict(status=status, contact=contact,
        target_region_reached=bool(in_target and not contact),endpoint_error_m=error,
        endpoint_speed_m_s=float(np.linalg.norm(x[2:])),min_clearance_m=minimum,
        max_tracking_error_m=float(np.max(np.array(samples)[:,11])),
        max_control_component_m_s2=max_u,infeasible_steps=n_infeasible,max_slack=max_slack,
        max_constraint_residual=max_residual,initial_barrier_values=initial_value,
        max_air_acceleration_m_s2=float(np.max(np.linalg.norm(np.array(samples)[:,7:9],axis=1))),
        final_time_s=float(samples[-1][0]),passage_crossings=passage_crossings(samples),
        central_transit=central_transit(samples,'B-A' if ref.reverse else 'A-B')))

def checksum(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def directional_authority(ref,field):
    """Wind work and exact tracking demand along the unchanged reference.

    This is a local tracking-feasibility diagnostic, not a reachability proof.
    A safety filter can leave the reference and must be assessed separately.
    """
    t=np.linspace(0,ref.duration,2001);p,v,a=ref.at(t)
    wind=np.array(field.at(p));air=acceleration(v,wind)
    speed=np.linalg.norm(v,axis=1)
    tangent=np.divide(v,speed[:,None],out=np.zeros_like(v),where=speed[:,None]>1e-9)
    along=np.sum(air*tangent,axis=1)
    required=a-air
    excess=np.max(np.abs(required),axis=1)-CFG.umax
    return dict(sampling_points=len(t),air_work_per_mass_m2_s2=float(np.trapezoid(np.sum(air*v,axis=1),t)),
        max_opposing_air_acceleration_m_s2=float(max(0.,-along.min())),
        max_assisting_air_acceleration_m_s2=float(max(0.,along.max())),
        max_required_control_component_m_s2=float(np.max(np.abs(required))),
        reference_time_over_actuator_limit_s=float(np.trapezoid((excess>0).astype(float),t)),
        qualification='Computed on the prescribed reference. Exceeding the actuator bound rules out exact tracking there; it does not prove that every alternative trajectory is impossible.')

def build(site=SITE):
    out=site/'data/robot';out.mkdir(exist_ok=True)
    fields={s:load_wind(s,site) for s in (1,2)}
    still=FixedWind(jnp.zeros_like(fields[1].grid),fields[1].dx)
    cases=[];refs={};baselines={}
    for route in ('central','lower','upper'):
        for direction in ('B-A','A-B'):
            key=f'{route}-{direction}';ref=Reference(route,direction,site);refs[key]=ref
            tt=np.arange(0,ref.duration+10+DT/2,DT);p,v,a=ref.at(tt)
            baselines[key]=simulate(ref,still)
            for scenario in (1,2):
                ident=f's{scenario}-{key}'
                print(f'Running {ident}: reference {ref.duration:.2f}s',flush=True)
                plain=simulate(ref,fields[scenario]);safe=simulate(ref,fields[scenario],True)
                record=dict(id=ident,scenario=scenario,route=route,direction=direction,
                    reference=dict(checks=ref.checks,samples=np.round(np.c_[tt,p,v,a],7).tolist(),passage_crossings=passage_crossings(np.c_[tt,p,v])),
                    directional_authority=directional_authority(ref,fields[scenario]),
                    still_air=baselines[key],unfiltered=plain,rpcbf=safe)
                (out/(ident+'.json')).write_text(json.dumps(record,separators=(',',':'),allow_nan=False))
                cases.append({k:record[k] for k in ('id','scenario','route','direction','directional_authority')} | dict(duration_s=float(tt[-1]), reference=ref.checks,
                    unfiltered=plain['summary'],rpcbf=safe['summary'],still_air=baselines[key]['summary']))
                print('  ',plain['summary']['status'],'/',safe['summary']['status'], 'QP infeasible',safe['summary']['infeasible_steps'],flush=True)
    checks=[]
    # Same controller interval; halve only RK4/contact substeps. Do not silently
    # retune the control law or reject unfavorable outcomes.
    for scenario,route,direction in [(s,r,d) for s in (1,2) for r in ('central','lower','upper') for d in ('B-A','A-B')]:
        key=f'{route}-{direction}';ref=refs[key]
        base=json.loads((out/f's{scenario}-{key}.json').read_text())
        for method in ('unfiltered','rpcbf'):
            fine=simulate(ref,fields[scenario],method=='rpcbf',10)
            a=np.array(base[method]['samples']);b=np.array(fine['samples'])
            end=min(a[-1,0],b[-1,0]);t=a[a[:,0]<=end,0]
            err=np.linalg.norm(np.column_stack([np.interp(t,a[:,0],a[:,i])-np.interp(t,b[:,0],b[:,i]) for i in (1,2)]),axis=1)
            checks.append(dict(scenario=scenario,route=route,direction=direction,method=method,
                coarse_substep_s=.01,fine_substep_s=.005,max_common_time_position_difference_m=float(err.max()),
                same_outcome=base[method]['summary']['status']==fine['summary']['status'],
                coarse_contact=base[method]['summary']['contact'],fine_contact=fine['summary']['contact']))
    sources=['scripts/airflow/robot_review.py','src/voi_diffusion_ipp/safety/flow_policy_value.py','src/voi_diffusion_ipp/safety/qp.py','src/voi_diffusion_ipp/safety/spline.py','src/voi_diffusion_ipp/quadrotor/aerodynamics.py']
    manifest=dict(schema=1,model=dict(equations='p_dot=v; v_dot=u+(c/m)*|w_xy(p)-v|*(w_xy(p)-v)',mass_kg=MASS,
        quadratic_drag_kg_per_m=DRAG,control_component_bound_m_s2=CFG.umax,kp=KP,kv=KV,
        controller_period_s=DT,integration_substep_s=.01,robot_radius_m=0.,safety_margin_m=CFG.safety_margin,
        backup_steps=CFG.backup_steps,backup_dt_s=CFG.backup_dt,backup_gain=CFG.backup_gain,cbf_alpha=CFG.cbf_alpha,spline='full',
        wind_knowledge='Known fixed wind, zero uncertainty; three identical inner fields. Oracle diagnostic, no learned belief.',
        dimensions='Planar x,y translation at the displayed 0.75m slice; horizontal relative air velocity, no vertical/attitude dynamics.',
        reference='C2 cubic interpolant of the existing route guides, 41 arc-length knots; rest-to-rest quintic timing; speed <=1m/s and each acceleration <=0.8m/s^2.',
        tracker='clip(a_ref+1.2*(p_ref-p)+1.8*(v_ref-v),-1.6,1.6); identical in both conditions; no nominal wind compensation.',
        termination='Stop at first swept point contact; otherwise duration+10s. Target-region membership is separate from endpoint error.',
        still_air='Same air-force law with w=0, same nominal tracker, no safety filter.',
        qualification='Conditional diagnostic in the existing provisional CFD fields. Not a physical-wind validation, learned-wind test, or global safety/reachability proof.'),
        sample_columns=['time_s','x_m','y_m','vx_m_s','vy_m_s','wx_m_s','wy_m_s','air_ax_m_s2','air_ay_m_s2','ux_m_s2','uy_m_s2','tracking_error_m','clearance_m','filter_correction_m_s2','qp_feasible','qp_slack','qp_residual'],
        cases=cases,timestep_refinement=checks,
        source_sha256={s:checksum(ROOT/s) for s in sources},route_guide_sha256=checksum(site/'app.js'),
        wind_sha256={str(s):checksum(site/f'data/scenario{s}/stationary-velocity.f32') for s in (1,2)})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    print(json.dumps({'refinement':checks,'cases':len(cases)},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,default=SITE)
    args=parser.parse_args();build(args.site)
