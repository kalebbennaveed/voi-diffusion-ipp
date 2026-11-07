"""Geometry and predeclared initial conditions for the user's five-column sketch."""
import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from voi_diffusion_ipp.environment import Environment
from monitoring_routes import clearance

DEFAULT_SPEC=Path(__file__).resolve().parents[2]/'results/airflow/cluster_revision_20261103/environment-spec.json'
CASES=[
    dict(id='center',label='Center of B · at rest',state=[1.45,.875,0.,0.],seed=8101),
    dict(id='west',label='West of B · at rest',state=[.80,.65,0.,0.],seed=8101),
    dict(id='east',label='East of B · at rest',state=[2.35,1.20,0.,0.],seed=8101),
    dict(id='north-moving',label='North of B · moving north',state=[1.45,1.30,0.,.30],seed=8101),
    dict(id='east-moving',label='Center of B · moving east · seed 9101',state=[1.45,.875,.35,0.],seed=9101),
    dict(id='a-deficit',label='A initially needs more sensing · seed 10101',state=[1.45,.875,0.,0.],seed=10101,q_A=.35,q_B=.60),
]

def environment(cfg,spec_path=DEFAULT_SPEC):
    # Geometry is continuous. Integer JSON literals must not become integer JAX
    # constants captured across the planner's float32 compilation context.
    spec=json.loads(Path(spec_path).read_text());bounds=np.asarray(spec['bounds'],dtype=float);ob=np.asarray(spec['obstacles'],dtype=float)
    dx=spec['dx'];nx,ny=np.rint((bounds[1]-bounds[0])/dx).astype(int)
    xx,yy=np.meshgrid(bounds[0,0]+(np.arange(nx)+.5)*dx,bounds[0,1]+(np.arange(ny)+.5)*dx)
    all_cells=np.c_[xx.ravel(),yy.ravel()];free=clearance(all_cells,ob,bounds)>0;cells=all_cells[free]
    member=np.stack([np.all(cells>=r['bounds'][0],axis=1)&np.all(cells<=r['bounds'][1],axis=1) for r in spec['regions']],axis=1)
    important=member.any(axis=1)
    env=Environment(jnp.asarray(cells),jnp.asarray(np.where(important,spec.get('region_target',cfg.region_target),spec.get('background_target',cfg.background_target))),
        jnp.asarray(ob),jnp.asarray(bounds),jnp.asarray(np.mean(spec['regions'][0]['bounds'],axis=0)),
        jnp.asarray(np.mean(spec['regions'][1]['bounds'],axis=0)),jnp.zeros(2),
        jnp.asarray(np.where(important,cfg.clarity_decay,cfg.background_decay)))
    grid=dict(grid=[int(nx),int(ny)],dx_m=dx,cells=cells,all_cells=all_cells,free_indices=np.flatnonzero(free),membership=member)
    return env,grid,spec

def initial(case,env,grid,cfg):
    x=jnp.asarray(case['state']);q=np.full(len(env.cells),cfg.initial_q)
    for i,label in enumerate(['A','B']):
        if 'q_'+label in case:q[grid['membership'][:,i]]=case['q_'+label]
    assert float(env.clearance(x))>.12
    return x,jnp.asarray(q)

def crossings(samples,spec):
    """Same geometric passage-gate definition, using the actual bordering pair."""
    rows=np.asarray(samples);a,b=np.asarray(spec['obstacles'])[spec['passage_obstacles']]
    d=b[:2]-a[:2];length=np.linalg.norm(d);axis=d/length;normal=np.array([-axis[1],axis[0]])
    signed=(rows[:,1:3]-a[:2])@normal;events=[]
    for i in np.flatnonzero(signed[:-1]*signed[1:]<0):
        f=-signed[i]/(signed[i+1]-signed[i]);p=rows[i,1:3]*(1-f)+rows[i+1,1:3]*f;c=float((p-a[:2])@axis)
        route='upper' if c < -a[2] else 'lower' if c > length+b[2] else 'central' if a[2]<c<length-b[2] else 'solid interval'
        events.append(dict(time_s=float(rows[i,0]*(1-f)+rows[i+1,0]*f),position_m=p.tolist(),route=route,
            direction='B-A' if signed[i+1]>signed[i] else 'A-B',speed_m_s=float(np.linalg.norm(rows[i,3:5]*(1-f)+rows[i+1,3:5]*f)) if rows.shape[1]>=5 else None))
    return events

def first_transfer(samples,spec):
    rows=np.asarray(samples);lo,hi=np.asarray(spec['regions'][0]['bounds']);in_a=np.all(rows[:,1:3]>=lo,axis=1)&np.all(rows[:,1:3]<=hi,axis=1)
    first=float(rows[np.flatnonzero(in_a)[0],0]) if in_a.any() else None
    forward=[c for c in crossings(rows,spec) if c['direction']=='B-A' and (first is None or c['time_s']<=first)]
    return dict(first_A_s=first,route=forward[-1]['route'] if first is not None and forward else None,attempted_crossings=forward)
