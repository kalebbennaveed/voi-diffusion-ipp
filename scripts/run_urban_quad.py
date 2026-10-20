"""Run actual planar MBD -> flatness -> nonlinear quad missions and export replay."""
import argparse
import json
import time
from dataclasses import asdict,replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from voi_diffusion_ipp.experiments.urban_quad import environment,config,initial_belief,build_cache,wind_load,CASE_NAMES,METHOD_NAMES
from voi_diffusion_ipp.planning.urban import optimizer,reference_predictor,route_library,decode,rollout
from voi_diffusion_ipp.quadrotor.config import VehicleConfig,ControllerConfig
from voi_diffusion_ipp.quadrotor.controller import GeometricController
from voi_diffusion_ipp.quadrotor.dynamics import Quadrotor
from voi_diffusion_ipp.quadrotor.types import State
from voi_diffusion_ipp.quadrotor.reference import QuinticReference,flat_reference


def clarity_step(q,p,cells,dt):
    gain=np.exp(-np.sum((cells-p)**2,axis=1)/4.)/.3
    f=lambda qq:gain*(1-qq)**2-.025*qq**2
    k1=f(q);k2=f(q+dt*k1/2);k3=f(q+dt*k2/2);k4=f(q+dt*k3)
    return np.clip(q+dt*(k1+2*k2+2*k3+k4)/6,0,1)


def clearance(p,rectangles):
    mid=(rectangles[:,:2]+rectangles[:,2:4])/2
    half=(rectangles[:,2:4]-rectangles[:,:2])/2
    d=np.abs(p[:2]-mid)-half
    sdf=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(d.max(axis=1),0)
    sdf=np.where(p[2]<rectangles[:,4]+.2,sdf,np.inf)
    return min(sdf.min(),p[0],p[1],40-p[0],28-p[1],p[2])


def write_json(path,data):
    def sanitize(v):
        if isinstance(v,dict):return {k:sanitize(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [sanitize(x) for x in v]
        if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
        return v
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(sanitize(data),separators=(',',':'),allow_nan=False));tmp.replace(path)


def run(case,method,seconds,seed,out,cfg,plan,predict):
    start=time.perf_counter();env=environment(case);vehicle=VehicleConfig();gains=ControllerConfig(max_horizontal_accel=cfg.umax)
    cells=np.asarray(env.cells);rectangles=np.asarray(env.rectangles)
    key=jax.random.PRNGKey(seed);b=initial_belief(jax.random.PRNGKey(8101))
    rng=np.random.default_rng(seed)
    # Identical prior, initial state and sensor-noise sequence across methods.
    state=State(np.array([6.,6.,3.]),np.zeros(3),np.eye(3),np.zeros(3),
        np.full(4,np.sqrt(vehicle.mass*vehicle.gravity/(4*vehicle.thrust_coefficient))))
    controller=GeometricController(vehicle,gains);plant=Quadrotor(vehicle,np.zeros(3),np.zeros(3))
    q=np.full(len(cells),.35);q[cells[:,0]<20]=.70
    eps=jax.random.normal(jax.random.PRNGKey(seed+100),(cfg.worlds+cfg.inner_samples,*b.sample_shape))
    cache=jax.jit(build_cache)(b,eps)
    condition=jax.jit(lambda bb,p,y:bb.condition(p,y))
    mean_fn=jax.jit(lambda bb,p:bb.mean_at(p))
    predict_belief=jax.jit(lambda bb,p:bb.predict(p))
    load_fn=jax.jit(lambda p:wind_load(p,case))
    from voi_diffusion_ipp.quadrotor.execution import executor
    execute=executor(lambda p:wind_load(p,case),cfg.dt,cfg.stage_steps,max_horizontal_accel=cfg.umax)
    xx,yy=np.meshgrid(np.linspace(1,39,20),np.linspace(1,27,14));grid=np.c_[xx.ravel(),yy.ravel()]
    wind=np.asarray(load_fn(jnp.asarray(grid)))
    total_cost=0.;minimum=100.;frames=[];beliefs=[];plans=[];errors=[];saturation=0;failure=None
    last_a=np.zeros(3);clock=0.;dt=.01
    def record(ref_position):
        frames.append(dict(t=round(clock,3),p=np.round(state.position,5).tolist(),
            quat=np.round(Rotation.from_matrix(state.rotation).as_quat(),6).tolist(),
            motors=np.round(state.motor_speed,2).tolist(),q=np.round(q,4).tolist(),
            ref=np.round(ref_position,4).tolist(),cost=round(total_cost,5)))
    record(state.position)
    periods=int(np.ceil(seconds/(cfg.dt*cfg.stage_steps)))
    for period in range(periods):
        x=jnp.asarray(np.r_[state.position[:2],state.velocity[:2]],jnp.float32)
        library=route_library(state.position[:2],env)
        key,k1,k2=jax.random.split(key,3)
        noise=jax.random.normal(k1,(cfg.worlds,cfg.stage_steps+cfg.future_steps,2))
        stamp=time.perf_counter()
        theta,cost,trace,initial_costs=plan(x,jnp.asarray(q,jnp.float32),b,cache,noise,library,k2)
        theta.block_until_ready();plan_time=time.perf_counter()-stamp
        if not np.isfinite(float(cost)):
            raise RuntimeError(f"No finite MBD policy prediction: {case}/{method} at {clock}s")
        pred=predict(theta,x,jnp.asarray(q,jnp.float32),b,cache,library)
        xs=np.asarray(pred.x);aa=np.asarray(pred.acceleration)
        index,speed,dwell=decode(theta)
        plans.append(dict(t=round(clock,3),index=int(index),speed=float(speed),dwell=float(dwell),
            predicted_cost=float(cost),seconds=plan_time,trace=np.asarray(trace).tolist(),
            points=xs[:,:2].round(4).tolist(),correction=float(jnp.mean(pred.correction)),
            qp_slack=float(jnp.max(pred.slack))))
        mu,cov=predict_belief(b,jnp.asarray(grid,jnp.float32))
        beliefs.append(dict(t=round(clock,3),mean=np.asarray(mu).round(4).tolist(),
                            std=np.sqrt(np.asarray(cov)[:,0,0]).round(4).tolist()))
        # Identical geometric controller + full rigid body, compiled as a scan.
        table_mean,_=cache.tabulate(b)
        polynomials=[]
        for segment in range(cfg.stage_steps):
            p0=np.r_[xs[segment,:2],3.];p1=np.r_[xs[segment+1,:2],3.]
            v0=np.r_[xs[segment,2:],0.];v1=np.r_[xs[segment+1,2:],0.]
            end_a=np.r_[aa[segment],0.]
            ref=QuinticReference(p0,v0,last_a,p1,v1,end_a,cfg.dt)
            polynomials.append(ref.coefficients);last_a=end_a
        _,logs=execute(jnp.asarray(state.pack(),jnp.float32),jnp.asarray(np.array(polynomials),jnp.float32),table_mean)
        states,reference_positions,commands,_=map(np.asarray,logs)
        for tick,(packed,ref_p,command) in enumerate(zip(states,reference_positions,commands)):
            state=State.unpack(packed.astype(float));clock+=dt
            saturation+=int(np.any(command>vehicle.max_motor_speed*.999))
            gap=clearance(state.position,rectangles)-.20;minimum=min(minimum,gap)
            errors.append(float(np.linalg.norm(state.position-ref_p)))
            old_def=np.maximum(.75-q,0).mean()
            q=clarity_step(q,state.position[:2],cells,dt)
            total_cost+=dt*.5*(old_def+np.maximum(.75-q,0).mean())
            if tick%10==9:record(ref_p)
            if gap<0 or not np.isfinite(state.pack()).all() or state.position[2]>8:
                failure="collision" if gap<0 else "tracking instability";break
        if failure:break
        # Acceleration residual: remove known thrust, gravity and drag. The
        # unknown external force is not used by the observer or controller.
        physical=np.r_[np.asarray(load_fn(jnp.asarray(state.position[None,:2])))[0],0.]
        plant.external_force=vehicle.mass*physical
        sensed=plant.derivative(state.pack(),command)[3:6]
        thrust=(plant.allocation@(state.motor_speed**2))[0]
        known=state.rotation[:,2]*thrust/vehicle.mass-np.array([0,0,vehicle.gravity])-vehicle.linear_drag*np.linalg.norm(state.velocity)*state.velocity/vehicle.mass
        obs=(sensed-known)[:2]+cfg.observation_std*rng.normal(size=2)
        b=condition(b,jnp.asarray(state.position[None,:2],jnp.float32),jnp.asarray(obs[None],jnp.float32))
        if period%5==0:
            print(f"{case} {method}: {clock:.0f}/{seconds:.0f}s, cost {total_cost:.2f}, clearance {minimum:.2f}m, plan {plan_time:.2f}s",flush=True)
    # Failed runs retain the full mission denominator with zero further sensing.
    if failure:
        remaining=max(0,seconds-clock)
        for _ in range(round(remaining/.1)):
            q=q/(1+.025*q*.1);total_cost+=.1*np.maximum(.75-q,0).mean()
    result=dict(model_revision="matched_limits_v3",case=case,method=method,seed=seed,mission_seconds=seconds,completed_seconds=clock,
        success=failure is None,failure=failure,cost=total_cost,min_clearance=minimum,
        rms_tracking_error=float(np.sqrt(np.mean(np.square(errors)))),max_tracking_error=max(errors),
        saturation_steps=saturation,wall_seconds=time.perf_counter()-start,frames=frames,beliefs=beliefs,plans=plans,
        final_mean_uncertainty=float(np.mean(beliefs[-1]['std'])),config=asdict(cfg),vehicle=asdict(vehicle),
        environment=dict(bounds=np.asarray(env.bounds).tolist(),buildings=rectangles.tolist(),regions=np.asarray(env.regions).tolist(),cells=cells.tolist(),grid=grid.tolist(),wind=wind.tolist()))
    assert np.isfinite(total_cost) and np.isfinite(result['rms_tracking_error'])
    target=out/f'{case}_{method}_{seed}.json';write_json(target,result)
    print(f"SAVED {target.name}: cost={total_cost:.4f}, success={failure is None}, tracking RMS={result['rms_tracking_error']:.3f}m",flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',default=list(CASE_NAMES));p.add_argument('--methods',nargs='+',default=list(METHOD_NAMES));p.add_argument('--seconds',type=float,default=120);p.add_argument('--seed',type=int,default=8101);p.add_argument('--quick',action='store_true');p.add_argument('--commit-seconds',type=float,default=2.0);p.add_argument('--output',default='output/urban_quad');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);cfg=config()
    commitment_steps=round(args.commit_seconds/cfg.dt)
    cfg=replace(cfg,stage_steps=commitment_steps,future_steps=100-commitment_steps)
    if args.quick:cfg=replace(cfg,future_steps=45,diffusion_steps=2,proposals=4)
    for case in args.cases:
        env=environment(case)
        for method in args.methods:
            target=out/f'{case}_{method}_{args.seed}.json'
            if target.exists():
                existing=json.loads(target.read_text())
                if existing.get('model_revision')=='matched_limits_v3' and existing['mission_seconds']==args.seconds and existing['config']==json.loads(json.dumps(asdict(cfg))) and all(p['predicted_cost'] is not None for p in existing['plans']):
                    print(f"Already complete: {target}",flush=True);continue
            run(case,method,args.seconds,args.seed,out,cfg,optimizer(env,cfg,method),reference_predictor(env,cfg,method))

if __name__=='__main__':main()
