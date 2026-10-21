"""Recorded nonlinear missions for the revised optimistic method.

All execution includes rigid-body attitude, motor dynamics, differential-flatness
feedforward and geometric feedback. The clock is simulation time: planning wall
latency is measured separately and no real-time execution claim is made.
"""
import argparse,json,time,hashlib
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import jax,jax.numpy as jnp
from scipy.spatial.transform import Rotation
from voi_diffusion_ipp.quarter.environment import Settings,environment,initial_belief,wind,metadata,table_sampler,TABLE_BASIS
from voi_diffusion_ipp.quarter.planner import optimizer,reference_predictor,library,seeds,sensing_points,decode
from voi_diffusion_ipp.quarter import sensing
from voi_diffusion_ipp.quarter.diffusion import cosine_schedule
from voi_diffusion_ipp.quadrotor.config import VehicleConfig
from voi_diffusion_ipp.quadrotor.types import State
from voi_diffusion_ipp.quadrotor.dynamics import Quadrotor
from voi_diffusion_ipp.quadrotor.reference import QuinticReference
from voi_diffusion_ipp.quadrotor.execution import executor
from voi_diffusion_ipp.quadrotor.aerodynamics import infer_wind
from voi_diffusion_ipp.wind.cache import TableMean

REVISION='quarter-optimistic-cosine-soft-v5'
METHODS=['proposed','blind','blind_filter','aware','optimistic','fading','no_learning','proposed_crn']

def write(path,data):
    # Null is reserved for missing data, never silently substituted for a NaN.
    text=json.dumps(data,separators=(',',':'),allow_nan=False)
    temp=path.with_suffix('.tmp');temp.write_text(text);temp.replace(path)


def run(layout,case,method,seconds,seed,cfg,out):
    started=time.perf_counter();env=environment(layout);vehicle=VehicleConfig();plan=optimizer(env,cfg,'proposed' if method in ('no_learning','proposed_crn') else method);predict=reference_predictor(env,cfg,method)
    b=initial_belief(cfg);sample=table_sampler(cfg.construction_worlds+cfg.worlds+cfg.validation_worlds)
    condition=jax.jit(lambda bb,p,y:bb.condition(p,y));prediction=jax.jit(lambda bb,p:bb.predict(p))
    sensor_step=jax.jit(lambda q,p0,p1,dt,h:sensing.step(q,p0,p1,env,cfg,dt,h))
    actual_wind=jax.jit(lambda p:wind(p,case,cfg.wind_strength))
    execute=executor(lambda p:wind(p,case,cfg.wind_strength),cfg.dt,cfg.commit_steps,dt=.01,max_horizontal_accel=cfg.umax,wind_velocity=True)
    key=jax.random.PRNGKey(seed);rng=np.random.default_rng(seed+9001)
    state=State(np.array([-28.,8.,3.2]),np.zeros(3),np.eye(3),np.zeros(3),np.full(4,np.sqrt(vehicle.mass*vehicle.gravity/(4*vehicle.thrust_coefficient))))
    plant=Quadrotor(vehicle,np.zeros(3),np.zeros(3));q=jnp.full(len(env.cells),.35,dtype=jnp.float32);q=q.at[:42].set(.65)
    visits=np.zeros((len(env.regions),4));sites=sensing_points(env);previous=seeds(len(env.regions))[0]
    inspection_phase=np.argmin(np.linalg.norm(sites-state.position[:2],axis=2),axis=1);continuation=None
    xx,zz=np.meshgrid(np.linspace(-37,37,49),np.linspace(-25.5,34.5,41));grid=np.c_[xx.ravel(),zz.ravel()].astype(np.float32)
    grid_wind=np.asarray(actual_wind(jnp.array(grid)))
    frames=[];beliefs=[];plans=[];observations=[];cost=0.;clock=0.;minimum=100.;failure=None;errors=[];last_a=np.zeros(3)
    trace_position=state.position.copy();target=out/f'{layout}_{case}_{method}_{seed}.json';initial_mean_std=None;common_sample_key=None
    def frame(reference):
        frames.append(dict(t=round(clock,4),p=np.round(state.position,5).tolist(),quat=np.round(Rotation.from_matrix(state.rotation).as_quat(),6).tolist(),q=np.asarray(q).round(5).tolist(),cost=cost,ref=np.asarray(reference).round(5).tolist(),velocity=state.velocity.round(5).tolist(),inspection_phase=inspection_phase.tolist(),wind=np.asarray(actual_wind(jnp.array(state.position[None,:2])))[0].round(4).tolist()))
    frame(state.position)
    def result(completed):
        return dict(revision=REVISION,completed=completed,layout=layout,case=case,method=method,seed=seed,mission_seconds=seconds,completed_seconds=clock,cost=cost,success=failure is None,failure=failure,min_clearance=minimum,config=asdict(cfg),vehicle=asdict(vehicle),frames=frames,beliefs=beliefs,plans=plans,observations=observations,wall_seconds=time.perf_counter()-started,environment={**metadata(layout),'grid':grid.tolist(),'grid_shape':[41,49],'wind':grid_wind.tolist(),'wind_units':'m/s'},model=dict(scenario_draws='shared base random draws across replans, transformed by the current posterior' if method=='proposed_crn' else 'fresh posterior draws each replan',planner='planar double integrator with quadratic relative-air-velocity drag',execution='nonlinear rigid-body quadrotor, motors, flatness and geometric feedback',planning_step=cfg.dt,execution_step=.01,schedule='cosine, warm-start interval [0, cosine_end]',cosine_cumulative=np.asarray(cosine_schedule(cfg.iterations,cfg.cosine_end)[1]).tolist(),belief='POAM posterior, fixed kernel/inducing locations; real-observation sufficient-statistic updates only',filter='spline design-policy value and penalized squared slack; no safety guarantee',policy='geometric route classes plus MBD target/passage logits, speed, dwell and crossing offsets',collision_penalty='full remaining mission with no additional sensing',sensing='Gaussian quality inside 45-degree ground footprint, building visibility masks'),rms_tracking_error=float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.,planning_seconds=[p['seconds'] for p in plans])
    for iteration in range(round(seconds/cfg.commit)):
        key,ks,kp=jax.random.split(key,3)
        if common_sample_key is None:common_sample_key=ks
        mean,fields=sample(b,common_sample_key if method=='proposed_crn' else ks);fields.block_until_ready()
        construction=fields[:cfg.construction_worlds];worlds=fields[cfg.construction_worlds:cfg.construction_worlds+cfg.worlds];validation=fields[-cfg.validation_worlds:]
        mu,cov=prediction(b,jnp.array(grid));std=np.sqrt(np.maximum(np.asarray(cov)[:,0,0],0))
        beliefs.append(dict(t=round(clock,4),mean=np.asarray(mu).round(4).tolist(),std=std.round(4).tolist(),rmse=float(np.sqrt(np.mean((np.asarray(mu)-grid_wind)**2)))))
        paths=library(state.position[:2],env,visits,inspection_phase,continuation);x=jnp.array(np.r_[state.position[:2],state.velocity[:2]],jnp.float32)
        eta=cfg.optimism if method!='fading' else cfg.optimism+(1-cfg.optimism)*min(clock/(seconds*.5),1.)
        start=time.perf_counter();theta,score,trace,expected,evaluations=plan(x,q,mean,construction,worlds,validation,paths,previous,kp,jnp.array(eta));theta.block_until_ready();elapsed=time.perf_counter()-start
        if not np.isfinite(float(score)):failure='no finite planning policy';break
        pred=predict(theta,x,q,mean,construction,paths);reference=np.asarray(pred.reference);accelerations=np.asarray(pred.acceleration)
        which,speed,dwell,offsets=decode(theta,len(env.regions),cfg)
        continuation=(int(which),paths,int(pred.cursor),float(pred.dwell_elapsed))
        plans.append(dict(t=clock,seconds=elapsed,compilation_included=iteration==0,predicted_optimistic=float(score),predicted_mean=float(expected),trace=np.asarray(trace).tolist(),evaluations=int(evaluations),policy=theta.tolist(),route_index=int(which),speed=float(speed),dwell=float(dwell),offsets=offsets.tolist(),max_slack=float(pred.max_slack),points=reference[:,:2].round(4).tolist()))
        polynomials=[]
        for i in range(cfg.commit_steps):
            p0=np.r_[reference[i,:2],3.2];p1=np.r_[reference[i+1,:2],3.2];v0=np.r_[reference[i,2:],0.];v1=np.r_[reference[i+1,2:],0.];a1=np.r_[accelerations[i],0.]
            polynomial=QuinticReference(p0,v0,last_a,p1,v1,a1,cfg.dt);polynomials.append(polynomial.coefficients);last_a=a1
        _,logs=execute(jnp.array(state.pack(),jnp.float32),jnp.array(np.array(polynomials),jnp.float32),TableMean(TABLE_BASIS,mean))
        states,refs,commands,_=map(np.asarray,logs)
        for step,(packed,ref,command) in enumerate(zip(states,refs,commands)):
            old_position=state.position.copy();state=State.unpack(packed.astype(float));clock+=.01
            close=np.linalg.norm(sites-state.position[:2],axis=2)<.40
            visits=np.where(close,visits+.01,0.)
            for site in range(len(inspection_phase)):
                stop=inspection_phase[site]
                if visits[site,stop]>=float(dwell):
                    inspection_phase[site]=(stop+1)%4
                    visits[site,stop]=0.
            old_def=float(jnp.maximum(env.targets-q,0).mean());q=sensor_step(q,jnp.array(old_position[:2]),jnp.array(state.position[:2]),jnp.array(.01),jnp.array(state.position[2]));new_def=float(jnp.maximum(env.targets-q,0).mean());cost+=.005*(old_def+new_def)
            gap=float(env.clearance(jnp.array(np.r_[state.position[:2],state.velocity[:2]])))-.275;minimum=min(minimum,gap)
            errors.append(float(np.linalg.norm(state.position-ref)))
            if step%10==9:frame(ref)
            if not np.isfinite(state.pack()).all() or not np.isfinite(float(q.sum())):failure='numerical instability';break
            if gap<0:failure='collision or environment boundary contact';break
            if state.position[2]<.5 or state.position[2]>7:failure='altitude tracking failure';break
        if failure:break
        physical=np.r_[np.asarray(actual_wind(jnp.array(state.position[None,:2])))[0],0.];plant.wind=physical
        residual=plant.derivative(state.pack(),command)[3:6]-(state.rotation[:,2]*(plant.allocation@(state.motor_speed**2))[0]/vehicle.mass-np.array([0,0,vehicle.gravity]))
        measured=infer_wind(state.velocity,residual)[:2]+cfg.observation_std*rng.normal(size=2)
        observations.append(dict(t=clock,position=state.position[:2].tolist(),measurement=measured.tolist()))
        if method!='no_learning':b=condition(b,jnp.array(state.position[None,:2],jnp.float32),jnp.array(measured[None],jnp.float32))
        previous=theta
        if iteration%5==0:
            print(f'{layout}/{case}/{method} {clock:.0f}/{seconds:g}s cost={cost:.3f} p={state.position[:2].round(2)} plan={elapsed:.2f}s slack={float(pred.max_slack):.3f}',flush=True)
            write(target,result(False))
    if failure:
        remaining=max(0,seconds-clock)
        for _ in range(round(remaining/.1)):
            q=q/(1+env.process_noise*q*.1);cost+=.1*float(jnp.maximum(env.targets-q,0).mean())
        print('FAILED:',failure,'at',clock,flush=True)
    write(target,result(True));print('SAVED',target,'cost',cost,'success',failure is None,flush=True)
    return result(True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--layout',choices=['two','multi'],default='two');p.add_argument('--case',choices=['forward','reverse','crosswind'],default='forward');p.add_argument('--methods',nargs='+',default=['proposed','blind','blind_filter']);p.add_argument('--seconds',type=float,default=240);p.add_argument('--seed',type=int,default=8101);p.add_argument('--horizon',type=float,default=60);p.add_argument('--iterations',type=int,default=6);p.add_argument('--proposals',type=int,default=8);p.add_argument('--worlds',type=int,default=6);p.add_argument('--validation-worlds',type=int,default=8);p.add_argument('--temperature',type=float,default=.15);p.add_argument('--wind',type=float,default=3.1);p.add_argument('--optimism',type=float,default=.5);p.add_argument('--commit',type=float,default=2);p.add_argument('--output',default='output/european_quarter');p.add_argument('--force',action='store_true');a=p.parse_args()
    cfg=Settings(horizon=a.horizon,iterations=a.iterations,proposals=a.proposals,worlds=a.worlds,validation_worlds=a.validation_worlds,temperature=a.temperature,wind_strength=a.wind,optimism=a.optimism,commit=a.commit,seed=a.seed)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for method in a.methods:
        if method not in METHODS:raise ValueError(method)
        path=out/f'{a.layout}_{a.case}_{method}_{a.seed}.json'
        if path.exists() and not a.force:
            prior=json.loads(path.read_text())
            if prior.get('completed') and prior.get('revision')==REVISION and prior.get('config')==asdict(cfg) and prior.get('mission_seconds')==a.seconds:print('Already completed',path,flush=True);continue
        run(a.layout,a.case,method,a.seconds,a.seed,cfg,out)
        # Release compiled executables before a different planner is built.
        jax.clear_caches()
if __name__=='__main__':main()
