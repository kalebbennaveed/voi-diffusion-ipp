"""Bounded saved-state component timings; no optimizer or mission runs."""
import copy,inspect,json,time,statistics,hashlib,platform
from pathlib import Path
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import monitoring_mbd as m
import monitoring_routes
from voi_diffusion_ipp.wind.gp import InnerFields,WorldField
from voi_diffusion_ipp.safety.flow_policy_value import values
from voi_diffusion_ipp.quarter.filter import soft_projection

root=Path('results/airflow/monitoring_coarse_review')
dest=Path('results/airflow/monitoring_runtime_component_profile.json')
protocol=json.loads((root/'s2-aware-protocol.json').read_text())
cfg=m.Settings(**protocol['settings']);env,grid=m.environment(cfg)
data=np.load(root/'s2-aware-plan-005.npz')
template=m.estimator(cfg,jax.random.PRNGKey(0)).planning_belief()
b=jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template),[jnp.asarray(data[k]) for k in sorted(data.files) if k.startswith('belief_leaf_')])
inner=InnerFields(b.basis,jnp.asarray(data['inner_coefficients']));outer=InnerFields(b.basis,jnp.asarray(data['outer_coefficients']))
x=jnp.asarray(data['initial_state']);q=jnp.asarray(data['initial_clarity']);theta=jnp.asarray(data['theta'])
report=json.loads(dest.read_text()) if dest.exists() else {'source':'s2-aware-plan-005.npz','mission_time_s':25,'jax':jax.__version__,'devices':list(map(str,jax.devices())),'platform':platform.platform(),'x64':jax.config.x64_enabled,'convention':'Synchronized isolated component timings, first call separate; no optimizer rerun. Separately compiled times are NOT an additive partition of recorded fused optimizer. No physical or numerical settings changed.','timings':{},'horizons':{}}
def save():dest.write_text(json.dumps(report,indent=2))
def measure(name,fn,repeats=3):
    ts=[]
    for i in range(repeats+1):
        start=time.perf_counter();out=fn();jax.block_until_ready(out);ts.append(time.perf_counter()-start)
    row={'first_s':ts[0],'warm_s':ts[1:],'median_s':statistics.median(ts[1:]),'max_s':max(ts[1:])}
    report['timings'][name]=row;save();print(name,json.dumps(row),flush=True)
    return out

# Exact source extraction separates the independent motion and clarity stages.
# Original body is preserved, and each split result is checked against the full objective.
source=inspect.getsource(m.absorbing_forecast_cost)
head,tail=source.split('    def one_world(item):',1)
head=head.replace('def absorbing_forecast_cost(', 'def profile_motion(')+'    return end\n'
clarity='def profile_clarity(end,q,env,cfg):\n    h=cfg.physics_dt\n    def one_world(item):'+tail
ns=dict(m.__dict__);exec(head,ns);exec(clarity,ns)
report['source_sha256']=hashlib.sha256(source.encode()).hexdigest()
for horizon in (() if report['horizons'] else (10.,20.,30.)):
    c=replace(cfg,horizon_seconds=horizon);th=theta[:c.steps*2]
    full=jax.jit(lambda th,x,q,b,inner,outer:m.absorbing_forecast_cost(th,x,q,b,inner,outer,env,c,True))
    motion=jax.jit(lambda th,x,q,b,inner,outer:ns['profile_motion'](th,x,q,b,inner,outer,env,c,True))
    clarity=jax.jit(lambda end,q:ns['profile_clarity'](end,q,env,c))
    cost=measure(f'{horizon:g}s_single_candidate_full',lambda:full(th,x,q,b,inner,outer),2)
    end=measure(f'{horizon:g}s_single_candidate_motion_rpcbf',lambda:motion(th,x,q,b,inner,outer),2)
    split=measure(f'{horizon:g}s_single_candidate_clarity',lambda:clarity(end,q),2)
    error=abs(float(cost)-float(split));assert error<1e-9,(horizon,error)
    report['horizons'][str(horizon)]={'split_cost_error':error,'cost':float(cost),'decisions':c.steps,'worlds':c.worlds,'physics_steps_per_world':round(horizon/c.physics_dt),'cells':len(q)};save()

refs,aa=m.nominal_reference(theta,x,cfg)
mean=jax.jit(lambda x,b:b.mean_at(x[:2]))
measure('poam_mean_one_position',lambda:mean(x,b),10)
field=jax.jit(lambda x,o:WorldField(o.basis,o.coefficients[0]).at(x[:2]))
measure('poam_sampled_field_one_position',lambda:field(x,outer),10)
track=jax.jit(lambda x,ref,a,b:m.tracking_control(x,ref,a,b,cfg,True))
nom=measure('pd_tracker_and_mean_compensation',lambda:track(x,refs[0],aa[0],b),10)
def constraints(x,b,inner):
    def f(z):
        value=values(z,b,inner,env,cfg)
        return value,value
    grad,B=jax.jacfwd(f,has_aux=True)(x)
    drift=m.physical.acceleration(x[2:],inner.at(x[:2])[:,0],xp=jnp)
    return grad[:,2:],-cfg.cbf_alpha*B-grad[:,:2]@x[2:]-jnp.max(grad[:,2:]@drift.T,axis=1)
con=jax.jit(constraints)
A,rhs=measure('rpcbf_backup_spline_derivatives_bounds',lambda:con(x,b,inner),10)
qp=jax.jit(lambda nom,A,rhs:soft_projection(nom,A,rhs,cfg.umax,cfg.slack_weight))
measure('relaxed_qp_only',lambda:qp(nom,A,rhs),10)
filter_fn=jax.jit(lambda x,nom,b,inner:m.relaxed_rpcbf(x,nom,b,inner,env,cfg))
u,_=measure('relaxed_rpcbf_complete_one_step',lambda:filter_fn(x,nom,b,inner),10)
hard=jax.jit(lambda x,nom,b,inner:m.filter_control(x,nom,b,inner,env,cfg))
measure('execution_hard_rpcbf_one_step',lambda:hard(x,nom,b,inner),10)
rk=jax.jit(lambda x,u,o:m.physical.rk4(x,u,WorldField(o.basis,o.coefficients[0]),cfg.physics_dt))
measure('windy_rk4_one_001s_step',lambda:rk(x,u,outer),10)
cs=jax.jit(lambda q,x:m.clarity.step(q,x[:2],x[:2],env.cells,cfg.sensor_sigma,cfg.sensor_R,env.process_noise,cfg.physics_dt))
measure('clarity_one_001s_step_all_cells',lambda:cs(q,x),10)
eps=jax.random.normal(jax.random.PRNGKey(99),(cfg.inner_samples+cfg.worlds,*b.sample_shape))
draw=jax.jit(lambda b,eps:b.inner_fields(eps))
measure('draw_five_coherent_fields',lambda:draw(b,eps),5)
measure('coarse_seed_bank',lambda:jnp.asarray(monitoring_routes.seed_bank(x,q,env,cfg,None)[0]),3)

# Replay actual observations only to restore a real estimator update, never a mission.
obs=np.load(root/'s2-aware-execution.npz')['observation_rows']
poam=m.estimator(cfg,jax.random.PRNGKey(protocol['seed']+101))
initial_x=jnp.array([[1.45,.875]])
truth=m.physical.load_wind(2,m.physical.SITE)
initial_y=truth.at(initial_x)+cfg.observation_std*np.random.default_rng(protocol['seed']).normal(size=(1,2))
poam.update(initial_x,initial_y,jax.random.PRNGKey(protocol['seed']+102),train_steps=cfg.poam_train_steps)
for cycle in range(4):
    batch=obs[(obs[:,0]>cycle*5)&(obs[:,0]<=(cycle+1)*5)]
    poam.update(jnp.asarray(batch[:,1:3]),jnp.asarray(batch[:,3:]),jax.random.PRNGKey(protocol['seed']+3000+cycle),train_steps=cfg.poam_train_steps,relocate=True)
batch=obs[(obs[:,0]>20)&(obs[:,0]<=25)]
def update():
    model=copy.copy(poam)
    model.update(jnp.asarray(batch[:,1:3]),jnp.asarray(batch[:,3:]),jax.random.PRNGKey(protocol['seed']+3004),train_steps=cfg.poam_train_steps,relocate=True)
    return model.mean,model.covariance,model.parameters
measure('poam_real_update_25_observations_10_train_relocate',update,2)
measure('poam_construct_planning_belief',lambda:poam.planning_belief(),3)
report['complete']=True;save();print('PROFILE_COMPLETE',flush=True)
