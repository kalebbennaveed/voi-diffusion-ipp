"""Old batched MBD pipeline applied to the approved circular environments.

Uses quarter.continuous.optimizer and quarter.multimode.search unchanged,
apart from the optimizer's optional environment-specific rollout argument.
Old .25 s forecast RK4 and C1 wind-table queries are explicit numerical choices.
The executed plant retains .01 s RK4 and the original .05 s hard RPCBF.
"""
from dataclasses import dataclass,replace
import numpy as np
import jax
import jax.numpy as jnp
import monitoring_mbd as m
import monitoring_routes
from voi_diffusion_ipp.quarter import continuous as old
from voi_diffusion_ipp.quarter.environment import ExtendedTableBasis
from voi_diffusion_ipp.wind.cache import TableMean
from voi_diffusion_ipp.wind.gp import InnerFields,WorldField
from voi_diffusion_ipp.wind.poam import POAM,select_inducing

@dataclass(frozen=True)
class Settings(old.ControlSettings):
    dt:float=.25
    horizon:float=15.
    commit:float=5.
    iterations:int=4
    proposals:int=28
    batch_size:int=28
    validation_worlds:int=8
    worlds:int=4
    construction_worlds:int=2
    mode_count:int=7
    multimode:bool=True
    initialization:str='modes'
    guidance_mode:str='off'
    temperature:float=.06
    cosine_end:float=.15
    latent_bound:float=12.
    optimism:float=1.
    mixture:float=0.
    sensor_sigma:float=.5
    sensor_R:float=.1
    umax:float=1.6
    safety_margin:float=.06
    safety_confidence:float=3.
    backup_dt:float=.15
    cbf_alpha:float=1.
    planning_clearance:float=.06
    violation_weight:float=1.
    slack_weight:float=15.
    prediction_filter:str='soft'
    terminate_predictions:bool=True
    @property
    def horizon_seconds(self):return self.horizon

def physical_settings(cfg):
    return replace(m.Settings(),dt=cfg.dt,horizon_seconds=cfg.horizon,commit_seconds=cfg.commit,
                   inner_samples=cfg.construction_worlds,worlds=cfg.worlds,reference_tracking=True,
                   tracking_kp=cfg.tracking_kp,tracking_kv=cfg.tracking_kv)

# Old table resolution, padded so all physical-workspace queries are interior.
SPACING=.2
LOWER=jnp.array([-1.,-1.],jnp.float32)
SHAPE=(35,51,2)
BASIS=ExtendedTableBasis(LOWER,jnp.array([SPACING,SPACING],jnp.float32))
yy,xx=np.meshgrid(np.arange(SHAPE[0]),np.arange(SHAPE[1]),indexing='ij')
POINTS=LOWER+jnp.asarray(np.c_[xx.ravel(),yy.ravel()],jnp.float32)*SPACING

def float32(tree):
    return jax.tree.map(lambda a:a.astype(jnp.float32) if hasattr(a,'dtype') and jnp.issubdtype(a.dtype,jnp.floating) else a,tree)

def matched_prior(env):
    """Survey-matched GP capacity and geometry, with zero wind observations.

    The survey used an untrained kernel initialized with key 731 and selected
    128 inducing locations from free-space geometry. Repeat only those steps;
    no survey file, wind value or measurement enters this prior.
    """
    with jax.experimental.disable_x64():
        locations=jnp.asarray(env.cells,jnp.float32)
        options=dict(noise_std=.1,prior_std=1.8,spectral_count=32,prior_support=np.asarray(POINTS)[::3])
        model=POAM(locations[:128],jax.random.PRNGKey(731),**options)
        centers=select_inducing(model.parameters,locations,model.lengths,128)
        return POAM(centers,jax.random.PRNGKey(731),**options).planning_belief()

def table_sampler(cfg):
    @jax.jit
    def sample(b,performance_key,construction_key):
        construction=b.sample_field(construction_key,cfg.construction_worlds)
        performance=b.sample_field(performance_key,cfg.worlds+cfg.validation_worlds)
        coefficients=jnp.concatenate((construction,performance))
        fields=jax.vmap(lambda c:b.basis.evaluate(POINTS,c))(coefficients).reshape((-1,*SHAPE))
        return b.mean_at(POINTS).reshape(SHAPE),fields
    return sample

def rollout(theta,x0,q0,mean_array,construction,physical,env,cfg,filter_on=True):
    mean=TableMean(BASIS,mean_array);inner=InnerFields(BASIS,construction);world=WorldField(BASIS,physical)
    reference,aa=old.nominal_reference(theta,x0,cfg)
    def step(carry,inputs):
        x,q,monitor,penalty,max_slack,gap,old_u,alive=carry
        ref,a=inputs
        u,slack=old.applied_control(x,ref,a,mean,inner,env,cfg,filter_on)
        # Same RK4 forecast interval as the original pipeline; same air-force law.
        xn=m.physical.rk4(x,u,world,cfg.dt)
        hit=m.contact_fraction(x[:2],xn[:2],env)<=1
        observing=alive&~hit
        xn=jnp.where(observing,xn,jnp.r_[x[:2],jnp.zeros(2)])
        qn=m.clarity.step(q,x[:2],xn[:2],env.cells,cfg.sensor_sigma,cfg.sensor_R,env.process_noise,cfg.dt)
        qn=jnp.where(observing,qn,q/(1+cfg.dt*env.process_noise*q))
        cost=cfg.dt*.5*(m.clarity.deficit(q,env.targets)+m.clarity.deficit(qn,env.targets))
        clearance=env.clearance(xn)
        soft=cfg.violation_weight*jax.nn.relu(cfg.planning_clearance-clearance)**2
        soft+=cfg.speed_penalty*jax.nn.relu(jnp.linalg.norm(xn[2:])-cfg.speed_max)**2
        new_gap=jnp.where(alive,jnp.minimum(gap,jnp.where(hit,0.,clearance)),gap)
        u=jnp.where(alive,u,jnp.zeros_like(u))
        return (xn,qn,monitor+cost,penalty+cfg.dt*soft*alive,jnp.maximum(max_slack,slack*alive),new_gap,u,observing),(xn,qn,u)
    initial=(x0,q0,jnp.array(0.),jnp.array(0.),jnp.array(0.),env.clearance(x0),jnp.zeros(2),jnp.array(True))
    end,(xs,qs,us)=jax.lax.scan(step,initial,(reference[:-1],aa))
    return old.Rollout(jnp.r_[x0[None],xs],jnp.r_[q0[None],qs],us,reference,end[2],end[3],end[4],end[5],~end[7])

def optimizer(env,cfg,aware=True):
    return old.optimizer(env,cfg,'aware' if aware else 'blind_filter',rollout_fn=rollout)

def seed_bank(x,q,env,cfg,previous=None):
    seeds,metadata=monitoring_routes.seed_bank(x,q,env,cfg,previous)
    # Original seven modes: hold plus six geometric starts. Previous is a
    # separate evaluated incumbent supplied to the original optimizer.
    return jnp.asarray(seeds[1:]),jnp.asarray(seeds[0]),metadata
