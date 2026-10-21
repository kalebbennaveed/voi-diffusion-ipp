"""Metre geometry shared by the rendered quarter and its experiment.

The unknown quantity is a static, spatial air-velocity field. Vehicle parameters
are known. The field is synthetic, not CFD or a measured European city.
"""
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from ..experiments.urban_quad import UrbanEnvironment,smooth_window
from ..wind.poam import POAM
from ..wind.cache import TableBasis,TableMean

@dataclass(frozen=True)
class Settings:
    dt: float=.1
    horizon: float=60.
    commit: float=2.
    worlds: int=8
    validation_worlds: int=12
    construction_worlds: int=2
    iterations: int=8
    proposals: int=16
    temperature: float=.15
    optimism: float=.5
    mixture: float=.35
    archive_size: int=8
    latent_bound: float=7.
    cosine_end: float=.75
    umax: float=2.2
    speed_max: float=3.1
    backup_steps: int=8
    backup_dt: float=.25
    backup_gain: float=2.
    cbf_alpha: float=1.5
    safety_margin: float=.4
    slack_weight: float=15.
    violation_weight: float=25.
    spline: str='full'
    sensor_sigma: float=2.2
    sensor_R: float=.3
    observation_std: float=.25
    wind_strength: float=3.1
    prior_std: float=1.8
    seed: int=8101

    def __post_init__(self):
        if not 0<self.dt<=.5:raise ValueError('Planning dt must be positive and <= 0.5 s')
        if abs(self.dt/.01-round(self.dt/.01))>1e-6:raise ValueError('Planning dt must be a multiple of the 0.01 s physics step')
        if abs(self.commit/self.dt-round(self.commit/self.dt))>1e-6:raise ValueError('Commit must be divisible by dt')
        if not 0<self.optimism<=1 or not 0<=self.mixture<1:raise ValueError('Invalid optimism/mixture')

    @property
    def steps(self):return round(self.horizon/self.dt)
    @property
    def commit_steps(self):return round(self.commit/self.dt)

CENTERS=np.array([[-28.,8.],[28.,-10.],[-28.,-18.],[28.,27.]],np.float32)
PASSAGES=[8.,-23.,27.]
BOUNDS=np.array([[-37.,-25.5],[37.,34.5]],np.float32)
RECTANGLES=np.array([[-19.,-20.,19.,5.5,18.],[-19.,10.5,19.,22.,15.]],np.float32)


def environment(layout='two',scale=1.,inspection_scale=1.,case='forward'):
    # At the smallest supported scale, passage A is 1.75 m across for the
    # 0.55 m aircraft. Building heights and vehicle dimensions do not shrink.
    if not .35<=scale<=1.5:raise ValueError('Geometry scale must preserve usable drone passages (0.35 to 1.5)')
    if not .4<=inspection_scale<=1.5:raise ValueError('Invalid inspection region scale')
    centers=(CENTERS[:2] if layout=='two' else CENTERS)*scale
    half_size=np.tile(np.array([4.5,5.])*inspection_scale,(len(centers),1))
    # The larger eastern task region is physical area, represented by extra
    # equally weighted monitoring cells. No route preference or turn-back
    # instruction enters the objective or the geometric initialization.
    if case in ('commitment','calm_a'):half_size[1,1]*=2
    regions=np.c_[centers-half_size,centers+half_size]
    cells=[]
    for i,(x,z) in enumerate(centers):
        larger=case in ('commitment','calm_a') and i==1
        yy=np.linspace(-8.4,8.4,13) if larger else np.linspace(-4.2,4.2,7)
        xx,zz=np.meshgrid(x+np.linspace(-3.6,3.6,6)*inspection_scale,z+yy*inspection_scale)
        cells.extend(np.c_[xx.ravel(),zz.ravel()])
    cells=jnp.array(np.array(cells),jnp.float32)
    rectangles=RECTANGLES.copy();rectangles[:,:4]*=scale
    bounds=BOUNDS*scale
    # Inspection footprints and the aircraft retain their physical dimensions.
    bounds[0]=np.minimum(bounds[0],regions[:,:2].min(axis=0)-.65)
    bounds[1]=np.maximum(bounds[1],regions[:,2:].max(axis=0)+.65)
    return UrbanEnvironment(jnp.array(rectangles),jnp.array(bounds),cells,jnp.full(len(cells),.75),jnp.full(len(cells),.014),jnp.array(regions))


def wind(p,case='forward',strength=3.1,scale=1.):
    p=jnp.atleast_2d(p)/scale;x,z=p[:,0],p[:,1]
    axial=smooth_window(x,-25,25,5)
    A=axial*smooth_window(z,5.4,10.6,.8)
    C=axial*smooth_window(z,22.,34.5,2.)
    if case in ('forward','reverse'):
        sign=1. if case=='forward' else -1.
        return strength*jnp.stack((sign*A,.09*A*jnp.sin(x/5)),axis=-1)
    if case=='crosswind':
        return strength*jnp.stack((-.80*A-.86*C,.55*A*jnp.cos(x/6)+.45*C*jnp.sin((x+4)/7)),axis=-1)
    if case in ('commitment','calm_a'):
        B=axial*smooth_window(z,-25.5,-20.,.8)
        # A spatial velocity field across all three passages: eastward travel
        # is assisted and westward return opposes the flow. The planner is
        # never given the case name or this physical field, only observations.
        active_A=A if case=='commitment' else jnp.zeros_like(A)
        return strength*jnp.stack((active_A+B+C,.09*(active_A-C)*jnp.sin(x/5)),axis=-1)
    raise ValueError(case)


def initial_belief(cfg):
    # Sites and corridors receive inducing locations, without oracle values.
    xx,zz=np.meshgrid(np.linspace(-31,31,9),[-23.,-10.,8.,27.])
    scale=getattr(cfg,'environment_scale',1.)
    support=None
    nodes=getattr(cfg,'wind_support_nodes',0)
    if nodes:
        # Numerical prior-covariance anchors, not wind observations. Exact
        # support covariance plus the spectral residual preserve a continuous
        # full GP draw, rather than truncating the field to inducing means.
        sx,sy=np.meshgrid(np.linspace(-45,45,nodes)*scale,np.linspace(-35,44,nodes)*scale)
        support=np.c_[sx.ravel(),sy.ravel()].astype(np.float32)
    return POAM(np.c_[xx.ravel(),zz.ravel()].astype(np.float32)*scale,jax.random.PRNGKey(731),
                noise_std=cfg.observation_std,lengths=(5.*scale,13.*scale),prior_std=cfg.prior_std,spectral_count=16,prior_support=support).planning_belief()

class ExtendedTableBasis(NamedTuple):
    lower:jax.Array
    spacing:jax.Array

    def evaluate(self,locations,field):
        # The physical experiment ends at its environment boundary, inside this
        # table. Soft-penalty forecasts and backup-policy evaluations can leave
        # that domain. A constant boundary extension keeps their already-large
        # violation costs finite instead of accidentally treating them as NaN.
        # Interior interpolation and its derivatives are exactly unchanged.
        upper=self.lower+self.spacing*jnp.array([field.shape[1]-2.001,field.shape[0]-2.001])
        clipped=jnp.clip(jnp.atleast_2d(locations),self.lower+self.spacing,upper)
        return TableBasis(self.lower,self.spacing).evaluate(clipped,field)

TABLE_BASIS=ExtendedTableBasis(jnp.array([-45.,-35.]),jnp.array([1.,1.]))
xx,zz=np.meshgrid(np.arange(-45.,46.),np.arange(-35.,45.))
TABLE_POINTS=jnp.array(np.c_[xx.ravel(),zz.ravel()],jnp.float32)
TABLE_SHAPE=(80,91,2)

def table_basis(scale=1.):
    return ExtendedTableBasis(TABLE_BASIS.lower*scale,TABLE_BASIS.spacing*scale)

# Separate factories allow static sample shapes without recompiling at each call.
def table_sampler(count,scale=1.):
    @jax.jit
    def sample(b,key):
        coeff=b.sample_field(key,count)
        fields=jax.vmap(lambda c:b.basis.evaluate(TABLE_POINTS*scale,c))(coeff).reshape((count,*TABLE_SHAPE))
        mean=b.mean_at(TABLE_POINTS*scale).reshape(TABLE_SHAPE)
        return mean,fields
    return sample


def metadata(layout='two',scale=1.,inspection_scale=1.,case='forward'):
    e=environment(layout,scale,inspection_scale,case)
    counts=[int(np.sum(np.all(np.asarray(e.cells)>=r[:2],axis=1)&np.all(np.asarray(e.cells)<=r[2:],axis=1))) for r in np.asarray(e.regions)]
    return dict(layout=layout,case=case,horizontal_scale=scale,inspection_scale=inspection_scale,bounds=np.asarray(e.bounds).tolist(),buildings=np.asarray(e.rectangles).tolist(),regions=np.asarray(e.regions).tolist(),region_cell_counts=counts,cells=np.asarray(e.cells).tolist(),targets=np.asarray(e.targets).tolist(),process_noise=np.asarray(e.process_noise).tolist(),passages=(np.array(PASSAGES)*scale).tolist(),flight_height=3.2,drone_span=.55)
