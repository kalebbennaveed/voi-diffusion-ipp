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
from ..wind.cache import TableBasis

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
    sensor_half_angle_degrees: float=45.
    observation_std: float=.25
    wind_strength: float=3.1
    prior_std: float=1.8
    seed: int=8101

    def __post_init__(self):
        if not 0<self.sensor_half_angle_degrees<90:raise ValueError('Sensor half-angle must be between 0 and 90 degrees')
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


def environment(layout='two',scale=1.,inspection_scale=1.,case='forward',geometry='quarter',detour_depth=2.):
    if geometry=='suite':
        from .benchmark_environments import make_environment
        return make_environment(case)
    if geometry in ('compact','compact_passages','compact_detour'):
        if layout!='two':raise ValueError('Compact development case has two inspection regions')
        centers=np.array([[-4.6,1.],[4.6,3.7]],np.float32)
        half=np.array([1.05,1.2],np.float32)
        regions=np.c_[centers-half,centers+half]
        offsets=np.array(np.meshgrid(np.linspace(-.9,.9,6),np.linspace(-1.05,1.05,7))).reshape(2,-1).T
        cells=np.concatenate([center+offsets for center in centers])
        rectangles=np.array([[-2.,-2.4,2.,-.2,12.],[-2.,2.2,2.,4.4,14.]],np.float32)
        bounds=np.array([[-6.7,-5.],[6.7,7.]],np.float32)
        if geometry=='compact_passages':
            # A is the middle passage, B the longer upper detour, C lower.
            # Keep the physical aircraft and sites unchanged, narrow A/C.
            rectangles[0,3]=.25;rectangles[1,1]=1.95;bounds[0,1]=-4.2
        elif geometry=='compact_detour':
            if not 0<=detour_depth<=3:raise ValueError('Detour depth must lie in [0,3] metres')
            # Extend the building separating middle A from upper B. The
            # upper passage retains its width while becoming farther away.
            rectangles[1,3]+=detour_depth;bounds[1,1]+=detour_depth
        return UrbanEnvironment(jnp.array(rectangles),jnp.array(bounds),
            jnp.array(cells),jnp.full(len(cells),.75),jnp.full(len(cells),.05),jnp.array(regions))
    if geometry!='quarter':raise ValueError(f'Unknown geometry: {geometry}')
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


def wind(p,case='forward',strength=3.1,scale=1.,geometry='quarter',wind_at_start=False):
    if geometry=='suite':
        from .benchmark_environments import BY_NAME,flow,flow_parameters
        return flow(p,flow_parameters(BY_NAME[case]))
    if geometry in ('compact','compact_passages','compact_detour'):
        p=jnp.atleast_2d(p);x,z=p[:,0],p[:,1]
        if geometry=='compact_detour' and case=='crosswind':
            axial=smooth_window(x,-6.7 if wind_at_start else -3.6,3.6,.9)
            A=axial*smooth_window(z,-.2,2.2,.35)
            C=axial*smooth_window(z,-5.,-2.4,.35)
            # Broad adverse longitudinal flow, with weaker spatial shear.
            # The upper detour B is calm; neither field nor labels enter MBD.
            return strength*jnp.stack((-(A+C),.12*A*jnp.sin(x/1.5)-.10*C*jnp.cos(x/2)),axis=-1)
        if geometry=='compact_passages' and case=='crosswind':
            # Synthetic static air-velocity jets, not CFD. They fill the
            # physically usable cross section of A/C; B is exactly calm.
            axial=smooth_window(x,-3.,3.,1.)
            A=axial*smooth_window(z,.25,1.95,.25)
            C=axial*smooth_window(z,-4.2,-2.4,.25)
            return jnp.stack((.15*(A+C)*jnp.sin(x/1.5),strength*(A-C)),axis=-1)
        axial=smooth_window(x,-3.6,3.6,.9)
        A=axial*smooth_window(z,.25,1.95,.25) if geometry=='compact_passages' else axial*smooth_window(z,-.2,2.2,.35)
        if case not in ('forward','reverse'):raise ValueError('Compact development uses matched forward/reversed wind')
        sign=1. if case=='forward' else -1.
        return strength*jnp.stack((sign*A,.09*A*jnp.sin(x/1.5)),axis=-1)
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
    if cfg.geometry=='suite':
        count=getattr(cfg,'wind_inducing_count',36)
        observations=getattr(cfg,'initial_wind_observations',())
        if len(observations)>=count:
            centers=np.asarray(observations,dtype=np.float32)[:count,:2]
        else:
            side=int(np.ceil(np.sqrt(count)))
            xx,yy=np.meshgrid(np.linspace(-7,7,side),np.linspace(-7,7,side))
            centers=np.c_[xx.ravel(),yy.ravel()].astype(np.float32)[:count]
        nodes=cfg.wind_support_nodes
        sx,sy=np.meshgrid(np.linspace(-10,10,nodes),np.linspace(-10,10,nodes))
        lengths=getattr(cfg,'wind_kernel_lengths',()) or (1.4,3.2)
        belief=POAM(centers,jax.random.PRNGKey(731),
            noise_std=cfg.observation_std,lengths=lengths,prior_std=cfg.prior_std,
            spectral_count=16,prior_support=np.c_[sx.ravel(),sy.ravel()].astype(np.float32)).planning_belief()
        if observations:
            data=jnp.asarray(observations,jnp.float32)
            belief=belief.condition(data[:,:2],data[:,2:],relocate=cfg.wind_relocate)
        return belief
    # Sites and corridors receive inducing locations, without oracle values.
    compact=getattr(cfg,'geometry','quarter') in ('compact','compact_passages','compact_detour')
    xx,zz=(np.meshgrid(np.linspace(-5.5,5.5,9),[-3.5,1.,3.7,5.6]) if compact else
           np.meshgrid(np.linspace(-31,31,9),[-23.,-10.,8.,27.]))
    detour=getattr(cfg,'geometry','quarter')=='compact_detour'
    # A fixed numerical grid across detour-depth comparisons avoids changing
    # posterior sample realizations merely by moving the known building edge.
    if detour:zz[-1,:]+=2.
    scale=getattr(cfg,'environment_scale',1.)
    support=None
    nodes=getattr(cfg,'wind_support_nodes',0)
    if nodes:
        # Numerical prior-covariance anchors, not wind observations. Exact
        # support covariance plus the spectral residual preserve a continuous
        # full GP draw, rather than truncating the field to inducing means.
        sx,sy=(np.meshgrid(np.linspace(-9,9,nodes),np.linspace(-6,11.8,nodes)) if detour else
               np.meshgrid(np.linspace(-9,9,nodes),np.linspace(-8,7.8,nodes)) if compact else
               np.meshgrid(np.linspace(-45,45,nodes)*scale,np.linspace(-35,44,nodes)*scale))
        support=np.c_[sx.ravel(),sy.ravel()].astype(np.float32)
    belief=POAM(np.c_[xx.ravel(),zz.ravel()].astype(np.float32)*(1 if compact else scale),jax.random.PRNGKey(731),
                noise_std=cfg.observation_std,lengths=((1.2,2.4) if compact else (5.*scale,13.*scale)),prior_std=cfg.prior_std,spectral_count=16,prior_support=support).planning_belief()
    observations=getattr(cfg,'initial_wind_observations',())
    if observations:
        data=jnp.asarray(observations,jnp.float32)
        belief=belief.condition(data[:,:2],data[:,2:],relocate=getattr(cfg,'wind_relocate',False))
    return belief

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

def table_basis(scale=1.,geometry='quarter'):
    if geometry=='suite':return ExtendedTableBasis(jnp.array([-10.,-10.]),jnp.array([.25,.25]))
    if geometry=='compact_detour':return ExtendedTableBasis(jnp.array([-9.,-6.]),jnp.array([.2,.2]))
    if geometry in ('compact','compact_passages'):return ExtendedTableBasis(jnp.array([-9.,-8.]),jnp.array([.2,.2]))
    return ExtendedTableBasis(TABLE_BASIS.lower*scale,TABLE_BASIS.spacing*scale)


def table_shape(geometry='quarter'):
    # Add rows for the farther detour; retain the same physical resolution.
    if geometry=='suite':return (81,81,2)
    return (90,91,2) if geometry=='compact_detour' else TABLE_SHAPE

# Separate factories allow static sample shapes without recompiling at each call.
def table_sampler(count,scale=1.,geometry='quarter'):
    basis=table_basis(scale,geometry)
    shape=table_shape(geometry)
    yy,xx=jnp.meshgrid(jnp.arange(shape[0]),jnp.arange(shape[1]),indexing='ij')
    points=(TABLE_POINTS*scale if geometry=='quarter' else basis.lower+jnp.c_[xx.ravel(),yy.ravel()]*basis.spacing)
    @jax.jit
    def sample(b,key):
        coeff=b.sample_field(key,count)
        fields=jax.vmap(lambda c:b.basis.evaluate(points,c))(coeff).reshape((count,*shape))
        mean=b.mean_at(points).reshape(shape)
        return mean,fields
    return sample


def planning_sampler(cfg):
    """Draw exactly the used coherent fields, with stable controller base draws."""
    basis=table_basis(cfg.environment_scale,cfg.geometry)
    shape=table_shape(cfg.geometry)
    yy,xx=jnp.meshgrid(jnp.arange(shape[0]),jnp.arange(shape[1]),indexing='ij')
    points=(TABLE_POINTS*cfg.environment_scale if cfg.geometry=='quarter' else
            basis.lower+jnp.c_[xx.ravel(),yy.ravel()]*basis.spacing)
    @jax.jit
    def sample(b,performance_key,construction_key):
        construction=b.sample_field(construction_key,cfg.construction_worlds)
        performance=b.sample_field(performance_key,cfg.worlds+cfg.validation_worlds)
        coefficients=jnp.concatenate((construction,performance))
        fields=jax.vmap(lambda c:b.basis.evaluate(points,c))(coefficients).reshape((-1,*shape))
        return b.mean_at(points).reshape(shape),fields
    return sample


def metadata(layout='two',scale=1.,inspection_scale=1.,case='forward',geometry='quarter',detour_depth=2.):
    e=environment(layout,scale,inspection_scale,case,geometry,detour_depth)
    counts=[int(np.sum(np.all(np.asarray(e.cells)>=r[:2],axis=1)&np.all(np.asarray(e.cells)<=r[2:],axis=1))) for r in np.asarray(e.regions)]
    compact=geometry in ('compact','compact_passages','compact_detour','suite')
    labels=({'A':1.,'B':5.6+detour_depth,'C':-3.6} if geometry=='compact_detour' else
            {'A':1.1,'B':5.6,'C':-3.3} if geometry=='compact_passages' else None)
    return dict(layout=layout,case=case,geometry=geometry,detour_depth=detour_depth if geometry=='compact_detour' else None,horizontal_scale=scale,inspection_scale=inspection_scale,bounds=np.asarray(e.bounds).tolist(),buildings=np.asarray(e.rectangles).tolist(),regions=np.asarray(e.regions).tolist(),region_cell_counts=counts,cells=np.asarray(e.cells).tolist(),targets=np.asarray(e.targets).tolist(),process_noise=np.asarray(e.process_noise).tolist(),passages=([] if geometry=='suite' else [1.,-3.6,5.6+(detour_depth if geometry=='compact_detour' else 0)] if compact else (np.array(PASSAGES)*scale).tolist()),passage_labels=labels,flight_height=3.2,drone_span=.55)
