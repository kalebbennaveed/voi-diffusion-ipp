"""Distinct persistent-monitoring tasks for disturbance-aware planning.

All fields are declared synthetic spatial air velocities, not CFD. The first
pair preserves the compact two-site geometry and exact opposite middle-passage
winds. New tasks vary topology and the distribution of monitoring cells.
Physical vehicle dimensions, known dynamics and sensing are unchanged.
"""
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from ..experiments.urban_quad import UrbanEnvironment

SCENARIO_REVISION='monitoring-suite-v9-traversable-service-aisles'

@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    topology: str
    task: str
    mechanism: str
    rectangles: tuple
    patches: tuple
    jets: tuple
    vortices: tuple = ()
    strength: float = 3.5
    bounds: tuple = ((-8., -8.), (8., 8.))
    original_sign: int = 0


def patch(x,y,hx=.8,hy=.8):
    return (x-hx,y-hy,x+hx,y+hy)

# Jet: centre x/y, support half-width x/y, velocity direction x/y.
SCENARIOS = (
 Scenario('original','Original A/B monitoring','Two obstacles, three passages',
  'Two compact inspection regions','Middle-passage wind changes transfer speed and circulation direction',
  ((-2,-2.4,2,-.2,12),(-2,2.2,2,4.4,14)),
  (patch(-4.6,1,1.05,1.2),patch(4.6,3.7,1.05,1.2)),(),strength=3.65,
  bounds=((-6.7,-5),(6.7,7)),original_sign=1),
 Scenario('reversed','Original A/B, reversed wind','Exact matched geometry and target cells',
  'Same monitoring task, exact negative physical field','Reversal tests whether route use responds to disturbance direction',
  ((-2,-2.4,2,-.2,12),(-2,2.2,2,4.4,14)),
  (patch(-4.6,1,1.05,1.2),patch(4.6,3.7,1.05,1.2)),(),strength=3.65,
  bounds=((-6.7,-5),(6.7,7)),original_sign=-1),
 Scenario('courtyards','Four-courtyard district','Six staggered building blocks; intersecting streets',
  'Four spatially separated courtyards','A circulating street flow changes the profitable order of four courtyard visits',
  ((-4.2,-2.8,-1.2,-.5,13),(1.2,-2.8,4.2,-.5,16),
   (-4.2,1.5,-1.2,3.8,15),(1.2,1.5,4.2,3.8,12),
   (-.5,-6,.7,-4,11),(-.5,5,.7,7,14)),
  (patch(-5.7,-4.6),patch(5.7,-4.6),patch(5.7,5.6),patch(-5.7,5.6)),
  ((0,-4.6,6.8,1.4,1,0),(5.7,.5,1.3,6.5,0,1),(0,5.6,6.8,1.3,-1,0),(-5.7,.5,1.3,6.5,0,-1)),strength=3.6),
 Scenario('exhaust','Industrial exhaust inspection','Eight process blocks with offset gates',
  'A sheltered staging site and five equipment inspection patches','Local exhaust jets disturb direct close inspection; lateral standoff views remain feasible',
  ((-4.7,-4.3,-2.7,-2.3,9),(-.8,-4.3,1.2,-2.3,11),(3.1,-4.3,5.1,-2.3,8),
   (-4.7,1.2,-2.7,3.2,14),(-.8,1.2,1.2,3.2,10),(3.1,1.2,5.1,3.2,12),
   (-1.1,5.1,.7,6.5,9),(-1.1,-6.8,.7,-5.4,10)),
  (patch(-6.4,-5.8,.65,.65),patch(-1.8,-.6,.55,.85),patch(2.1,-.6,.55,.85),
   patch(6.2,-.6,.55,.85),patch(-3.7,4.5,.8,.55),patch(4.1,4.5,.8,.55)),
  ((-1.8,-.6,.75,2.3,0,-1),(2.1,-.6,.75,2.3,0,1),
   (6.2,-.6,.75,2.3,0,-1),(-3.7,4.1,1.6,.65,1,0),(4.1,4.1,1.6,.65,-1,0)),strength=4.0),
 Scenario('slalom','Serpentine street survey','Four alternating long barriers and two corner blocks',
  'Continuous narrow street segments rather than two target boxes','Opposite outer-street flows and a sheltered central street change survey order and approach direction',
  ((-5.5,-5.3,-3.9,1.2,14),(-2.4,-1.2,-.8,5.3,12),(.8,-5.3,2.4,1.2,15),
   (3.9,-1.2,5.5,5.3,13),(-7,4,-6,6,10),(6,-6,7,-4,10)),
  ((-3.45,-5.8,-2.85,2.2),(-.3,-2.2,.3,5.8),(2.85,-5.8,3.45,2.2),(-6.8,2,-5.8,3.3),(5.8,-3.3,6.8,-2)),
  ((-3.2,-.5,1,5.3,0,1),(3.2,-.5,1,5.3,0,-1)),strength=3.6),
 Scenario('atrium','Atrium ring and gated courtyards','Nested ring walls, two entrances, inner pavilion',
  'Five patches inside and outside the ring','South and east entrance jets penalize direct transfers; the west inner route remains sheltered',
  ((-4.6,-4.6,-1,-3.2,13),(1,-4.6,4.6,-3.2,13),(-4.6,3.2,4.6,4.6,15),
   (-4.6,-3.2,-3.2,3.2,12),(3.2,-3.2,4.6,.1,12),(3.2,2.1,4.6,3.2,12),
   (-.8,-.8,.8,.8,9)),
  (patch(-.1,-6),patch(6,1.1,.75,.65),patch(0,1.9,.8,.55),patch(-1.9,0,.55,.75),patch(0,-1.9,.8,.55)),
  ((0,-2,3,.85,1,0),(2,0,.85,3,0,1)),strength=3.6),
 Scenario('freight','Freight-terminal service network','Six offset cargo rows, two loading barriers and multiple cross-aisles',
  'A sheltered staging bay and five service areas inside and around the racks','A central loading-lane jet penalizes direct transfers; exterior cross-aisles provide longer sheltered connections',
  ((-5.8,-5.3,-3.8,-1.7,9),(-1.6,-5.3,-.1,-2,10),(2.1,-5.3,3.7,-1.7,9),
   (-5.8,1.4,-3.8,5.3,10),(-1.6,2.2,-.1,5.3,9),(2.1,1.4,3.7,5.3,11),
   (5.8,-4.2,6.7,-2.2,8),(5.8,3,6.7,5,8)),
  (patch(-6.6,-.1,.45,.8),patch(-2.7,-4.5,.35,.75),patch(1,-4.5,.35,.75),
   patch(5,-.1,.5,.8),patch(-2.7,4.4,.35,.75),patch(1,4.4,.35,.75)),
  ((0,-.1,5.9,1.25,-1,0),),strength=3.65),
 Scenario('perimeter','Shielded-facade perimeter patrol','Large central structure, five detached annexes',
  'Distributed perimeter samples along four facades','Corner jets and lee-side shelter alter where the robot can hold useful sensing positions',
  ((-2.,-2.,2.,2.,18),(-6.8,-5.8,-4.4,-3.6,12),(4.3,-5.8,6.5,-3.6,14),
   (-6.8,3.8,-4.5,6,10),(4.4,3.8,6.8,6,12),(-1,5.7,1,7,13)),
  ((-1.8,-3.6,1.8,-2.9),(2.9,-1.8,3.6,1.8),(-1.8,2.9,1.8,3.6),(-3.6,-1.8,-2.9,1.8)),
  ((0,-3.25,4.2,.9,-1,0),(3.25,0,.9,4.2,0,-1),(0,3.25,4.2,.9,1,0),(-3.25,0,.9,4.2,0,1)),strength=3.6),
 Scenario('interchange','Three-branch infrastructure interchange','Eight horizontal and vertical branch-separating blocks',
  'Six critical junction and branch-end patches','Opposing horizontal branch flows penalize direct transfers; sheltered north and south branches create longer connections',
  ((-5.2,-2.6,-1.2,-1.1,14),(-5.2,1.1,-1.2,2.6,15),(1.2,-2.6,5.2,-1.1,13),(1.2,1.1,5.2,2.6,16),
   (-2.6,3.9,-1.1,6.5,12),(1.1,3.9,2.6,6.5,12),(-2.6,-6.5,-1.1,-3.9,10),(1.1,-6.5,2.6,-3.9,10)),
  (patch(-6.3,0),patch(6.3,0),patch(0,6.3),patch(0,-6.3),patch(-.1,2.9,.65,.5),patch(.1,-2.9,.65,.5)),
  ((-3,0,4.1,.95,1,0),(3,0,4.1,.95,-1,0)),
  ((0,0,2.3,1.7),),strength=3.65),
 Scenario('distributed','Distributed mixed-priority survey','Eight staggered islands with connected open loops',
  'Six unequal target areas; many cells spread across the environment','A diagonal flow band intersects the large target and short connections, while edge loops provide shelter',
  ((-5.5,-2.6,-3.6,-.5,14),(-1.8,-3.5,.4,-1.6,11),(2.3,-2.6,4.3,-.5,13),
   (-4.9,2,-2.6,3.8,15),(-.7,1.2,1.5,3.3,10),(3.7,2.2,5.5,4.1,12),
   (-1.5,5.5,.5,6.8,9),(4.8,-6.6,6.5,-4.8,10)),
  (patch(-6.4,-4.5,.5,.9),patch(1.7,-5.7,1.5,.65),patch(6.4,.7,.45,1.8),
   patch(-6.1,5.8,.7,.8),patch(2.2,5.4,1,.7),patch(.5,-.2,.75,.55)),
  ((0,-.2,7,1.4,-1,0),(6.2,.5,1,3.5,-.25,-1),(1.7,-5.5,2.8,1,-1,.2)),
  ((-5,5,2.5,1.1),),strength=3.5),
)
BY_NAME={s.name:s for s in SCENARIOS}


def flow(p, parameters):
    """Analytic synthetic velocity; parameters may be dynamic JAX arguments."""
    jets,vortices,strength,sign=parameters
    p=jnp.atleast_2d(p);x,y=p[:,0],p[:,1]
    from ..experiments.urban_quad import smooth_window
    # Original case remains exactly the old compact middle-passage field.
    A=smooth_window(x,-3.6,3.6,.9)*smooth_window(y,-.2,2.2,.35)
    original=strength*jnp.stack((sign*A,sign*.09*A*jnp.sin(x/1.5)),axis=-1)
    delta=p[:,None,:]-jets[None,:,:2]
    half=jnp.maximum(jets[:,2:4],1e-4)
    edge=jnp.minimum(.6,half*.35)
    wx=smooth_window(delta[:,:,0],-half[None,:,0],half[None,:,0],edge[None,:,0])
    wy=smooth_window(delta[:,:,1],-half[None,:,1],half[None,:,1],edge[None,:,1])
    w=wx*wy
    u=w@jets[:,4:6]
    vdelta=p[:,None,:]-vortices[None,:,:2]
    radius=jnp.maximum(vortices[:,2],1e-4)
    weight=jnp.exp(-jnp.sum(vdelta*vdelta,axis=-1)/(2*radius[None]**2))
    circular=jnp.stack((-vdelta[:,:,1],vdelta[:,:,0]),axis=-1)/radius[None,:,None]
    u+=jnp.sum(weight[:,:,None]*circular*vortices[None,:,3,None],axis=1)
    magnitude=jnp.sqrt(jnp.maximum(jnp.sum(u*u,axis=-1,keepdims=True),1e-12))
    u=strength*u/jnp.maximum(1.,magnitude)
    return jnp.where(sign!=0,original,u)


def flow_parameters(s):
    jets=np.zeros((8,6),np.float32);jets[:,2:4]=1
    for i,j in enumerate(s.jets):jets[i]=j
    vortices=np.zeros((3,4),np.float32);vortices[:,2]=1
    for i,v in enumerate(s.vortices):vortices[i]=v
    return (jnp.asarray(jets),jnp.asarray(vortices),jnp.array(s.strength),jnp.array(s.original_sign,dtype=jnp.float32))


def make_environment(name):
    s=BY_NAME[name];rect=list(s.rectangles)
    while len(rect)<8:rect.append((50+len(rect),50,50.5+len(rect),50.5,1))
    patches=np.asarray(s.patches,np.float32);cells=[]
    # Equal area weights for ordinary patches; distributed survey explicitly
    # samples proportional to area, to avoid treating unequal sites as equal.
    area=np.prod(patches[:,2:]-patches[:,:2],axis=1)
    weights=area/area.sum() if name=='distributed' else np.full(len(patches),1/len(patches))
    counts=np.floor(weights*168).astype(int)
    for i in np.argsort(-(weights*168-counts))[:168-counts.sum()]:counts[i]+=1
    for patch_,count in zip(patches,counts):
        # Deterministic low-discrepancy coverage of each rectangle, in physical m.
        if s.original_sign:
            xx,yy=np.meshgrid(np.linspace(-.9,.9,6),np.linspace(-1.05,1.05,7))
            offsets=np.c_[xx.ravel(),yy.ravel()]
            cells.extend(np.tile(offsets,(2,1))+(patch_[:2]+patch_[2:])/2)
        else:
            k=np.arange(count)+.5;uv=np.c_[k/count,np.mod(k*.6180339887498949,1)]
            cells.extend(patch_[:2]+uv*(patch_[2:]-patch_[:2]))
    regions=list(s.patches)
    while len(regions)<6:regions.append((50,50,51,51))
    return UrbanEnvironment(jnp.asarray(rect,jnp.float32),jnp.asarray(s.bounds,jnp.float32),jnp.asarray(cells,jnp.float32),
        jnp.full(168,.75),jnp.full(168,.05),jnp.asarray(regions,jnp.float32))


def survey_locations(env,count=16):
    """Known-geometry space-filling station locations; never query field values."""
    lo,hi=np.asarray(env.bounds)
    xx,yy=np.meshgrid(np.linspace(lo[0]+.7,hi[0]-.7,11),np.linspace(lo[1]+.7,hi[1]-.7,11))
    points=np.c_[xx.ravel(),yy.ravel()]
    from .rrt import segment_free
    rect=np.asarray(env.rectangles)[:,:4]
    valid=np.array([p for p in points if segment_free(p,p,rect,np.asarray(env.bounds),.35)])
    selected=[int(np.argmin(np.linalg.norm(valid-(np.asarray(env.regions[0,:2])+np.asarray(env.regions[0,2:]))/2,axis=1)))]
    for _ in range(count-1):
        distance=np.min(np.linalg.norm(valid[:,None]-valid[np.asarray(selected)][None],axis=-1),axis=1)
        selected.append(int(np.argmax(distance)))
    return valid[selected].astype(np.float32)


def describe(name):
    s=BY_NAME[name]
    return dict(name=s.name,revision=SCENARIO_REVISION,title=s.title,topology=s.topology,task=s.task,disturbance_mechanism=s.mechanism,
        real_obstacles=len(s.rectangles),real_monitoring_regions=len(s.patches),
        synthetic_field=True,flow_units='m/s',vehicle='same nonlinear 22-state quadrotor and motors',
        planned_space='2D at 3.2 m flight height',monitoring_cells=168,nominal_wind_cap=s.strength,
        region_weighting='area-proportional' if name=='distributed' else 'equal per patch')
