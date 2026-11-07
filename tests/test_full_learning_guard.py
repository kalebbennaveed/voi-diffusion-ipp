"""Check the added obstacle guard against the exact double-integrator derivative."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
import numpy as np
import jax
import jax.numpy as jnp
from types import SimpleNamespace
import full_learning_monitoring as model

class ConstantBasis:
    def evaluate(self,p,values):return jnp.atleast_2d(values)
class ConstantMean:
    def __init__(self,w):self.w=jnp.asarray(w)
    def mean_at(self,p):return jnp.atleast_2d(self.w)

def test_obstacle_guard_exact_distance_hessian_and_wind_support():
    x=jnp.array([2.,1.,-.3,.2]);ob=jnp.array([[0.,0.,.7],[3.,2.,.3]])
    env=SimpleNamespace(obstacles=ob);cfg=SimpleNamespace(safety_margin=.06)
    mean=ConstantMean([1.,-.8]);sigma=.2;basis=ConstantBasis()
    A,b=model.obstacle_constraints(x,mean,jnp.array([[sigma]]),basis,env,cfg)
    h=lambda z:jnp.linalg.norm(z[:2]-ob[:,:2],axis=1)-ob[:,2]-.06
    grad=jax.jacfwd(h)(x);H=jax.jacfwd(jax.jacfwd(h))(x)
    np.testing.assert_allclose(A,-grad[:,:2],atol=1e-12)
    R=3*np.sqrt(sigma*sigma+.01);u=jnp.array([.3,-.2])
    for angle in np.linspace(0,2*np.pi,47):
        w=mean.w+R*jnp.array([np.cos(angle),np.sin(angle)])
        d=(.1/.680)*jnp.linalg.norm(w-x[2:])*(w-x[2:])
        hdot=grad[:,:2]@x[2:]
        hddot=grad[:,:2]@(u+d)+jnp.einsum('i,kij,j->k',x[2:],H[:,:2,:2],x[2:])
        exact=hddot+3*hdot+2.25*h(x)
        assert np.all(np.asarray(exact)>=np.asarray(b-A@u)-1e-10)


def test_hard_filter_enforces_actuator_box_beyond_qp_tolerance():
    import cluster_environment as ce
    cfg=model.settings()
    env,_,_=ce.environment(model.legacy.physical_settings(cfg),model.m.physical.ROOT/'results/airflow/cluster_revision_20261103/v2/environment-spec.json')
    basis,points,shape=model.table_grid(env.bounds,.5)
    mean=model.TableMean(basis,jnp.zeros(shape))
    fields=model.InnerFields(basis,jnp.zeros((cfg.construction_worlds,*shape)))
    std=jnp.zeros((*shape[:2],1));x=jnp.array([1.45,.875,0.,0.])
    control,info=model.control(x,jnp.array([cfg.umax+1e-5,0.]),mean,fields,std,basis,env,cfg,True)
    assert np.max(np.abs(np.asarray(control)))<=cfg.umax
    assert bool(info.feasible)
