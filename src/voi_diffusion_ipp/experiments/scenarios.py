"""Three fixed-physics scenarios. Truth uses the same correlated basis in all ablations."""
import jax.numpy as jnp
from ..environment import Environment
from ..wind import prior


def make_scenario(cfg):
    a = jnp.array([1.15,0.]); b=jnp.array([-1.45,-.6])
    offsets=jnp.array([[-.16,-.16],[0,-.16],[.16,-.16],[-.16,0],[0,0],[.16,0],[-.16,.16],[0,.16],[.16,.16]])
    cells=jnp.concatenate((a+offsets,b+offsets[:4]))
    targets=jnp.concatenate((jnp.full(9,.8),jnp.full(4,.65)))
    obstacles=jnp.array([[0.,1.05,.79],[0.,-1.05,.79]])
    bounds=jnp.array([[-2.4,-1.85],[2.4,1.85]])
    probe=jnp.array([-.55,0.])
    if cfg.scenario == 'irrelevant_uncertainty':
        # Useful monitoring is entirely on the left; uncertainty to the right has no task value.
        cells=cells.at[:9].set(jnp.array([-1.35,.5])+offsets)
        a=jnp.array([-1.35,.5])
    elif cfg.scenario not in ('corridor','no_worthwhile_learning'):
        raise ValueError(f'Unknown scenario {cfg.scenario}')
    env=Environment(cells,targets,obstacles,bounds,a,b,probe)
    cx,cy=jnp.meshgrid(jnp.linspace(-2,2,5),jnp.linspace(-1.5,1.5,3))
    centers=jnp.stack((cx.ravel(),cy.ravel()),axis=-1)
    noise=cfg.observation_std if cfg.scenario!='no_worthwhile_learning' else 10.
    belief=prior(centers,cfg.wind_length,cfg.wind_std,noise,cfg.independent_cells)
    if cfg.scenario=='irrelevant_uncertainty':
        # Low uncertainty on useful left routes, large uncertainty beyond the passage.
        std=jnp.where(centers[:,0]<0,.08,.9)
        belief=belief._replace(covariance=jnp.diag(std**2))
    x=jnp.array([-1.45,-.15,0.,0.])
    q=jnp.full(cells.shape[0],.15)
    return env,belief,x,q
