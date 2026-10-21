"""Independent checks of the monitoring dynamics questioned in the run audit."""
import numpy as np
import jax,jax.numpy as jnp
from scipy.integrate import solve_ivp
from voi_diffusion_ipp.quarter.environment import environment
from voi_diffusion_ipp.quarter.continuous import ControlSettings
from voi_diffusion_ipp.quarter import sensing
from voi_diffusion_ipp.clarity import deficit

def test_stationary_footprint_clarity_matches_independent_riccati_integration():
    cfg=ControlSettings(environment_scale=.35,inspection_scale=.55)
    env=environment('two',.35,.55);position=(env.regions[0,:2]+env.regions[0,2:])/2
    q=jnp.full(len(env.cells),.35)
    integrate=jax.jit(lambda q:jax.lax.fori_loop(0,100,lambda _,q:sensing.step(q,position,position,env,cfg,.2),q))
    got=np.asarray(integrate(q));distance=np.linalg.norm(np.array(env.cells-position),axis=1)
    gain=np.exp(-distance**2/cfg.sensor_sigma**2)/cfg.sensor_R*(distance<=3.2)
    independent=solve_ivp(lambda t,y:gain*(1-y)**2-np.array(env.process_noise)*y*y,(0,20),np.full(len(q),.35),rtol=1e-11,atol=1e-12).y[:,-1]
    np.testing.assert_allclose(got,independent,atol=1e-5)
    # Remote cells receive no observation and follow exact unsensed decay.
    np.testing.assert_allclose(got[42:],.35/(1+.014*.35*20),atol=2e-6)
    assert np.all(got[:42]>.75)

def test_deficit_is_per_cell_and_cannot_cancel_poorly_monitored_cells():
    assert np.isclose(float(deficit(jnp.array([.9,.6]),jnp.array([.75,.75]))),.075)
