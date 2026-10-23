"""Ground-facing footprint: Gaussian quality inside a finite camera cone."""
import jax.numpy as jnp
from ..clarity import deficit

def rate(q,p,env,cfg,height=3.2):
    delta=env.cells-p;r2=jnp.sum(delta*delta,axis=1)
    # Block visibility through a building, independently of inspection labels.
    lo=env.rectangles[None,:,:2];hi=env.rectangles[None,:,2:4]
    dd=delta[:,None,:];origin=p[None,None,:]
    nonzero=jnp.abs(dd)>1e-8;safe=jnp.where(nonzero,dd,1.)
    t1=(lo-origin)/safe;t2=(hi-origin)/safe
    near=jnp.where(nonzero,jnp.minimum(t1,t2),jnp.where((origin>=lo)&(origin<=hi),-jnp.inf,jnp.inf))
    far=jnp.where(nonzero,jnp.maximum(t1,t2),jnp.where((origin>=lo)&(origin<=hi),jnp.inf,-jnp.inf))
    blocked=jnp.any(jnp.maximum(jnp.max(near,axis=2),0.)<=jnp.minimum(jnp.min(far,axis=2),1.),axis=1)
    radius=jnp.maximum(height,0.)*jnp.tan(jnp.deg2rad(cfg.sensor_half_angle_degrees))
    gain=jnp.exp(-r2/cfg.sensor_sigma**2)/cfg.sensor_R*(r2<=radius**2)*(~blocked)
    return gain*(1-q)**2-env.process_noise*q*q

def step(q,p0,p1,env,cfg,dt,height=3.2):
    f=lambda qq,pp:rate(qq,pp,env,cfg,height)
    a=f(q,p0);b=f(q+dt*a/2,(p0+p1)/2);c=f(q+dt*b/2,(p0+p1)/2);d=f(q+dt*c,p1)
    return jnp.clip(q+dt*(a+2*b+2*c+d)/6,0,1)
