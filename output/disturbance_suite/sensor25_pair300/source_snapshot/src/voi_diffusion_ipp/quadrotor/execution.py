"""Compiled execution of the SAME lab rigid-body equations and tracker.

This accelerates integration; it does not replace the plant with a kinematic
animation. The physical field enters the plant only. Feedback gets a frozen
posterior mean and measured state, never the physical load.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .config import VehicleConfig,ControllerConfig
from .controller import allocation_matrix,geometric_wrench
from .dynamics import rigid_body_derivative
from .reference import flat_reference
from .types import State
from .aerodynamics import acceleration as air_acceleration


def executor(load,segment_dt=.4,segments=5,dt=.01,max_horizontal_accel=None,wind_velocity=False):
    vehicle=VehicleConfig();gains=ControllerConfig(max_horizontal_accel=max_horizontal_accel,
                                                 limit_total_horizontal_force=wind_velocity)
    allocation=jnp.asarray(allocation_matrix(vehicle));inverse=jnp.linalg.inv(allocation)
    def execute(packed,polynomials,mean):
        def tick(x,i):
            state=State.unpack(x)
            segment=i//round(segment_dt/dt)
            t=(i%round(segment_dt/dt))*dt
            c=polynomials[segment]
            values=[]
            for _ in range(5):
                values.append(jnp.sum(c*t**jnp.arange(len(c))[:,None],axis=0))
                c=c[1:]*jnp.arange(1,len(c))[:,None]
            p,v,a,j,s=values
            learned=jnp.r_[mean.mean_at(state.position[None,:2])[0],0.]
            speed=jnp.sqrt(jnp.maximum(jnp.sum(state.velocity**2),1e-24))
            drag=vehicle.linear_drag*speed*state.velocity/vehicle.mass
            compensation=-air_acceleration(state.velocity,learned,xp=jnp) if wind_velocity else drag-learned
            ref=flat_reference(p,v,a,j,s,compensation=compensation,xp=jnp)
            wrench=geometric_wrench(state.position,state.velocity,state.rotation,state.angular_velocity,
                                   ref,vehicle,gains,xp=jnp)
            command=jnp.sqrt(jnp.clip(inverse@wrench,0,vehicle.max_motor_speed**2))
            # The coherent spatial field is queried at every RK4 substage.
            def derivative(xx):
                if wind_velocity:
                    wind=jnp.r_[load(xx[None,:2])[0],0.]
                    return rigid_body_derivative(xx,command,vehicle,allocation,wind,jnp.zeros(3),xp=jnp)
                force=vehicle.mass*jnp.r_[load(xx[None,:2])[0],0.]
                return rigid_body_derivative(xx,command,vehicle,allocation,jnp.zeros(3),force,xp=jnp)
            k1=derivative(x);k2=derivative(x+dt*k1/2);k3=derivative(x+dt*k2/2);k4=derivative(x+dt*k3)
            xn=x+dt*(k1+2*k2+2*k3+k4)/6
            u,_,vt=jnp.linalg.svd(xn[6:15].reshape(3,3),full_matrices=False)
            R=u@jnp.diag(jnp.array([1.,1.,jnp.linalg.det(u@vt)]))@vt
            xn=xn.at[6:15].set(R.ravel())
            return xn,(xn,p,command,k1[3:6])
        return jax.lax.scan(tick,packed,jnp.arange(round(segments*segment_dt/dt)))
    return jax.jit(execute)
