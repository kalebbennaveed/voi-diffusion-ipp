"""Execute an MBD trajectory using the lab's nonlinear quadrotor equations.

The planar outer control/filter is recomputed from the actual reached state
every planning dt. Specific force is held between those samples. Flatness maps
that command to attitude on each smooth interval; the motor/attitude dynamics
resolve command jumps. No path interpolation changes the optimized decision.
"""
import jax
import jax.numpy as jnp
from ..quadrotor.config import VehicleConfig,ControllerConfig
from ..quadrotor.controller import allocation_matrix,geometric_wrench
from ..quadrotor.dynamics import rigid_body_derivative
from ..quadrotor.reference import flat_reference
from ..quadrotor.types import State
from ..wind.cache import TableMean
from ..wind.gp import InnerFields
from .environment import table_basis
from .continuous import nominal_reference,applied_control


def executor(env,cfg,load,filter_on=True,dt=.01):
    vehicle=VehicleConfig();gains=ControllerConfig(max_horizontal_accel=cfg.umax,limit_total_horizontal_force=True)
    allocation=jnp.asarray(allocation_matrix(vehicle));inverse=jnp.linalg.inv(allocation)
    ticks=round(cfg.dt/dt)
    if abs(ticks*dt-cfg.dt)>1e-8:raise ValueError('Execution dt must divide planning dt')
    @jax.jit
    def execute(packed,theta,mean_array,construction):
        x0=jnp.r_[packed[:2],packed[3:5]]
        reference,acceleration=nominal_reference(theta,x0,cfg)
        basis=table_basis(cfg.environment_scale)
        mean=TableMean(basis,mean_array);fields=InnerFields(basis,construction)
        def interval(packed,inputs):
            ref,a=inputs
            reached=jnp.r_[packed[:2],packed[3:5]]
            u,slack=applied_control(reached,ref,a,mean,fields,env,cfg,filter_on)
            def tick(packed,i):
                state=State.unpack(packed);t=i*dt
                p=jnp.r_[ref[:2]+t*ref[2:]+.5*t*t*a,3.2]
                v=jnp.r_[ref[2:]+t*a,0.]
                # The outer control already includes earned wind compensation.
                # Its held value supplies thrust direction; yaw is fixed.
                flat=flat_reference(p,v,jnp.r_[u,0.],xp=jnp)
                wrench=geometric_wrench(state.position,state.velocity,state.rotation,state.angular_velocity,
                    flat,vehicle,gains,xp=jnp,horizontal_acceleration=u)
                command=jnp.sqrt(jnp.clip(inverse@wrench,0,vehicle.max_motor_speed**2))
                def derivative(xx):
                    wind=jnp.r_[load(xx[None,:2])[0],0.]
                    return rigid_body_derivative(xx,command,vehicle,allocation,wind,jnp.zeros(3),xp=jnp)
                k1=derivative(packed);k2=derivative(packed+dt*k1/2);k3=derivative(packed+dt*k2/2);k4=derivative(packed+dt*k3)
                xn=packed+dt*(k1+2*k2+2*k3+k4)/6
                left,_,right=jnp.linalg.svd(xn[6:15].reshape(3,3),full_matrices=False)
                rotation=left@jnp.diag(jnp.array([1.,1.,jnp.linalg.det(left@right)]))@right
                xn=xn.at[6:15].set(rotation.ravel())
                return xn,(xn,p,command,k1[3:6],u,slack)
            return jax.lax.scan(tick,packed,jnp.arange(ticks))
        final,logs=jax.lax.scan(interval,packed,(reference[:cfg.commit_steps],acceleration[:cfg.commit_steps]))
        return final,jax.tree.map(lambda a:a.reshape((-1,*a.shape[2:])),logs)
    return execute
