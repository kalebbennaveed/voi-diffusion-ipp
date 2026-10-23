"""Execute an MBD trajectory using the lab's nonlinear quadrotor equations.

The planar outer control/filter is recomputed from the actual reached state
on a separate execution clock. Specific force is held between those samples. Flatness maps
that command to attitude on each smooth interval; the motor/attitude dynamics
resolve command jumps. No path interpolation changes the optimized decision.
"""
import jax
import jax.numpy as jnp
from dataclasses import replace
from ..quadrotor.config import VehicleConfig,ControllerConfig
from ..quadrotor.controller import allocation_matrix,geometric_wrench
from ..quadrotor.dynamics import rigid_body_derivative
from ..quadrotor.reference import flat_reference
from ..quadrotor.types import State
from ..wind.cache import TableMean
from ..wind.gp import InnerFields
from .environment import table_basis
from .continuous import nominal_reference,applied_control
from ..safety.flow_policy_value import filter_control
from ..quadrotor.aerodynamics import acceleration as drag


def executor(env,cfg,load,filter_on=True,dt=.01):
    vehicle=VehicleConfig();gains=ControllerConfig(max_horizontal_accel=cfg.umax,limit_total_horizontal_force=True)
    allocation=jnp.asarray(allocation_matrix(vehicle));inverse=jnp.linalg.inv(allocation)
    ticks=round(cfg.dt/dt)
    if abs(ticks*dt-cfg.dt)>1e-8:raise ValueError('Execution dt must divide planning dt')
    control_ticks=round(cfg.execution_dt/dt)
    execution_cfg=replace(cfg,safety_margin=cfg.execution_margin)
    @jax.jit
    def execute(packed,theta,mean_array,construction,reference_origin=None,start_index=0):
        x0=jnp.r_[packed[:2],packed[3:5]] if reference_origin is None else reference_origin
        reference,acceleration=nominal_reference(theta,x0,cfg)
        basis=table_basis(cfg.environment_scale,cfg.geometry)
        mean=TableMean(basis,mean_array);fields=InnerFields(basis,construction)
        def interval(packed,inputs):
            ref,a=inputs
            def tick(carry,i):
                packed,u,slack=carry
                state=State.unpack(packed);t=i*dt
                p=jnp.r_[ref[:2]+t*ref[2:]+.5*t*t*a,3.2]
                v=jnp.r_[ref[2:]+t*a,0.]
                def update(_):
                    reached=jnp.r_[packed[:2],packed[3:5]]
                    current_reference=jnp.r_[p[:2],v[:2]]
                    if filter_on and cfg.execution_filter=='hard':
                        nominal=a+cfg.tracking_kp*(p[:2]-reached[:2])+cfg.tracking_kv*(v[:2]-reached[2:])-drag(reached[2:],mean.mean_at(reached[:2])[0],xp=jnp)
                        command,info=filter_control(reached,nominal,mean,fields,env,execution_cfg)
                        return command,info.slack
                    return applied_control(reached,current_reference,a,mean,fields,env,execution_cfg,filter_on)
                u,slack=jax.lax.cond(i%control_ticks==0,update,lambda _:(u,slack),None)
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
                return (xn,u,slack),(xn,p,command,k1[3:6],u,slack)
            final,logs=jax.lax.scan(tick,(packed,jnp.zeros(2),jnp.array(0.)),jnp.arange(ticks))
            return final[0],logs
        refs=jax.lax.dynamic_slice(reference,(start_index,0),(cfg.commit_steps,4))
        aa=jax.lax.dynamic_slice(acceleration,(start_index,0),(cfg.commit_steps,2))
        final,logs=jax.lax.scan(interval,packed,(refs,aa))
        return final,jax.tree.map(lambda a:a.reshape((-1,*a.shape[2:])),logs)
    return execute
