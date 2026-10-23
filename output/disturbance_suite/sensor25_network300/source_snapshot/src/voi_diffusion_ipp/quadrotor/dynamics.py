"""22-state rigid body with motor lag, drag and rotor angular momentum.

RK4 holds motor commands constant during each controller step.
"""

import numpy as np

from .config import VehicleConfig
from .controller import allocation_matrix, hat
from .types import State


def rigid_body_derivative(packed, command, vehicle, allocation, wind, external_force, *, xp=np):
    """The existing 22-state plant, with an optional differentiable backend."""
    s, p = State.unpack(packed), vehicle
    inertia = xp.asarray(p.inertia)
    direction = xp.array([1.0, 1.0, -1.0, -1.0])
    e3 = xp.array([0.0, 0.0, 1.0])
    motor_dot = (xp.clip(command, 0, p.max_motor_speed) - s.motor_speed) / p.motor_time_constant
    thrust, mx, my, mz = allocation @ (s.motor_speed * xp.abs(s.motor_speed))
    air_velocity = s.velocity - wind
    # max avoids the undefined AD derivative of norm at zero. Below this
    # threshold the force is < 1e-24 N; ordinary regression values are unchanged.
    speed = xp.sqrt(xp.maximum(xp.sum(air_velocity**2), 1e-24))
    spin = xp.sqrt(xp.maximum(xp.sum(s.angular_velocity**2), 1e-24))
    force = s.rotation[:, 2] * thrust - p.linear_drag * speed * air_velocity + external_force - p.mass * p.gravity * e3
    moment = xp.array([mx, my, mz]) - p.rotor_inertia * (direction @ motor_dot) * e3
    moment = moment - p.angular_drag * spin * s.angular_velocity
    momentum = inertia * s.angular_velocity - p.rotor_inertia * (direction @ s.motor_speed) * e3
    omega_dot = (moment - xp.cross(s.angular_velocity, momentum)) / inertia
    return xp.concatenate((s.velocity, force / p.mass, (s.rotation @ hat(s.angular_velocity, xp)).ravel(), omega_dot, motor_dot))


class Quadrotor:
    def __init__(self, vehicle: VehicleConfig, wind: np.ndarray, external_force: np.ndarray):
        self.vehicle = vehicle
        self.inertia = np.asarray(vehicle.inertia)
        self.allocation = allocation_matrix(vehicle)
        self.motor_direction = np.array([1.0, 1.0, -1.0, -1.0])
        self.wind = np.asarray(wind, dtype=float)
        self.external_force = np.asarray(external_force, dtype=float)

    def derivative(self, packed: np.ndarray, command: np.ndarray) -> np.ndarray:
        return rigid_body_derivative(packed, command, self.vehicle, self.allocation,
                                     self.wind, self.external_force)

    def step(self, state: State, command: np.ndarray, dt: float) -> State:
        x = state.pack()
        k1 = self.derivative(x, command)
        k2 = self.derivative(x + dt * k1 / 2, command)
        k3 = self.derivative(x + dt * k2 / 2, command)
        k4 = self.derivative(x + dt * k3, command)
        next_state = State.unpack(x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6)
        # Keep R on SO(3) after the numerical integration.
        u, _, vt = np.linalg.svd(next_state.rotation)
        correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
        next_state.rotation = u @ correction @ vt
        return next_state
