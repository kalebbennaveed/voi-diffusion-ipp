"""SO(3) geometric tracker with position and attitude feedback.

References include the heading and angular feedforward obtained from
flat-state conversion.

Source: Kaleb Ben Naveed's lab-experiment-sim-starter, commit c6d6200.
The array-backend argument lets JAX roll out this same controller, rather
than maintaining a second control law.
"""

import numpy as np

from .config import ControllerConfig, VehicleConfig
from .types import Reference, State


def hat(v: np.ndarray, xp=np) -> np.ndarray:
    x, y, z = v
    return xp.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(matrix: np.ndarray, xp=np) -> np.ndarray:
    return xp.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def desired_rotation(force: np.ndarray, yaw: float, heading: np.ndarray | None = None, *, xp=np) -> np.ndarray:
    magnitude = xp.linalg.norm(force)
    if xp is np and magnitude < 1e-9:
        raise ValueError("Desired force is zero; attitude is undefined")
    b3 = force / magnitude
    heading = xp.array([xp.cos(yaw), xp.sin(yaw), 0.0]) if heading is None else xp.asarray(heading, dtype=float)
    b2 = xp.cross(b3, heading)
    alternative = xp.cross(b3, xp.eye(3)[xp.argmin(xp.abs(b3))])
    b2 = xp.where(xp.sum(b2*b2) < 1e-18, alternative, b2)
    b2 = b2 / xp.linalg.norm(b2)
    return xp.column_stack((xp.cross(b2, b3), b2, b3))


def geometric_wrench(position, velocity, rotation, omega, reference, vehicle, gains, *, xp=np):
    """Kaleb's existing feedback/feedforward law, shared by NumPy and JAX."""
    e3 = xp.array([0.0, 0.0, 1.0])
    inertia = xp.diag(xp.asarray(vehicle.inertia))
    force = -gains.kx * (position - reference.position) - gains.kv * (velocity - reference.velocity) + vehicle.mass * (vehicle.gravity * e3 + reference.acceleration)
    if gains.max_horizontal_accel is not None:
        limit=vehicle.mass*gains.max_horizontal_accel
        drag=0. if gains.limit_total_horizontal_force else vehicle.linear_drag*xp.sqrt(xp.maximum(xp.sum(velocity**2),1e-24))*velocity[:2]
        force=xp.concatenate((xp.clip(force[:2]-drag,-limit,limit)+drag,force[2:]))
    rd = desired_rotation(force, reference.yaw, reference.heading, xp=xp)
    e_rotation = 0.5 * vee(rd.T @ rotation - rotation.T @ rd, xp)
    e_omega = omega - rotation.T @ rd @ reference.angular_velocity
    thrust = force @ rotation[:, 2]
    moment = -gains.kR * e_rotation - gains.kOmega * e_omega + xp.cross(omega, inertia @ omega) - inertia @ (hat(omega, xp) @ rotation.T @ rd @ reference.angular_velocity - rotation.T @ rd @ reference.angular_acceleration)
    return xp.concatenate((xp.reshape(thrust, (1,)), moment))


def allocation_matrix(vehicle: VehicleConfig) -> np.ndarray:
    """Motor order: (+x,-y), (-x,+y), (+x,+y), (-x,-y)."""
    kf, km = vehicle.thrust_coefficient, vehicle.torque_coefficient
    lx, ly = vehicle.arm_x, vehicle.arm_y
    return np.array([[kf, kf, kf, kf], [-ly*kf, ly*kf, ly*kf, -ly*kf], [-lx*kf, lx*kf, -lx*kf, lx*kf], [-km, -km, km, km]])


class GeometricController:
    def __init__(self, vehicle: VehicleConfig, gains: ControllerConfig):
        self.vehicle = vehicle
        self.gains = gains
        self.inertia = np.diag(vehicle.inertia)
        self.inverse_allocation = np.linalg.inv(allocation_matrix(vehicle))

    def wrench(self, state: State, reference: Reference) -> np.ndarray:
        return geometric_wrench(state.position, state.velocity, state.rotation,
                                state.angular_velocity, reference, self.vehicle, self.gains)

    def __call__(self, state: State, reference: Reference) -> np.ndarray:
        squared_speed = self.inverse_allocation @ self.wrench(state, reference)
        # Limit commands to forward rotation and the maximum motor speed.
        return np.sqrt(np.clip(squared_speed, 0.0, self.vehicle.max_motor_speed**2))
