"""Vehicle and tracker settings from lab-experiment-sim-starter (MIT)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleConfig:
    mass: float = 0.680
    inertia: tuple = (0.005, 0.005, 0.003)
    gravity: float = 9.81
    arm_x: float = 0.08
    arm_y: float = 0.08
    motor_time_constant: float = 0.02
    rotor_inertia: float = 6.62e-6
    thrust_coefficient: float = 1.91e-6
    torque_coefficient: float = 2.7e-7
    linear_drag: float = 0.1
    angular_drag: float = 0.003
    max_motor_speed: float = 2520.0


@dataclass(frozen=True)
class ControllerConfig:
    kx: float = 3.0
    kv: float = 5.0
    kR: float = 0.90
    kOmega: float = 0.120
    # None preserves the lab controller. Urban inspection applies the same
    # compensated horizontal command limits as its double-integrator planner.
    max_horizontal_accel: float | None = None
