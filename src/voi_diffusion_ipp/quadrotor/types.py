"""The interfaces between a user's policy, the tracker, and the vehicle."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


@dataclass
class State:
    position: Vector
    velocity: Vector
    rotation: Vector # body -> world, shape (3, 3)
    angular_velocity: Vector # body frame, rad/s
    motor_speed: Vector # rad/s, shape (4,)

    def pack(self) -> Vector:
        return np.concatenate((self.position, self.velocity, self.rotation.ravel(), self.angular_velocity, self.motor_speed))

    @classmethod
    def unpack(cls, x: Vector) -> "State":
        return cls(x[:3], x[3:6], x[6:15].reshape(3, 3), x[15:18], x[18:22])


@dataclass
class Reference:
    position: Vector
    velocity: Vector
    acceleration: Vector
    yaw: float = 0.0
    angular_velocity: Vector = field(default_factory=lambda: np.zeros(3))
    angular_acceleration: Vector = field(default_factory=lambda: np.zeros(3))
    heading: Vector | None = None # desired body x axis from flat-state conversion


class Policy(Protocol):
    def __call__(self, t: float, state: State) -> Reference:
        """Return the desired position, velocity, acceleration and heading."""
        ...


class Tracker(Protocol):
    def __call__(self, state: State, reference: Reference) -> Vector:
        """Return four desired motor speeds, in rad/s."""
        ...
