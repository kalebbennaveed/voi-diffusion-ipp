"""Known quadratic air drag; uncertainty is in spatial air velocity, not mass."""
import numpy as np

MASS = .680
DRAG = .1


def acceleration(velocity, wind, *, xp=np):
    relative = wind - velocity
    speed = xp.sqrt(xp.maximum(xp.sum(relative**2, axis=-1, keepdims=True), 1e-20))
    return (DRAG / MASS) * speed * relative


def infer_wind(velocity, acceleration_residual, *, xp=np):
    """Invert the known drag law on the measured external acceleration."""
    magnitude = xp.sqrt(xp.maximum(xp.sum(acceleration_residual**2, axis=-1, keepdims=True), 1e-20))
    relative = acceleration_residual / xp.sqrt((DRAG / MASS) * magnitude)
    return velocity + relative
