"""C2 planar references and differential flatness with jerk/snap feedforward.

Extends the lab starter's zero-jerk/snap mapping. Position/velocity/acceleration
are one consistent quintic polynomial; no rendered positions are fabricated.
"""
import numpy as np

from .controller import hat, vee
from .types import Reference


def normalized_derivatives(v, dv, ddv, xp=np):
    n = xp.linalg.norm(v)
    if xp is np and n < 1e-8:
        raise ValueError("Flatness direction is singular")
    b = v / n
    dn = b @ dv
    db = (dv - b * dn) / n
    ddn = db @ dv + b @ ddv
    ddb = (ddv - b * ddn - 2 * db * dn) / n
    return b, db, ddb


def flat_reference(p, v, a, jerk=None, snap=None, yaw=0., yaw_rate=0., yaw_accel=0.,
                   compensation=None, gravity=9.81, xp=np):
    """Known drag and earned wind compensation are held in this conversion.

    Compensation shifts required thrust, not the kinematic reference. Its
    derivatives are treated as zero within a control sample (feedback term).
    """
    jerk = xp.zeros(3) if jerk is None else xp.asarray(jerk)
    snap = xp.zeros(3) if snap is None else xp.asarray(snap)
    compensation = xp.zeros(3) if compensation is None else xp.asarray(compensation)
    force = xp.asarray(a) + compensation + xp.array([0., 0., gravity])
    b3, db3, ddb3 = normalized_derivatives(force, jerk, snap, xp)
    h = xp.array([xp.cos(yaw), xp.sin(yaw), 0.])
    dh = yaw_rate * xp.array([-xp.sin(yaw), xp.cos(yaw), 0.])
    ddh = yaw_accel * xp.array([-xp.sin(yaw), xp.cos(yaw), 0.]) - yaw_rate**2*h
    cross = xp.cross(b3, h)
    dcross = xp.cross(db3, h) + xp.cross(b3, dh)
    ddcross = xp.cross(ddb3, h) + 2*xp.cross(db3, dh) + xp.cross(b3, ddh)
    b2, db2, ddb2 = normalized_derivatives(cross, dcross, ddcross, xp)
    b1 = xp.cross(b2, b3)
    db1 = xp.cross(db2, b3) + xp.cross(b2, db3)
    ddb1 = xp.cross(ddb2, b3) + 2*xp.cross(db2, db3) + xp.cross(b2, ddb3)
    R = xp.column_stack((b1, b2, b3))
    dR = xp.column_stack((db1, db2, db3))
    ddR = xp.column_stack((ddb1, ddb2, ddb3))
    omega = vee(R.T @ dR, xp)
    alpha = vee(R.T @ ddR - hat(omega, xp) @ hat(omega, xp), xp)
    return Reference(xp.asarray(p), xp.asarray(v), xp.asarray(a)+compensation,
                     yaw, omega, alpha, h)


class QuinticReference:
    def __init__(self, p0, v0, a0, p1, v1, a1, duration):
        self.duration = float(duration)
        T = self.duration
        self.coefficients = np.zeros((6, 3))
        self.coefficients[:3] = [p0, v0, np.asarray(a0)/2]
        M = np.array([[T**3, T**4, T**5], [3*T**2,4*T**3,5*T**4],
                      [6*T,12*T**2,20*T**3]])
        rhs = np.array([np.asarray(p1)-p0-T*np.asarray(v0)-T*T*np.asarray(a0)/2,
                        np.asarray(v1)-v0-T*np.asarray(a0), np.asarray(a1)-a0])
        self.coefficients[3:] = np.linalg.solve(M, rhs)

    def at(self, t):
        t = np.clip(t, 0., self.duration)
        c = self.coefficients.copy()
        values = []
        for _ in range(5):
            values.append(np.sum(c * t**np.arange(len(c))[:, None], axis=0))
            c = c[1:] * np.arange(1, len(c))[:, None]
        return values
