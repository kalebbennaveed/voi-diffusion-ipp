"""Declared physical wind cases, separate from the controller's POAM belief.

These parameters are simulator inputs, not weather forecasts supplied to the
planner. Each mission must archive its resolved specification for independent
physical reintegration. The original case preserves PassageWind exactly.
"""

from copy import deepcopy
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .two_region import PassageWind

_BASE = {
    "family": "localized_passage_jet_v1",
    "amplitude": 1.8,
    "axial_fraction": 1.0,
    "cross_fraction": 0.2,
    "center": [0.0, 0.0],
    "longitudinal_width": 1.1,
    "transverse_width": 0.35,
}

WIND_CASES = {
    "original": _BASE,
    "reversed": _BASE | {"amplitude": -1.8},
    "crosswind": _BASE | {"amplitude": 1.6, "axial_fraction": 0.25, "cross_fraction": 1.0},
    "offroute": _BASE
    | {"center": [2.25, -1.6], "longitudinal_width": 0.45, "transverse_width": 0.18},
    "calm": _BASE | {"amplitude": 0.0},
}

CASE_LABELS = {
    "original": "Original A-to-B passage wind",
    "reversed": "Reversed B-to-A passage wind",
    "crosswind": "Crosswind across the middle passage",
    "offroute": "Localized wind away from the main route",
    "calm": "Calm physical field",
}


def wind_case_spec(case):
    if case not in WIND_CASES:
        raise ValueError(f"Unknown physical wind case: {case}")
    return deepcopy(WIND_CASES[case])


class LocalizedWind(NamedTuple):
    tangent: jax.Array
    normal: jax.Array
    center: jax.Array
    amplitude: jax.Array
    axial_fraction: jax.Array
    cross_fraction: jax.Array
    longitudinal_width: jax.Array
    transverse_width: jax.Array

    def at(self, positions):
        p = jnp.atleast_2d(positions) - self.center
        axial, lateral = p @ self.tangent, p @ self.normal
        envelope = jnp.exp(
            -0.5 * ((axial / self.longitudinal_width) ** 4 + (lateral / self.transverse_width) ** 2)
        )
        return (
            self.amplitude
            * envelope[:, None]
            * (self.axial_fraction * self.tangent + self.cross_fraction * self.normal)
        )


def physical_wind(env, spec):
    """Construct truth only. Never attach this object/spec to a wind belief."""
    if spec["family"] != "localized_passage_jet_v1":
        raise ValueError("Unsupported physical field family")
    if min(spec["longitudinal_width"], spec["transverse_width"]) <= 0:
        raise ValueError("Positive physical field widths required")
    tangent = (env.roi_b - env.roi_a) / jnp.linalg.norm(env.roi_b - env.roi_a)
    normal = jnp.array([-tangent[1], tangent[0]])
    if spec["center"] == [0.0, 0.0] and spec["axial_fraction"] == 1.0:
        return PassageWind(
            tangent,
            normal,
            jnp.array(spec["amplitude"]),
            jnp.array(spec["cross_fraction"]),
            jnp.array(spec["longitudinal_width"]),
            jnp.array(spec["transverse_width"]),
        )
    return LocalizedWind(
        tangent,
        normal,
        jnp.asarray(spec["center"]),
        jnp.array(spec["amplitude"]),
        jnp.array(spec["axial_fraction"]),
        jnp.array(spec["cross_fraction"]),
        jnp.array(spec["longitudinal_width"]),
        jnp.array(spec["transverse_width"]),
    )
