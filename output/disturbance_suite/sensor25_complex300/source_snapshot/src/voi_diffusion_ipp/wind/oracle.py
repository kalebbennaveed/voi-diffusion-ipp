"""Explicit privileged known-wind reference, never used as a learning backend."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .gp import InnerFields


class KnownBasis(NamedTuple):
    physical: object

    def evaluate(self, locations, coefficient):
        return self.physical.at(locations) * coefficient[0]


class KnownWind(NamedTuple):
    basis: KnownBasis

    @property
    def sample_shape(self):
        return (1, 2)

    def mean_at(self, locations):
        return self.basis.physical.at(locations)

    def predict(self, locations):
        mean = self.mean_at(locations)
        return mean, jnp.zeros((len(mean), 2, 2))

    def condition(self, positions, measured_wind, relocate=False):
        return self

    def inner_fields(self, eps):
        return InnerFields(self.basis, jnp.ones((len(eps), 1, 2)))


class KnownCache(NamedTuple):
    coefficients: jax.Array

    def relocate(self, belief, new_positions=None):
        return self

    def tabulate(self, belief):
        # No interpolation needed for this explicitly known analytic reference.
        return belief, InnerFields(belief.basis, self.coefficients)
