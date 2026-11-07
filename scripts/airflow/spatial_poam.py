"""Resolved spatial POAM inference for the unrestricted mapping diagnostic.

No robot, RPCBF, truth-field or temporal model is accepted by this module.
This profile is intentionally separate from preserved robot experiments.
Planning-field sampling and total MBD runtime require separate validation.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from voi_diffusion_ipp.wind.poam import POAM, select_inducing, spatial_observations
from waypoint_poam import condition, train


@dataclass(frozen=True)
class SpatialPOAMSettings:
    observation_first_inducing: bool = False
    inducing_count: int = 768
    lengths_m: tuple = (.2, .5, 1., 2.)
    prior_std_m_s: float = 1.8
    observation_std_m_s: float = .1
    training_batch_size: int = 256
    amplitude_learning_rate: float = .01
    # The four attention-network Adam steps retain the original 0.001 rate.


class SpatialPOAM:
    """Ordinary POAM statistics and full posterior, with an explicit profile.

    `update` assimilates only the caller's measurements. Collecting a batch is
    the caller's responsibility; the diagnostic uses 50 measurements per call.
    Neither positions nor observations are normalized in physical units. Only
    the attention-network inputs are mapped to the geometry's [-1,1] rectangle.
    """
    def __init__(self, spatial_candidates, bounds, settings=SpatialPOAMSettings(), seed=731,
                 training_seed=442, prior_support=None, factorized_support=False):
        if not jax.config.x64_enabled:
            raise ValueError('Spatial POAM inference requires JAX x64; enable it before constructing the model')
        self.settings = settings
        candidates = jnp.asarray(spatial_candidates)
        if len(candidates) < settings.inducing_count:
            raise ValueError('Need at least inducing_count distinct spatial candidates')
        options = dict(lengths=settings.lengths_m, prior_std=settings.prior_std_m_s,
                       noise_std=settings.observation_std_m_s, attention_bounds=bounds,
                       prior_support=prior_support)
        key = jax.random.PRNGKey(seed)
        model = POAM(candidates[:settings.inducing_count], key, **options)
        centers = select_inducing(model.parameters, candidates, model.lengths, settings.inducing_count)
        model = POAM(centers, key, **options)
        self.belief = jax.tree.map(lambda a:a.astype(jnp.float64)
            if hasattr(a,'dtype') and jnp.issubdtype(a.dtype,jnp.floating) else a, model.planning_belief())
        if factorized_support:
            self.belief=self.belief._replace(basis=self.belief.basis._replace(support_inverse=jnp.zeros((0,0),dtype=jnp.float64)))
        self.mask = jnp.zeros(settings.inducing_count, bool)
        self.adam_m = jax.tree.map(jnp.zeros_like, self.belief.basis.parameters)
        self.adam_v = self.adam_m
        self.adam_step = jnp.array(0)
        self.positions = np.empty((0,2)); self.measurements = np.empty((0,2))
        self.rng = np.random.default_rng(training_seed)

    def update(self, positions, measurements, train_kernel=True):
        x,y = spatial_observations(positions,measurements)
        self.positions = np.concatenate((self.positions,np.asarray(x)))
        self.measurements = np.concatenate((self.measurements,np.asarray(y)))
        self.belief,self.mask = condition(self.belief,x,y,self.mask,observation_first=self.settings.observation_first_inducing)
        loss = None
        if train_kernel:
            # A fresh minibatch for each of four EM Adam steps, as in POAM.
            indices = self.rng.integers(0,len(self.positions),size=(4,self.settings.training_batch_size))
            self.belief,self.adam_m,self.adam_v,self.adam_step,loss = train(
                self.belief,jnp.asarray(self.positions[indices]),jnp.asarray(self.measurements[indices]),
                jnp.array(len(self.positions),dtype=jnp.float64),self.adam_m,self.adam_v,self.adam_step,
                self.settings.amplitude_learning_rate)
        return loss

    def predict(self, positions):
        return self.belief.predict(positions)
