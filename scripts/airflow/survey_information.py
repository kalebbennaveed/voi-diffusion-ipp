"""Route-targeted wind reduction for an embedded survey planning objective.

Only a finite, task-relevant set of target locations is scored. The objective
uses the current POAM inducing posterior and its full predictive covariance;
it does not replace the plant, wind sampler, or actual POAM updates. In
particular, the conditional GP residual remains in prediction but receives no
reward: a fixed-inducing POAM update cannot reduce that residual.

Two distinct calculations are exposed: fixed-inducing POAM reduction, and
full-predictive-GP conditioning reduction. The latter scores potential survey
measurements including residual information; it is not a claim that a sparse
POAM update achieves that reduction. Actual online updates may relocate their
inducing locations. Neither calculation forecasts that relocation or gives an
exact task value of information.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp


class Targets(NamedTuple):
    weighted_cross: jax.Array
    total_variance: jax.Array
    reducible_variance: jax.Array


class Reduction(NamedTuple):
    absolute: jax.Array
    fraction_total: jax.Array
    fraction_reducible: jax.Array


class FullTargets(NamedTuple):
    positions: jax.Array
    weights: jax.Array
    projection: jax.Array
    covariance_projection: jax.Array
    total_variance: jax.Array
    inducing_cholesky: jax.Array
    whitened_projection: jax.Array
    variance: jax.Array


class FullReduction(NamedTuple):
    absolute: jax.Array
    fraction_total: jax.Array


def prepare_targets(belief, positions, weights):
    """Prepare a weighted target covariance projection once per planning call.

    The caller supplies nonnegative task-relevance weights. Zero weights give
    exactly zero benefit for every possible measurement path. Weights need not
    sum to one; normalizing them gives a weighted average wind-variance change.
    Two statistically independent velocity components are included.
    """
    weights = jnp.maximum(jnp.asarray(weights), 0)
    cross = belief.basis.kernel_centers(positions)
    A = cross @ belief.inverse
    projection = A @ belief.covariance
    reducible = jnp.maximum(jnp.sum(projection * A, axis=1), 0)
    _, full_covariance = belief.predict(positions)
    variance = full_covariance[:, 0, 0]
    return Targets(
        jnp.sqrt(weights)[:, None] * projection,
        2 * jnp.sum(weights * variance),
        2 * jnp.sum(weights * reducible),
    )


def target_variance_reduction(belief, observation_positions, targets):
    """Weighted total predictive-variance reduction from prospective samples.

    Let S=q(U)'s covariance, A_X=K_XZ K_ZZ^-1 and R=sigma_n^2 I.
    The fixed-inducing POAM sufficient-statistic update is

      S+ = S - S A_X.T (R + A_X S A_X.T)^-1 A_X S.

    Target prediction is K_TT-A_T K_ZT+A_T S A_T.T. Its residual
    term cancels when taking the change, so the returned trace reduction uses
    the full model without pretending the residual is learned. Measurement
    values are unnecessary because this covariance update is deterministic.
    Existing POAM solves include their own numerical regularization; the
    algebra here uses the supplied current S and inverse exactly.
    """
    A = belief.basis.kernel_centers(observation_positions) @ belief.inverse
    covariance = A @ belief.covariance @ A.T
    covariance = (covariance + covariance.T) / 2
    covariance += belief.noise_std**2 * jnp.eye(len(observation_positions), dtype=A.dtype)
    root = jnp.linalg.cholesky(covariance)
    cross = targets.weighted_cross @ A.T
    solved = jsp.linalg.solve_triangular(root, cross.T, lower=True)
    absolute = 2 * jnp.sum(solved * solved)
    # Roundoff may place the ratio a few ulps outside [0, 1]. Preserve the
    # unmodified absolute quantity for audits and clamp only displayed ratios.
    fraction = lambda denominator: jnp.clip(absolute / jnp.maximum(denominator, 1e-30), 0, 1)
    return Reduction(absolute, fraction(targets.total_variance), fraction(targets.reducible_variance))


def prepare_full_targets(belief, positions, weights):
    """Prepare full predictive targets using solves, not a rounded inverse.

    Concentrated observation-supported inducing centers can make Kzz poorly
    conditioned. Multiplying a float32 copy of Kzz^-1 caused a negative
    observation covariance in the first survey pilot, despite positive
    definite float64 inference. Cholesky solves retain the same posterior
    equations and existing 1e-6 kernel regularization without clipping any
    residual covariance. The factor is computed once per target set.
    """
    weights = jnp.maximum(jnp.asarray(weights), 0)
    K = belief.old_K
    root = jnp.linalg.cholesky((K + K.T) / 2 + 1e-6 * jnp.eye(len(K), dtype=K.dtype))
    cross = belief.basis.kernel_centers(positions).T
    whitened = jsp.linalg.solve_triangular(root, cross, lower=True)
    projection = jsp.linalg.solve_triangular(root.T, whitened, lower=False).T
    covariance_projection = projection @ belief.covariance
    variance = (jnp.diag(belief.kernel(positions, positions))
                - jnp.sum(whitened * whitened, axis=0)
                + jnp.sum(covariance_projection * projection, axis=1))
    return FullTargets(positions, weights, projection,
                       covariance_projection, 2 * jnp.sum(weights * variance),
                       root, whitened, variance)


def full_target_variance_reduction_components(belief, observation_positions, targets):
    """Per-target weighted gain of the *current full predictive GP*.

    Delta Sigma_TT = Sigma_TX (Sigma_XX + sigma_n^2 I)^-1 Sigma_XT.

    This retains every current predictive covariance term, including the
    conditional residual. It is a survey objective over a finite target set,
    not a forecast of the reduction a relocated finite POAM model will attain.
    The achieved POAM change should be recorded separately. This quantity is
    not asserted to upper-bound that relocated model's change.
    """
    kernel_xz = belief.basis.kernel_centers(observation_positions)
    whitened = jsp.linalg.solve_triangular(targets.inducing_cholesky, kernel_xz.T, lower=True)
    A = jsp.linalg.solve_triangular(targets.inducing_cholesky.T, whitened, lower=False).T
    cross = belief.kernel(targets.positions, observation_positions)
    cross -= targets.whitened_projection.T @ whitened
    cross += targets.covariance_projection @ A.T
    covariance = (belief.kernel(observation_positions, observation_positions)
                  - whitened.T @ whitened + A @ belief.covariance @ A.T)
    covariance += belief.noise_std**2 * jnp.eye(len(observation_positions), dtype=A.dtype)
    root = jnp.linalg.cholesky((covariance + covariance.T) / 2)
    solved = jsp.linalg.solve_triangular(root, (jnp.sqrt(targets.weights)[:, None] * cross).T, lower=True)
    return 2 * jnp.sum(solved * solved, axis=0)


def full_target_variance_reduction(belief, observation_positions, targets):
    """Aggregate the stable full-predictive-GP target conditioning gain."""
    components = full_target_variance_reduction_components(belief, observation_positions, targets)
    absolute = jnp.sum(components)
    fraction = jnp.clip(absolute / jnp.maximum(targets.total_variance, 1e-30), 0, 1)
    return FullReduction(absolute, fraction)
