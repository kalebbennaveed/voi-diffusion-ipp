from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.environment import Environment
from voi_diffusion_ipp.planning.budgeted import evaluate, future_reference, reference_matrices
from voi_diffusion_ipp.wind.poam import POAM


def fixture():
    cfg = replace(
        Config(),
        linear_drag=1.0,
        stage_steps=2,
        future_steps=15,
        backup_steps=3,
        worlds=1,
        inner_samples=1,
        dt=0.2,
        reject_collisions=False,
    )
    belief = POAM(
        jnp.array([[-1.0, -0.5], [-1.0, 0.5], [1.0, -0.5], [1.0, 0.5]]),
        jax.random.PRNGKey(801),
        spectral_count=4,
    ).planning_belief()
    env = Environment(
        jnp.array([[-0.5, 0.0], [0.5, 0.0]]),
        jnp.full(2, 0.7),
        jnp.array([[0.0, 2.0, 0.2]]),
        jnp.array([[-4.0, -4.0], [4.0, 4.0]]),
        jnp.array([-0.5, 0.0]),
        jnp.array([0.5, 0.0]),
        jnp.zeros(2),
        jnp.full(2, 0.16),
    )
    return cfg, belief, env


def test_clamped_reference_uses_measured_initial_state_and_posterior_only():
    cfg, b, env = fixture()
    x = jnp.array([-0.5, 0.1, 0.2, -0.1])
    vector = jnp.zeros(18).at[12:].set(jnp.array([0.2, 0.1, 0.3, -0.1, 0.4, 0.2]))
    matrix = reference_matrices(9.0, 45)
    before = future_reference(vector, x, b, env, matrix, cfg)
    after = b.condition(jnp.array([[0.0, 0.0]]), jnp.array([[0.8, -0.5]]))
    changed = future_reference(vector, x, after, env, matrix, cfg)
    np.testing.assert_allclose(before[0, 0], x[:2], atol=1e-7)
    np.testing.assert_allclose(before[0, 1], x[2:], atol=1e-7)
    np.testing.assert_allclose(changed[0, :2], before[0, :2], atol=1e-7)
    assert float(jnp.linalg.norm(changed[1:, 0] - before[1:, 0])) > 0.01


def test_causal_update_and_future_noise_cannot_change_earlier_execution():
    cfg, b, env = fixture()
    eps = jax.random.normal(jax.random.PRNGKey(802), (2, *b.sample_shape))
    vector = jnp.zeros(18).at[12:].set(0.1)
    vector = vector.at[4:12].set(jnp.tile(jnp.array([jnp.arctanh(-0.5 / 3.88), 0.0]), 4))
    x = jnp.array([-0.5, 0.0, 0.0, 0.0])
    q = jnp.array([0.7, 0.7])
    noise = jnp.zeros((1, 17, 2))
    fn = jax.jit(lambda n: evaluate(vector, x, q, b, None, n, env, cfg, True, True, eps))
    actual = fn(noise)
    changed = fn(noise.at[:, 2:].set(10.0))
    # Future observations are recorded but never available within that frozen stage.
    np.testing.assert_array_equal(actual[1][0].controls, changed[1][0].controls)
    np.testing.assert_array_equal(actual[1][1].controls, changed[1][1].controls)
    passive = jax.jit(lambda: evaluate(vector, x, q, b, None, noise, env, cfg, False, True, eps))()
    np.testing.assert_array_equal(actual[1][0].x, passive[1][0].x)
    assert not np.allclose(actual[1][1].controls, passive[1][1].controls)
    assert np.isfinite(actual[0])


@pytest.mark.parametrize("features", ["probe", "route_knots"])
def test_multistage_updates_affect_only_later_future_controls(features):
    cfg, b, env = fixture()
    cfg = replace(cfg, budgeted_update_steps=5, hypothetical_relocation=True,
                  budgeted_feedback_features=features)
    eps = jax.random.normal(jax.random.PRNGKey(803), (2, *b.sample_shape))
    vector = jnp.zeros(18).at[12:].set(0.1)
    vector = vector.at[4:12].set(jnp.tile(jnp.array([jnp.arctanh(-0.5 / 3.88), 0.0]), 4))
    x, q = jnp.array([-0.5, 0.0, 0.0, 0.0]), jnp.array([0.7, 0.7])
    noise = jnp.zeros((1, 17, 2))
    fn = jax.jit(lambda n: evaluate(vector, x, q, b, None, n, env, cfg, True, True, eps))
    baseline = fn(noise)
    altered = fn(noise.at[:, 2:7].set(5.0))
    np.testing.assert_array_equal(baseline[1][0].controls, altered[1][0].controls)
    np.testing.assert_array_equal(baseline[1][1].controls[:, :5], altered[1][1].controls[:, :5])
    assert not np.allclose(baseline[1][1].controls[:, 5:], altered[1][1].controls[:, 5:])
    terminal = fn(noise.at[:, 12:].set(10.0))
    np.testing.assert_array_equal(baseline[1][1].controls, terminal[1][1].controls)


def test_route_features_respond_to_local_learning_without_a_probe_change():
    from typing import NamedTuple

    class SpatialBelief(NamedTuple):
        slope: float

        def predict(self, locations):
            p = jnp.atleast_2d(locations)
            mean = jnp.stack((self.slope * p[:, 0], jnp.zeros(len(p))), axis=-1)
            return mean, jnp.broadcast_to(jnp.eye(2) * 0.04, (len(p), 2, 2))

    cfg, _, env = fixture()
    vector = jnp.zeros(18).at[4:12].set(jnp.tile(jnp.array([0.3, 0.0]), 4)).at[12].set(0.4)
    x = jnp.array([-0.5, 0.1, 0.2, -0.1])
    matrix = reference_matrices(3.0, 15)
    before, after = SpatialBelief(0.0), SpatialBelief(0.5)
    probe_before = future_reference(vector, x, before, env, matrix, cfg)
    probe_after = future_reference(vector, x, after, env, matrix, cfg)
    np.testing.assert_array_equal(probe_before, probe_after)
    local = replace(cfg, budgeted_feedback_features="route_knots")
    route_before = future_reference(vector, x, before, env, matrix, local)
    route_after = future_reference(vector, x, after, env, matrix, local)
    assert float(jnp.max(jnp.linalg.norm(route_after[:, 0]-route_before[:, 0], axis=-1))) > 0.05
    np.testing.assert_allclose(route_after[0, 0], x[:2], atol=1e-7)
    np.testing.assert_allclose(route_after[0, 1], x[2:], atol=1e-7)
    np.testing.assert_array_equal(route_before, probe_before)


def test_route_feature_configuration_rejects_unsupported_combinations():
    with pytest.raises(ValueError, match="feedback features"):
        replace(Config(), budgeted_feedback_features="unknown")
    with pytest.raises(ValueError, match="compact policy"):
        replace(Config(), budgeted_feedback_features="route_knots", budgeted_policy="branches")


def test_omitting_zero_target_cells_preserves_full_grid_task_cost():
    cfg, b, env = fixture()
    # A background cell still has nonzero, decaying clarity, but zero target.
    env = env._replace(
        cells=jnp.concatenate((env.cells, jnp.array([[2.0, -1.0]]))),
        targets=jnp.array([0.7, 0.7, 0.0]),
        process_noise=jnp.array([0.16, 0.16, 0.3]),
    )
    compact = env._replace(
        cells=env.cells[:2], targets=env.targets[:2], process_noise=env.process_noise[:2]
    )
    x, q = jnp.array([-0.5, 0.0, 0.0, 0.0]), jnp.array([0.7, 0.4, 0.8])
    eps = jax.random.normal(jax.random.PRNGKey(804), (2, *b.sample_shape))
    noise = jnp.zeros((1, 17, 2))
    vector = jnp.zeros(18)
    full = jax.jit(lambda: evaluate(vector, x, q, b, None, noise, env, cfg, True, True, eps))()
    reduced = jax.jit(
        lambda: evaluate(vector, x, q[:2], b, None, noise, compact, cfg, True, True, eps)
    )()
    np.testing.assert_allclose(full[0], reduced[0] * 2 / 3, atol=1e-7, rtol=1e-6)
    for f, r in zip(full[1], reduced[1], strict=True):
        np.testing.assert_array_equal(f.x, r.x)
        np.testing.assert_array_equal(f.controls, r.controls)
        np.testing.assert_allclose(f.q[:, :, :2], r.q, atol=1e-7)


def test_two_routes_are_selected_from_observed_posterior_and_keep_the_terminal_state():
    from typing import NamedTuple

    class Posterior(NamedTuple):
        measured_mean: jax.Array

        def predict(self, locations):
            return self.measured_mean[None], jnp.eye(2)[None] * 0.04

    cfg, _, env = fixture()
    cfg = replace(cfg, budgeted_policy="branches")
    x = jnp.array([0.0, 0.1, 0.15, -0.08])
    # Separate continuations; the optimized gate uses measured x-wind only.
    y = jnp.r_[
        jnp.zeros(4),
        jnp.tile(jnp.array([-0.5, 0.0]), 4),
        jnp.tile(jnp.array([0.5, 0.0]), 4),
        jnp.array([0.0, 1.0, 0.0, 0.0]),
    ]
    matrix = reference_matrices(3.0, 15)
    left = future_reference(y, x, Posterior(jnp.array([-0.4, 0.0])), env, matrix, cfg)
    right = future_reference(y, x, Posterior(jnp.array([0.4, 0.0])), env, matrix, cfg)
    for ref in [left, right]:
        np.testing.assert_allclose(ref[0, 0], x[:2], atol=1e-7)
        np.testing.assert_allclose(ref[0, 1], x[2:], atol=1e-7)
    assert float(left[-1, 0, 0]) < -1.0
    assert float(right[-1, 0, 0]) > 1.0


def test_two_route_rollout_preserves_causal_observation_timing():
    cfg, b, env = fixture()
    cfg = replace(
        cfg, budgeted_policy="branches", budgeted_update_steps=5, hypothetical_relocation=True
    )
    eps = jax.random.normal(jax.random.PRNGKey(816), (2, *b.sample_shape))
    y = jnp.r_[
        jnp.zeros(4),
        jnp.tile(jnp.array([-0.12, -0.1]), 4),
        jnp.tile(jnp.array([0.12, 0.1]), 4),
        jnp.array([0.0, 1.0, 0.0, 0.0]),
    ]
    x, q = jnp.array([-0.5, 0.0, 0.0, 0.0]), jnp.array([0.7, 0.7])
    noise = jnp.zeros((1, 17, 2))
    fn = jax.jit(lambda n: evaluate(y, x, q, b, None, n, env, cfg, True, True, eps))
    active = fn(noise)
    changed = fn(noise.at[:, 2:7].set(5.0))
    np.testing.assert_array_equal(active[1][0].x, changed[1][0].x)
    np.testing.assert_array_equal(active[1][1].controls[:, :5], changed[1][1].controls[:, :5])
    assert not np.allclose(active[1][1].controls[:, 5:], changed[1][1].controls[:, 5:])
    frozen = jax.jit(lambda: evaluate(y, x, q, b, None, noise, env, cfg, False, True, eps))()
    np.testing.assert_array_equal(active[1][0].x, frozen[1][0].x)
    np.testing.assert_array_equal(active[1][0].q, frozen[1][0].q)
    assert np.isfinite(active[0])


def test_seven_two_route_starts_preserve_the_common_first_action_and_allow_staying():
    from voi_diffusion_ipp.planning.budgeted import split_vector, warm_starts

    cfg, _, env = fixture()
    env = env._replace(obstacles=jnp.array([[0.0, -0.72, 0.5], [0.0, 0.72, 0.5]]))
    x = jnp.array([-0.5, 0.0, 0.0, 0.0])
    compact = warm_starts(x, env, cfg, True)
    cfg = replace(cfg, budgeted_policy="branches")
    branched = warm_starts(x, env, cfg, True)
    assert compact.shape == (7, 18)
    assert branched.shape == (7, 24)
    assert np.isfinite(branched).all()
    np.testing.assert_array_equal(branched[-1, :4], compact[-1, :4])
    u, routes, _ = split_vector(branched[3], cfg)
    changed = branched[3].at[4:].add(0.7)
    np.testing.assert_array_equal(u, split_vector(changed, cfg)[0])
    assert routes.shape == (2, 4, 2)
    assert not np.array_equal(routes[0], routes[1])
