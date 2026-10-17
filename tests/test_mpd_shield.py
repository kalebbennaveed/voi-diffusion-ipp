from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.environment import Environment
from voi_diffusion_ipp.planning.budgeted import segment
from voi_diffusion_ipp.planning.mbd import score
from voi_diffusion_ipp.planning.safe_mpd import optimize
from voi_diffusion_ipp.safety.mpd_shield import filter_control, monitor
from voi_diffusion_ipp.wind import InnerFields, WorldField


class ConstantBasis(NamedTuple):
    placeholder: jax.Array

    def evaluate(self, p, coeff):
        return jnp.broadcast_to(coeff[0], jnp.atleast_2d(p).shape)


class ConstantBelief(NamedTuple):
    mean: jax.Array

    def mean_at(self, p):
        return jnp.broadcast_to(self.mean, jnp.atleast_2d(p).shape)


def fixture():
    cfg = replace(
        Config(),
        linear_drag=1.0,
        dt=0.2,
        backup_steps=8,
        backup_dt=0.15,
        safety_filter="sampled_mpd_shield",
    )
    env = Environment(
        jnp.array([[0.0, 0.0]]),
        jnp.array([0.7]),
        jnp.array([[0.0, 0.0, 0.5]]),
        jnp.array([[-4.0, -4.0], [4.0, 4.0]]),
        jnp.array([-2.0, 0.0]),
        jnp.array([2.0, 0.0]),
        jnp.zeros(2),
        jnp.array([0.16]),
    )
    basis = ConstantBasis(jnp.array(0.0))
    inner = InnerFields(basis, jnp.zeros((3, 1, 2)))
    belief = ConstantBelief(jnp.zeros(2))
    return cfg, env, inner, belief


def test_backup_rejects_a_clear_next_step_with_unsafe_stopping_path():
    cfg, env, inner, belief = fixture()
    x = jnp.array([-0.94, 0.0, 1.0, 0.0])
    ok, _, _ = jax.jit(lambda: monitor(x, jnp.zeros(2), belief, inner, env, cfg))()
    assert not ok
    # The prospective step is clear, but checking only that step would accept.
    assert -0.94 + (1 - np.exp(-0.2)) < -0.56
    safe, info = filter_control(x, jnp.zeros(2), belief, inner, env, cfg)
    assert info.active and safe[0] < 0


def test_safe_input_is_unchanged_and_outer_truth_is_rejected():
    cfg, env, inner, belief = fixture()
    x, u = jnp.array([-2.0, 0.0, 0.0, 0.0]), jnp.array([0.1, 0.1])
    safe, info = jax.jit(lambda: filter_control(x, u, belief, inner, env, cfg))()
    np.testing.assert_array_equal(safe, u)
    assert info.accepted and not info.active
    with pytest.raises(TypeError, match="InnerFields"):
        filter_control(x, u, belief, WorldField(inner.basis, inner.coefficients[0]), env, cfg)


def test_all_samples_must_pass_and_terminal_recovery_is_checked():
    cfg, env, inner, belief = fixture()
    x, u = jnp.array([-2.0, 0.0, 0.0, 0.0]), jnp.zeros(2)
    changed = inner._replace(coefficients=inner.coefficients.at[2, 0, 0].set(0.9))
    ok, violation, speed = monitor(x, u, belief, changed, env, cfg)
    assert violation < 0 and speed > cfg.shield_terminal_speed and not ok


def test_reference_reconsideration_is_explicit_and_default_remains_latched():
    cfg, env, inner, belief = fixture()
    x, u = jnp.array([-2.0, 0.0, 0.0, 0.0]), jnp.array([0.1, 0.1])
    _, paper = filter_control(x, u, belief, inner, env, cfg, active=True)
    assert paper.active and not paper.accepted
    control, reference = filter_control(
        x, u, belief, inner, env, replace(cfg, shield_latch=False), active=True
    )
    assert reference.accepted and not reference.active
    np.testing.assert_array_equal(control, u)


def test_physics_uses_applied_control_without_freezing_and_latch_persists():
    cfg, env, inner, belief = fixture()
    x = jnp.array([-0.94, 0.0, 1.0, 0.0])
    wind = WorldField(inner.basis, inner.coefficients[0])
    run = jax.jit(
        lambda: segment(
            x,
            jnp.array([0.7]),
            jnp.zeros((8, 2)),
            belief,
            inner,
            wind,
            jnp.zeros((8, 2)),
            env,
            cfg,
            False,
            actuator_controls=True,
        )
    )()
    assert run.shield_active.all()
    # Independently solve the constant-u damped double integrator for every interval.
    for i in range(8):
        p, v, u = np.array(run.x[i, :2]), np.array(run.x[i, 2:]), np.array(run.controls[i])
        vn = u + (v - u) * np.exp(-cfg.dt)
        pn = p + u * cfg.dt + (v - u) * (1 - np.exp(-cfg.dt))
        # Classical RK4 truncates exp(-dt) at order four. Bound its remainder
        # analytically rather than comparing a 0.2s step to machine precision.
        bound = np.max(np.abs(v - u)) * cfg.dt**5 / 120 + 2e-7
        np.testing.assert_allclose(run.x[i + 1], np.r_[pn, vn], atol=bound, rtol=0)
    assert not np.array_equal(run.x[0], run.x[1])
    next_run = segment(
        run.x[-1],
        run.q[-1],
        jnp.ones((2, 2)),
        belief,
        inner,
        wind,
        jnp.zeros((2, 2)),
        env,
        cfg,
        False,
        shield_latched=True,
    )
    assert next_run.shield_active.all()


def test_safe_mpd_score_uses_transformed_controls_and_final_output_is_transformed():
    cfg = replace(Config(), diffusion_steps=4, proposals=8, candidate_chunk=2)

    def transform(y):
        # An exact, idempotent test shield; direct raw averaging is wrong here.
        effective = jnp.clip(y, 0.0, 0.3)
        return jnp.sum((effective - 0.2) ** 2), effective

    r = jax.jit(lambda k: optimize(transform, jnp.ones(3), k, cfg))(jax.random.PRNGKey(12))
    h = r.history
    assert np.max(np.abs(h.candidates - h.effective_candidates)) > 0.1
    for i in range(cfg.diffusion_steps):
        expected, _ = score(
            h.noisy_before[i], h.effective_candidates[i], h.costs[i], h.alphabar[i], cfg.temperature
        )
        np.testing.assert_allclose(h.score[i], expected, atol=3e-5, rtol=1e-5)
    assert (r.vector >= 0).all() and (r.vector <= 0.3).all()
    np.testing.assert_allclose(r.cost, transform(r.vector)[0], atol=1e-6)
