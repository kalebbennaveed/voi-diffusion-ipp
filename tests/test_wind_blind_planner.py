from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.environment import Environment
from voi_diffusion_ipp.planning.budgeted import segment
from voi_diffusion_ipp.planning.wind_blind import StillAir, evaluate_nominal
from voi_diffusion_ipp.wind.oracle import KnownBasis, KnownWind


def fixture():
    cfg = replace(
        Config(),
        dt=0.2,
        stage_steps=2,
        future_steps=8,
        linear_drag=1.0,
        backup_steps=6,
        backup_dt=0.15,
        worlds=1,
        inner_samples=1,
    )
    env = Environment(
        jnp.array([[-0.5, 0.0], [0.5, 0.0]]),
        jnp.full(2, 0.7),
        jnp.array([[0.0, 2.0, 0.2]]),
        jnp.array([[-2.0, -2.0], [2.0, 2.0]]),
        jnp.array([-0.5, 0.0]),
        jnp.array([0.5, 0.0]),
        jnp.zeros(2),
        jnp.full(2, 0.16),
    )
    return cfg, env


def test_nominal_forecast_has_no_wind_or_safety_dependency(monkeypatch):
    # The baseline must not secretly call the filter while choosing its path.
    def forbidden(*args, **kwargs):
        raise AssertionError("Wind-aware safety leaked into nominal planning")

    monkeypatch.setattr("voi_diffusion_ipp.planning.budgeted.filter_control", forbidden)
    cfg, env = fixture()
    vector = jnp.zeros(18).at[:4].set(jnp.arctanh(jnp.array([0.2, -0.1, 0.2, -0.1])))
    x = jnp.array([-0.5, 0.1, 0.1, 0.0])
    fn = jax.jit(lambda v: evaluate_nominal(v, x, jnp.full(2, 0.7), env, cfg))
    cost, (first, _) = fn(vector)
    # Independent exact constant-input damped dynamics for the first time step.
    u = np.array([0.2, -0.1]) * cfg.umax + np.asarray(x[2:])
    decay = np.exp(-cfg.dt)
    v = u + (np.asarray(x[2:]) - u) * decay
    p = np.asarray(x[:2]) + u * cfg.dt + (np.asarray(x[2:]) - u) * (1 - decay)
    np.testing.assert_allclose(first.x[0, 1], np.r_[p, v], atol=3e-6)
    assert np.isfinite(cost)
    altered = fn(vector.at[12:].set(20.0))
    np.testing.assert_array_equal(first.x, altered[1][0].x)
    np.testing.assert_array_equal(fn(vector)[1][1].x, altered[1][1].x)
    np.testing.assert_array_equal(first.correction, 0.0)


def test_actual_wind_blind_nominal_still_passes_through_full_rpcbf():
    cfg, env = fixture()
    # A moving state near the right boundary requires a real braking intervention.
    x = jnp.array([1.85, 0.0, 0.5, 0.0])
    still = StillAir()
    belief = KnownWind(KnownBasis(still))
    inner = belief.inner_fields(jnp.zeros((1, 1, 2)))
    references, noise = jnp.ones((2, 2)) * 0.5, jnp.zeros((2, 2))
    fn = lambda safety: jax.jit(
        lambda: segment(
            x,
            jnp.full(2, 0.7),
            references,
            belief,
            inner,
            still,
            noise,
            env,
            cfg,
            False,
            nominal_wind_aware=False,
            apply_safety=safety,
        )
    )()
    safe, raw = fn(True), fn(False)
    assert float(safe.correction.max()) > 0.1
    assert float(safe.x[-1, 0]) < float(raw.x[-1, 0])
    np.testing.assert_array_equal(raw.correction, 0.0)


def test_wind_compensation_is_absent_from_wind_blind_nominal_control():
    class Constant(NamedTuple):
        value: object

        def at(self, x):
            return jnp.broadcast_to(self.value, jnp.atleast_2d(x).shape)

    cfg, env = fixture()
    belief = KnownWind(KnownBasis(Constant(jnp.array([0.6, -0.4]))))
    x = jnp.zeros(4)
    args = (
        x,
        jnp.full(2, 0.7),
        jnp.zeros((2, 2)),
        belief,
        None,
        StillAir(),
        jnp.zeros((2, 2)),
        env,
        cfg,
        False,
    )
    blind = segment(*args, nominal_wind_aware=False, apply_safety=False)
    aware = segment(*args, nominal_wind_aware=True, apply_safety=False)
    np.testing.assert_array_equal(blind.controls[0], jnp.zeros(2))
    np.testing.assert_allclose(aware.controls[0], jnp.array([-0.6, 0.4]))
