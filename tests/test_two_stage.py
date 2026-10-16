import inspect
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.scenarios import make_scenario
from voi_diffusion_ipp.planning.two_stage import (
    evaluate_one,
    make_buffers,
    matched_no_information,
    route,
)
from voi_diffusion_ipp.safety.rpcbf import values
from voi_diffusion_ipp.wind import WorldField


def test_sequential_states_posterior_and_nonanticipativity():
    cfg = replace(
        Config(), stage_steps=3, future_steps=3, backup_steps=4, inner_samples=2, worlds=2
    )
    env, b, x, q = make_scenario(cfg)
    buf = make_buffers(b, jax.random.PRNGKey(4), cfg)
    y = jnp.zeros((1 + cfg.branches) * cfg.knots * 2)
    f = jax.jit(
        lambda yy: evaluate_one(
            yy,
            x,
            q,
            b,
            buf.world_coefficients[0],
            buf.observation_noise[0],
            buf.inner_prior_eps,
            buf.inner_future_eps,
            env,
            cfg,
        )
    )
    r = f(y)
    np.testing.assert_array_equal(r.future.x[0], r.first.x[-1])
    np.testing.assert_array_equal(r.future.q[0], r.first.q[-1])
    post = b.condition(r.first.x[1:, :2], r.first.observations)
    np.testing.assert_allclose(r.posterior.mean, post.mean, atol=1e-6)
    assert np.linalg.norm(r.posterior.mean) > 0
    np.testing.assert_allclose(r.cost, r.first.cost + r.future.cost, atol=1e-6)
    no = jax.jit(
        lambda: matched_no_information(
            r, y, b, buf.world_coefficients[0], buf.inner_future_eps, env, cfg
        )
    )()
    np.testing.assert_array_equal(no.x[0], r.future.x[0])
    np.testing.assert_array_equal(no.q[0], r.future.q[0])
    assert "world" not in inspect.signature(route).parameters
    assert route(x, q, b, env, cfg) == route(x, q, b, env, cfg)
    # Changing only an unused branch must not alter any trajectory in this scenario.
    inactive = 1 - int(r.branch)
    altered = y.reshape((3, cfg.knots, 2)).at[inactive + 1].set(2.0).ravel()
    rr = f(altered)
    np.testing.assert_allclose(rr.future.x, r.future.x, atol=1e-6)
    with pytest.raises(TypeError):
        values(x, b, WorldField(b.basis, buf.world_coefficients[0]), env, cfg)
