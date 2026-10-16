from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.dynamics import step
from voi_diffusion_ipp.experiments.scenarios import make_scenario
from voi_diffusion_ipp.planning.rollout import rollout
from voi_diffusion_ipp.safety import filter_control, values
from voi_diffusion_ipp.wind import WorldField


def test_posterior_changes_admissibility_without_truth():
    cfg = Config()
    env, b, x, _q = make_scenario(cfg)
    eps = jax.random.normal(jax.random.PRNGKey(73), (cfg.inner_samples, *b.mean.shape))
    post = b.condition(jnp.tile(jnp.array([[-0.65, 0.0]]), (12, 1)), jnp.zeros((12, 2)))
    x = jnp.array([-0.9, 0.0, 0.2, 0.0])
    u = jnp.array([0.65, 0.0])
    f = jax.jit(lambda b: filter_control(x, u, b, b.inner_fields(eps), env, cfg))
    _, old = f(b)
    _, new = f(post)
    assert new.correction < old.correction - 0.5
    assert new.admissible_fraction > old.admissible_fraction


def test_truth_does_not_change_first_action_but_changes_motion():
    cfg = replace(Config(), stage_steps=2, backup_steps=4, inner_samples=2)
    env, b, x, q = make_scenario(cfg)
    inner = b.inner_fields(jnp.ones((2, *b.mean.shape)))
    f = jax.jit(
        lambda c: rollout(
            x,
            q,
            jnp.ones((2, 2)) * 0.2,
            b,
            inner,
            WorldField(b.basis, c),
            jnp.zeros((2, 2)),
            env,
            cfg,
        )
    )
    a = f(jnp.zeros_like(b.mean))
    c = f(jnp.ones_like(b.mean) * 0.2)
    np.testing.assert_array_equal(a.controls[0], c.controls[0])
    assert not np.allclose(a.x[1], c.x[1])
    assert not np.allclose(a.observations, c.observations)


def test_wall_collision_prevention_and_fast_spline():
    cfg = replace(Config(), backup_steps=16, inner_samples=2, dt=0.05)
    env, b, _, _q = make_scenario(cfg)
    # Move circles away; test a wall with an analytically intelligible braking task.
    env = env._replace(
        obstacles=jnp.array([[50.0, 50.0, 0.1], [60.0, 60.0, 0.1]]),
        bounds=jnp.array([[-5.0, -5.0], [1.0, 5.0]]),
    )
    b = b._replace(mean=jnp.zeros_like(b.mean), covariance=jnp.eye(len(b.mean)) * 1e-9)
    inn = b.inner_fields(jnp.zeros((2, *b.mean.shape)))
    x = jnp.array([0.3, 0.0, 0.8, 0.0])
    fn = jax.jit(lambda x: filter_control(x, jnp.array([1.0, 0.0]), b, inn, env, cfg))
    unsafe = x
    for _ in range(30):
        u, info = fn(x)
        x = step(x, u, lambda p: jnp.zeros(2), cfg.dt)
        unsafe = step(unsafe, jnp.array([1.0, 0.0]), lambda p: jnp.zeros(2), cfg.dt)
        assert x[0] < 1.0 and info.feasible
    assert unsafe[0] > 1.5
    xf = jnp.array([0.2, 0.0, 0.5, 0.1])
    full = values(xf, b, inn, env, cfg)
    fast = values(xf, b, inn, env, replace(cfg, spline="fast"))
    np.testing.assert_allclose(full, fast, atol=2e-4)
