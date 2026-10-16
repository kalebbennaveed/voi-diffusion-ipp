from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.scenarios import make_scenario
from voi_diffusion_ipp.planning.mpc import initial_policy
from voi_diffusion_ipp.planning.rollout import continuation, rollout
from voi_diffusion_ipp.planning.two_stage import evaluate_one, make_buffers
from voi_diffusion_ipp.wind import WorldField


def test_multiperiod_continuation_matches_manual_causal_updates():
    cfg = replace(
        Config(),
        scenario="localized_corridor",
        control_representation="waypoint",
        stage_steps=3,
        future_steps=6,
        backup_steps=3,
        inner_samples=1,
        future_updates=True,
    )
    env, b, x, q = make_scenario(cfg)
    world = WorldField(b.basis, jnp.array([[0.7, -0.4]]))
    u = jnp.tile(jnp.array([[-0.4, 0.0]]), (6, 1))
    eps = jnp.ones((1, *b.mean.shape)) * 0.5
    noise = jax.random.normal(jax.random.PRNGKey(28), (6, 2))
    actual = jax.jit(lambda: continuation(x, q, u, b, eps, world, noise, env, cfg, True))()
    first = rollout(x, q, u[:3], b, b.inner_fields(eps), world, noise[:3], env, cfg)
    posterior = b.condition(first.x[1:, :2], first.observations)
    second = rollout(
        first.x[-1],
        first.q[-1],
        u[3:],
        posterior,
        posterior.inner_fields(eps),
        world,
        noise[3:],
        env,
        cfg,
    )
    np.testing.assert_allclose(actual.x, jnp.concatenate((first.x, second.x[1:])), atol=2e-6)
    np.testing.assert_allclose(actual.q, jnp.concatenate((first.q, second.q[1:])), atol=2e-6)
    np.testing.assert_allclose(actual.cost, first.cost + second.cost, atol=1e-6)
    frozen = continuation(x, q, u, b, eps, world, noise, env, cfg, False)
    np.testing.assert_allclose(actual.x[:4], frozen.x[:4], atol=1e-6)
    assert not np.allclose(actual.controls[3:], frozen.controls[3:])
    noobs = replace(cfg, measurements=False)
    noobs_out = continuation(x, q, u, b, eps, world, noise, env, noobs, True)
    np.testing.assert_array_equal(noobs_out.x, frozen.x)


def test_future_observations_cannot_change_earlier_controls():
    cfg = replace(
        Config(),
        scenario="localized_corridor",
        control_representation="waypoint",
        stage_steps=3,
        future_steps=6,
        backup_steps=3,
        inner_samples=1,
        future_updates=True,
    )
    env, b, x, q = make_scenario(cfg)
    world = WorldField(b.basis, jnp.array([[0.7, -0.4]]))
    u = jnp.tile(jnp.array([[-0.4, 0.0]]), (6, 1))
    eps = jnp.zeros((1, *b.mean.shape))
    run = jax.jit(lambda noise: continuation(x, q, u, b, eps, world, noise, env, cfg, True))
    baseline = run(jnp.zeros((6, 2)))
    changed = run(jnp.ones((6, 2)) * 10)
    np.testing.assert_array_equal(baseline.controls[:3], changed.controls[:3])
    assert not np.allclose(baseline.controls[3:], changed.controls[3:])


def test_optional_qp_feasibility_density_rejects_unrecoverable_control_states():
    cfg = replace(
        Config(),
        scenario="localized_corridor",
        stage_steps=3,
        future_steps=6,
        backup_steps=3,
        inner_samples=1,
        worlds=1,
        umax=0.05,
        reject_collisions=False,
        wind_std=0.01,
    )
    env, b, _, q = make_scenario(cfg)
    x = jnp.array([2.2, 0.0, 1.0, 0.0])
    y = initial_policy(x, q, b, env, cfg)
    buf = make_buffers(b, jax.random.PRNGKey(7), cfg)

    def evaluate(c):
        return jax.jit(
            lambda: evaluate_one(
                y,
                x,
                q,
                b,
                buf.world_coefficients[0],
                buf.observation_noise[0],
                buf.inner_prior_eps,
                buf.inner_future_eps,
                env,
                c,
                "current_belief",
            )
        )()

    unconstrained = evaluate(cfg)
    strict = evaluate(replace(cfg, require_feasible_candidates=True))
    assert not np.all(unconstrained.first.feasible)
    assert np.isfinite(unconstrained.cost) and np.isinf(strict.cost)
