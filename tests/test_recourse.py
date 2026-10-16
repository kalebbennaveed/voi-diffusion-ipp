import inspect
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.scenarios import make_scenario
from voi_diffusion_ipp.planning.mpc import initial_policy
from voi_diffusion_ipp.planning.rollout import waypoint_controller
from voi_diffusion_ipp.planning.two_stage import evaluate_one, make_buffers, recourse_costs, route


def test_task_recourse_responds_to_monitoring_need_without_hidden_truth():
    cfg = replace(
        Config(),
        branch_rule="task",
        control_representation="waypoint",
        future_steps=15,
        dt=0.1,
        backup_steps=3,
        inner_samples=1,
        recourse_samples=2,
    )
    env, belief, _, _ = make_scenario(cfg)
    env = env._replace(
        cells=jnp.array([[0.65, 0.0], [-0.65, 0.0]]),
        targets=jnp.full(2, 0.8),
        obstacles=env.obstacles.at[:, :2].set(20.0),
    )
    belief = belief._replace(covariance=belief.covariance * 1e-8)
    futures = jnp.tile(env.cells[:, None], (1, cfg.future_steps, 1))
    select = jax.jit(lambda q: route(jnp.zeros(4), q, belief, env, cfg, futures))
    assert int(select(jnp.array([0.05, 0.95]))) == 0
    assert int(select(jnp.array([0.95, 0.05]))) == 1
    assert "world" not in inspect.signature(recourse_costs).parameters
    assert "coeff" not in inspect.signature(recourse_costs).parameters


def test_waypoint_policy_uses_observed_state_and_available_belief():
    cfg = Config()
    _, belief, _, _ = make_scenario(cfg)
    x = jnp.array([0.0, 0.0, 0.1, -0.1])
    target = jnp.array([0.5, 0.0])
    old = waypoint_controller(x, target, belief, cfg)
    learned = belief.condition(jnp.zeros((8, 2)), jnp.tile(jnp.array([[0.3, -0.2]]), (8, 1)))
    new = waypoint_controller(x, target, learned, cfg)
    np.testing.assert_allclose(old - new, learned.predict(x[:2])[0][0], atol=1e-6)


def test_localized_field_makes_probe_measurements_decision_relevant():
    cfg = replace(Config(), scenario="localized_corridor", wind_length=0.55, wind_std=0.8)
    _, belief, _, _ = make_scenario(cfg)
    positions = jnp.array([[0.0, 0.0], [-1.45, -0.6], [-0.65, 0.0]])
    phi = np.asarray(belief.basis.features(positions))
    _, joint = belief.joint(positions)
    np.testing.assert_allclose(joint, 0.8**2 * phi @ phi.T, atol=1e-7)
    far = belief.condition(jnp.tile(positions[1], (12, 1)), jnp.zeros((12, 2)))
    near = belief.condition(jnp.tile(positions[2], (12, 1)), jnp.zeros((12, 2)))
    assert near.predict(positions[:1])[1][0, 0, 0] < 0.1 * far.predict(positions[:1])[1][0, 0, 0]


def test_no_measurements_gives_no_artificial_recourse_value():
    cfg = replace(
        Config(),
        control_representation="waypoint",
        branch_rule="task",
        measurements=False,
        stage_steps=3,
        future_steps=3,
        backup_steps=3,
        inner_samples=1,
        recourse_samples=1,
        worlds=1,
    )
    env, belief, x, q = make_scenario(cfg)
    buffers = make_buffers(belief, jax.random.PRNGKey(9), cfg)
    vector = initial_policy(x, q, belief, env, cfg)

    def evaluate(mode):
        return jax.jit(
            lambda: evaluate_one(
                vector,
                x,
                q,
                belief,
                buffers.world_coefficients[0],
                buffers.observation_noise[0],
                buffers.inner_prior_eps,
                buffers.inner_future_eps,
                env,
                cfg,
                mode,
            )
        )()

    informed = evaluate("proposed")
    frozen = evaluate("current_belief")
    np.testing.assert_array_equal(informed.future.x, frozen.future.x)
    np.testing.assert_array_equal(informed.future.q, frozen.future.q)
    np.testing.assert_array_equal(informed.cost, frozen.cost)
