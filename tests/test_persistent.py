from dataclasses import replace
from itertools import pairwise

import jax
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from voi_diffusion_ipp.clarity import step
from voi_diffusion_ipp.config import Config, load_config
from voi_diffusion_ipp.experiments.checkpoint import PlanCache
from voi_diffusion_ipp.experiments.run import mission
from voi_diffusion_ipp.experiments.scenarios import make_scenario


@pytest.mark.parametrize("name", ["archipelago", "channels", "mosaic"])
def test_persistent_scenario_has_distributed_demand_and_spatial_process_matrix(name):
    cfg = load_config(f"configs/persistent_{name}.yaml")
    env, _, x, q = make_scenario(cfg)
    assert cfg.mission_horizons * cfg.stage_steps * cfg.dt >= 120
    assert len(env.cells) == 475 and len(env.obstacles) >= 5
    assert np.all(np.asarray(env.targets) > 0)
    assert np.all(np.asarray(q) > np.asarray(env.targets))
    process = np.asarray(env.process_noise)
    assert process.shape == (475,) and np.all(process > 0)
    if name != "archipelago":
        assert process.max() / process.min() > 2
    # Without sensing, every cell loses its target within the mission.
    final = np.asarray(q) / (
        1 + process * np.asarray(q) * cfg.mission_horizons * cfg.stage_steps * cfg.dt
    )
    assert np.all(final < np.asarray(env.targets))
    assert float(env.clearance(x)) > cfg.safety_margin
    from voi_diffusion_ipp.viz.demo_server import validate_request

    live_cfg, _, _ = validate_request({"scenario": f"persistent_{name}"})
    assert live_cfg.mission_horizons * live_cfg.stage_steps * live_cfg.dt >= 120


def test_patchy_Q_clarity_matches_independent_covariance_integration():
    cfg = load_config("configs/persistent_mosaic.yaml")
    env, _, x, q0 = make_scenario(cfg)
    cells, process = np.asarray(env.cells, dtype=float), np.asarray(env.process_noise, dtype=float)
    start, end = np.asarray(x[:2]), np.array([1.4, 0.7])
    times = np.arange(41) * cfg.dt

    def rhs(t, P):
        p = start + (end - start) * t / times[-1]
        c2 = np.exp(-np.sum((cells - p) ** 2, axis=1) / cfg.sensor_sigma**2)
        return process - c2 * P**2 / cfg.sensor_R

    ref = solve_ivp(
        rhs, (0, times[-1]), 1 / np.asarray(q0) - 1, t_eval=times, rtol=1e-10, atol=1e-11
    )
    advance = jax.jit(
        lambda q, a, b: step(
            q, a, b, env.cells, cfg.sensor_sigma, cfg.sensor_R, env.process_noise, cfg.dt
        )
    )
    q = q0
    history = [q]
    for a, b in pairwise(times):
        q = advance(q, start + (end - start) * a / times[-1], start + (end - start) * b / times[-1])
        history.append(q)
    np.testing.assert_allclose(history, 1 / (1 + ref.y.T), atol=1e-4)


def test_numeric_plan_checkpoint_recovers_identical_execution(tmp_path):
    cfg = replace(
        Config(),
        scenario="localized_corridor",
        control_representation="waypoint",
        branch_rule="task",
        stage_steps=3,
        future_steps=3,
        backup_steps=3,
        worlds=1,
        inner_samples=1,
        recourse_samples=1,
        knots=2,
        diffusion_steps=2,
        proposals=2,
        candidate_chunk=1,
        mission_horizons=2,
    )

    class Interrupted(Exception):
        pass

    def interrupt(event, payload):
        if event == "executed" and payload["horizon"] == 0:
            raise Interrupted()

    with pytest.raises(Interrupted):
        mission(
            cfg,
            "proposed",
            41,
            tmp_path / "resume",
            progress=False,
            observer=interrupt,
            checkpoint=True,
        )
    recovered = mission(cfg, "proposed", 41, tmp_path / "resume", progress=False, checkpoint=True)
    direct = mission(cfg, "proposed", 41, tmp_path / "direct", progress=False)
    assert recovered["resumed_plan_count"] == 1
    assert recovered["integrated_deficit"] == direct["integrated_deficit"]
    stem = "localized_corridor_proposed_41.npz"
    with np.load(tmp_path / "resume" / stem) as a, np.load(tmp_path / "direct" / stem) as b:
        for key in ["x", "q", "control", "observations", "diffusion_candidates"]:
            if key in a:
                np.testing.assert_array_equal(a[key], b[key])
    with pytest.raises(ValueError, match="configuration"):
        PlanCache(
            tmp_path / "resume" / "localized_corridor_proposed_41.checkpoints",
            replace(cfg, dt=0.2),
            "proposed",
            41,
        )


def test_persistent_revisit_and_recovery_counts_are_not_dwell_counts():
    from voi_diffusion_ipp.experiments.persistence import summarize

    x = np.array([[0, 0], [0, 0], [0, 0], [5, 0], [5, 0], [5, 0], [0, 0], [0, 0]])
    q = np.array([[0.7], [0.6], [0.4], [0.4], [0.4], [0.4], [0.6], [0.7]])
    result = summarize(x, q, [[0, 0]], [0.5], [0.035], 0.5, 1.0)
    assert result["sensing_pass_counts"] == [2]
    assert result["clarity_recovery_counts"] == [1]
    assert result["median_revisit_seconds"] == 6
    stationary = summarize(np.zeros_like(x), q, [[0, 0]], [0.5], [0.035], 0.5, 1.0)
    assert stationary["sensing_pass_counts"] == [1]


def test_full_poam_demo_recovers_partial_recording(tmp_path):
    import json

    from voi_diffusion_ipp.viz.demo_data import export_demo

    cfg = replace(
        load_config("configs/persistent_mosaic.yaml"),
        stage_steps=3,
        future_steps=3,
        mission_horizons=2,
        backup_steps=3,
        worlds=1,
        inner_samples=1,
        recourse_samples=1,
        knots=2,
        diffusion_steps=2,
        proposals=2,
        warm_starts=1,
        grid_nx=5,
        grid_ny=5,
        poam_inducing_x=2,
        poam_inducing_y=2,
        poam_train_steps=1,
    )

    class Interrupted(Exception):
        pass

    def stop(message):
        if "Period 1/2: executed" in message:
            raise Interrupted()

    with pytest.raises(Interrupted):
        export_demo(cfg, "proposed", 42, tmp_path, status=stop)
    partial = json.loads(next(tmp_path.glob("*.partial.json")).read_text())
    assert len(partial["execution"]) == 1
    messages = []
    path = export_demo(cfg, "proposed", 42, tmp_path, status=messages.append)
    data = json.loads(path.read_text())
    assert data["metrics"]["resumed_plan_count"] == 1
    assert len(data["planning"]) == 2 and len(data["beliefs"]) == 3
    assert any("recovered recorded plan" in s for s in messages)
    np.testing.assert_array_equal(data["mission"]["x"][:4], partial["execution"][0]["x"])
    np.testing.assert_array_equal(data["mission"]["q"][:4], partial["execution"][0]["q"])
    assert data["environment"]["process_noise"] == partial["environment"]["process_noise"]
