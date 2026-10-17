"""Physical intervention checks against independent NumPy equations and geometry."""

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.experiments.two_region import make_two_region
from voi_diffusion_ipp.experiments.wind_cases import WIND_CASES, physical_wind, wind_case_spec

ROOT = Path(__file__).resolve().parents[1]
module = importlib.util.spec_from_file_location(
    "mission_audit", ROOT / "scripts/audit_budgeted_mission.py"
)
audit_module = importlib.util.module_from_spec(module)
module.loader.exec_module(audit_module)


@pytest.fixture(scope="module")
def setup():
    spec = yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text())
    cfg = load_config(ROOT / "configs/two_region_budgeted.yaml")
    return spec, make_two_region(spec, cfg)


def test_original_unchanged_and_reversal_is_exact_at_arbitrary_points(setup):
    _, (env, _, _, _, original) = setup
    points = jnp.asarray(np.random.default_rng(22).uniform([-2.7, -2.1], [2.7, 2.1], (401, 2)))
    forward = physical_wind(env, wind_case_spec("original")).at(points)
    reverse = physical_wind(env, wind_case_spec("reversed")).at(points)
    np.testing.assert_array_equal(forward, original.at(points))
    np.testing.assert_array_equal(reverse, -forward)


@pytest.mark.parametrize("case", list(WIND_CASES))
def test_independent_physical_reintegration_equation_matches_every_case(case, setup):
    spec, (env, _, _, _, _) = setup
    points = np.random.default_rng(42).uniform([-2.7, -2.1], [2.7, 2.1], (501, 2))
    physical_spec = wind_case_spec(case)
    production = physical_wind(env, physical_spec).at(jnp.asarray(points))
    independent = audit_module.physical_wind_numpy(points, spec, physical_spec)
    np.testing.assert_allclose(production, independent, atol=2e-6, rtol=3e-6)


def test_offroute_field_is_nonzero_but_not_on_the_direct_transfer(setup):
    _, (env, _, _, _, _) = setup
    spec = wind_case_spec("offroute")
    field = physical_wind(env, spec)
    fraction = jnp.linspace(0, 1, 401)[:, None]
    route = env.roi_a + fraction * (env.roi_b - env.roi_a)
    assert float(jnp.linalg.norm(field.at(route), axis=1).max()) < 1e-6
    assert float(jnp.linalg.norm(field.at(jnp.array(spec["center"])))) > 1.8


def test_truth_case_is_not_given_to_estimator_and_registry_is_not_mutated(setup):
    _, (env, model, _, _, _) = setup
    mean, cov = model.predict(env.cells[:30])
    for case in WIND_CASES:
        spec = wind_case_spec(case)
        physical_wind(env, spec).at(env.cells)
        spec["center"][0] = 100
        assert wind_case_spec(case)["center"][0] != 100
    after_mean, after_cov = model.predict(env.cells[:30])
    np.testing.assert_array_equal(after_mean, mean)
    np.testing.assert_array_equal(after_cov, cov)


def test_additional_historical_main_route_survey_is_common_and_geometric(setup):
    """The low-value regime adds noisy-data locations, never hidden mean values."""
    import importlib.util

    module_spec = importlib.util.spec_from_file_location(
        "mission_survey_test", ROOT / "scripts/run_budgeted_mission.py"
    )
    driver = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(driver)
    _, (env, *_) = setup
    original = driver.survey_positions(env, "outer-survey")
    extended = driver.survey_positions(env, "routes-survey")
    assert original.shape == (42, 2)
    assert extended.shape == (63, 2)
    np.testing.assert_array_equal(extended[:42], original)
    np.testing.assert_allclose(extended[42], env.roi_a)
    np.testing.assert_allclose(extended[-1], env.roi_b)
    positions = np.asarray(extended)
    obstacles = np.asarray(env.obstacles)
    gaps = (
        np.linalg.norm(positions[:, None] - obstacles[None, :, :2], axis=-1) - obstacles[None, :, 2]
    )
    assert gaps.min() > 0.06
