"""Mission-time optimism is explicit and does not alter ordinary criteria."""
import importlib.util
from pathlib import Path

import pytest

from voi_diffusion_ipp.quarter.continuous import ControlSettings

spec = importlib.util.spec_from_file_location(
    'quarter_schedule_runner', Path(__file__).resolve().parents[2]/'scripts/run_full_control_quarter.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_explicit_schedule_reaches_expected_cost_after_60_seconds():
    cfg = ControlSettings(optimism=.5, optimism_anneal_seconds=60.)
    assert [runner.optimism_fraction(cfg, 'fading', t, 300.) for t in (0., 30., 60., 300.)] == [.5, .75, 1., 1.]
    assert runner.optimism_fraction(cfg, 'fading', 30., 600.) == .75


def test_unspecified_duration_preserves_existing_half_mission_schedule():
    cfg = ControlSettings(optimism=.5)
    assert runner.optimism_fraction(cfg, 'fading', 75., 300.) == .75
    assert runner.optimism_fraction(cfg, 'fading', 150., 300.) == 1.


def test_static_optimism_is_unchanged_by_schedule_configuration():
    cfg = ControlSettings(optimism=.5, optimism_anneal_seconds=60.)
    assert runner.optimism_fraction(cfg, 'proposed', 200., 300.) == .5


@pytest.mark.parametrize('duration', [0., -1.])
def test_invalid_annealing_duration_is_rejected(duration):
    with pytest.raises(ValueError, match='duration'):
        ControlSettings(optimism_anneal_seconds=duration)
