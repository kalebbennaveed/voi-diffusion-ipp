"""Task calibration must establish revisiting value independently of planner wins."""

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
module = importlib.util.spec_from_file_location(
    "two_region_design", ROOT / "scripts/design_two_region_task.py"
)
design = importlib.util.module_from_spec(module)
module.loader.exec_module(design)


def test_regions_are_large_disjoint_and_shuttle_is_geometrically_feasible():
    spec = yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text())
    _cells, masks, obstacles = design.layout(spec)
    assert masks.sum(1).tolist() == [90, 90]
    assert not np.any(masks[0] & masks[1])
    times = np.arange(12001) * 0.01
    path = design.reference_path(spec, times, "shuttle", 2)
    np.testing.assert_array_equal(path[0], spec["region_a"])
    clearance = (
        np.linalg.norm(path[:, None] - obstacles[None, :, :2], axis=-1) - obstacles[None, :, 2]
    )
    assert clearance.min() > 0.20
    # Exact componentwise maximum acceleration of a quintic rest-to-rest move.
    peak = (
        (10 * np.sqrt(3) / 3)
        * np.abs(np.array(spec["region_b"]) - spec["region_a"])
        / spec["transfer_seconds"] ** 2
    )
    assert peak.max() < spec["maximum_control_acceleration"]


def test_shuttle_beats_optimistic_confined_region_bound_with_refined_integration():
    spec = yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text())
    cells, masks, _ = design.layout(spec)
    lower, error = design.confined_region_bound(spec, cells, masks, 0.5, 0.16)
    assert error < 1e-6
    costs = []
    for dt in [0.05, 0.0125]:
        fine = spec | {"dt": dt}
        times = np.arange(round(fine["mission_seconds"] / dt) + 1) * dt
        path = design.reference_path(fine, times, "shuttle", 2)
        out = design.evaluator(cells, masks, fine)(jnp.array(path), 0.5, 0.16)
        costs.append(float(out[0]))
        assert np.all(np.asarray(out[3]) >= 0) and np.all(np.asarray(out[3]) <= 1)
    assert abs(costs[0] - costs[1]) < 0.002
    assert max(costs) < 0.5 * lower


def test_bound_retains_cross_region_sensing_and_zero_decay_limit():
    spec = yaml.safe_load((ROOT / "configs/two_region_design.yaml").read_text())
    cells, masks, _ = design.layout(spec)
    narrow, _ = design.confined_region_bound(spec, cells, masks, 0.5, 0.16)
    broad, _ = design.confined_region_bound(spec, cells, masks, 3.0, 0.16)
    zero, _ = design.confined_region_bound(spec, cells, masks, 0.5, 0.0)
    assert 0 <= broad < narrow
    assert zero == 0
