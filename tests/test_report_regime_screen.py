"""Guard publication against incomplete or misidentified scientific recordings."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "report_regime", ROOT / "scripts/report_regime_screen.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture
def recorded_study(tmp_path, monkeypatch):
    # Use actual completed full-POAM telemetry, not fabricated planner outcomes.
    monkeypatch.setattr(report, "ROOT", tmp_path)
    worktree = tmp_path / "worktree"
    records = worktree / "results/regime_screen/screen"
    records.mkdir(parents=True)
    original = ROOT / "demo/data/field_mixed_proposed_302.json"
    d = json.loads(original.read_text())
    file = records / "field_mixed_proposed_302.demo.json"
    file.write_text(original.read_text())
    protocol = worktree / "docs/regime_screen/protocol.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text(
        json.dumps(
            {
                "source_sha256": d["source_sha256"],
                "cases": [["screen", 302]],
                "configs": {"screen": yaml.safe_dump({"profile": "smoke", **d["config"]})},
                "methods": ["proposed", "current_belief"],
                "duration_seconds": (len(d["mission"]["q"]) - 1) * d["config"]["dt"],
            }
        )
    )
    return worktree, tmp_path / "report", file, d


def test_actual_unpaired_record_remains_unpaired(recorded_study):
    worktree, output, _, _ = recorded_study
    _, _, states = report.prepare(worktree, output)
    assert states == [{"scenario": "field_mixed", "seed": 302, "completed_methods": ["proposed"]}]


@pytest.mark.parametrize("fault", ["source", "configuration", "identity", "duration", "truncated"])
def test_invalid_scientific_record_cannot_be_published(recorded_study, fault):
    worktree, output, file, d = recorded_study
    if fault == "source":
        d["source_sha256"]["src/voi_diffusion_ipp/clarity.py"] = "changed"
    elif fault == "configuration":
        d["config"]["clarity_decay"] *= 2
    elif fault == "identity":
        d["metrics"]["seed"] += 1
    elif fault == "duration":
        protocol = worktree / "docs/regime_screen/protocol.json"
        p = json.loads(protocol.read_text())
        p["duration_seconds"] = 90
        protocol.write_text(json.dumps(p))
    else:
        d["mission"]["q"].pop()
    file.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        report.prepare(worktree, output)
    assert not (output / "archival_protocol.json").exists()
