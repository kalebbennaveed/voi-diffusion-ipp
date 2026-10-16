import json
from dataclasses import replace

import pytest

from voi_diffusion_ipp.config import Config, config_dict
from voi_diffusion_ipp.experiments.suite import completed_run


def test_resume_rejects_mismatched_experiment_and_incomplete_output(tmp_path):
    cfg = Config()
    path = tmp_path / "corridor_proposed_10.json"
    path.write_text(json.dumps({"config": config_dict(cfg)}))
    assert not completed_run(path, cfg)
    path.with_suffix(".npz").touch()
    assert completed_run(path, replace(cfg, seeds=(10, 11)))
    with pytest.raises(ValueError, match="Configuration differs"):
        completed_run(path, replace(cfg, obstacle_radius=0.65))
