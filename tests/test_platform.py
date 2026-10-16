import ast
from pathlib import Path

import jax

from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.planning.mpc import planner


def test_configs_hashable_and_no_reference_imports():
    for path in Path("configs").glob("*.yaml"):
        cfg = load_config(path)
        hash(cfg)
        planner(cfg, "proposed")
    forbidden = {"mbd", "rpcbf", "vg_dip", "torch", "gpytorch"}
    for p in Path("src").rglob("*.py"):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(a.name.split(".")[0] not in forbidden for a in node.names)
            if isinstance(node, ast.ImportFrom) and not node.level:
                assert (node.module or "").split(".")[0] not in forbidden
        assert "sys.path" not in p.read_text()
    assert jax.devices()
