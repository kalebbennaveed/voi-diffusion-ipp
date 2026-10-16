"""Numeric-only optimization checkpoints for deterministic mission recovery.

Recovery replays executed dynamics, observations and POAM updates from the seed;
it reuses only previously computed optimizer results. No pickle files are read.
"""

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..config import config_dict
from ..planning.mbd import DiffusionHistory, OptimizationResult


def source_hashes():
    root = Path(__file__).resolve().parents[3]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "src").rglob("*.py"))
    }


class PlanCache:
    def __init__(self, directory, cfg, method, seed):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        metadata = json.loads(
            json.dumps(
                {
                    "config": config_dict(cfg),
                    "method": method,
                    "seed": seed,
                    "source_sha256": source_hashes(),
                }
            )
        )
        manifest = self.directory / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != metadata:
            raise ValueError(
                "Checkpoint configuration or numerical source changed; use a new output directory"
            )
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")

    def load(self, horizon):
        path = self.directory / f"plan_{horizon:04d}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            hist = DiffusionHistory(
                **{k: jnp.asarray(data["history_" + k]) for k in DiffusionHistory._fields}
            )
            opt = OptimizationResult(
                **{
                    k: hist if k == "history" else jnp.asarray(data[k])
                    for k in OptimizationResult._fields
                }
            )
            return opt, float(data["seconds"])

    def save(self, horizon, opt, seconds):
        path = self.directory / f"plan_{horizon:04d}.npz"
        temporary = path.with_suffix(".tmp")
        arrays = {
            k: np.asarray(getattr(opt, k)) for k in OptimizationResult._fields if k != "history"
        }
        arrays.update(
            {"history_" + k: np.asarray(getattr(opt.history, k)) for k in DiffusionHistory._fields}
        )
        arrays["seconds"] = np.asarray(seconds)
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
