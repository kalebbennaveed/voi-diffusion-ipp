"""Compare legacy and reusable M-step with the SAME stable pivot tie handling."""

import json
import subprocess
import time
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from voi_diffusion_ipp.wind.poam import POAM, select_inducing

ROOT = Path(__file__).resolve().parents[1]
source = subprocess.run(
    ["git", "show", "78c32af:src/voi_diffusion_ipp/wind/poam.py"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout
legacy = types.ModuleType("legacy_poam")
legacy.__package__ = "voi_diffusion_ipp.wind"
exec(compile(source, "legacy_poam", "exec"), legacy.__dict__)  # noqa: S102 - pinned local Git reference
legacy.select_inducing = select_inducing
models = [legacy.POAM, POAM]
arrays = []
rows = []
for model in models:
    b = model(
        jnp.array([[-1.0, -0.5], [-1.0, 0.5], [1.0, -0.5], [1.0, 0.5]]),
        jax.random.PRNGKey(9201),
        spectral_count=4,
    )
    record = {}
    timings = []
    for i in range(4):
        x = jax.random.uniform(jax.random.PRNGKey(9202 + i), (20, 2), minval=-1.0, maxval=1.0)
        y = jnp.stack([jnp.sin(x[:, 0]), 0.3 * jnp.cos(x[:, 1])], axis=1)
        t = time.perf_counter()
        losses = b.update(x, y, jax.random.PRNGKey(9210 + i), train_steps=10)
        timings.append(time.perf_counter() - t)
        record.update({f"{i}_param_{k}": np.asarray(v) for k, v in b.parameters.items()})
        record.update(
            {
                f"{i}_{k}": np.asarray(getattr(b, k))
                for k in ["mean", "covariance", "a", "B", "z", "old_K"]
            }
        )
        record[f"{i}_losses"] = np.asarray(losses)
        record[f"{i}_prediction"] = np.asarray(b.predict(x)[0])
    arrays.append(record)
    rows.append({"implementation": model.__module__, "update_seconds": timings})
errors = {k: float(np.max(np.abs(arrays[1][k] - v))) for k, v in arrays[0].items()}
out = {
    "label": "M-step equivalence with identical deterministic pivot handling in BOTH implementations",
    "timing": rows,
    "maximum_errors": errors,
    "tolerances": {"atol": 3e-6, "rtol": 3e-5},
    "original_issue": "Uncorrected pivot ties amplified 1e-7 M-step roundoff into a different inducing selection; old raw snapshot is retained. This comparison intentionally isolates the gradient refactor after fixing ties.",
}
(ROOT / "docs/budgeted_planner/poam_reusable_gradient_check.json").write_text(
    json.dumps(out, indent=2) + "\n"
)
for k, v in arrays[0].items():
    np.testing.assert_allclose(arrays[1][k], v, atol=3e-6, rtol=3e-5, err_msg=k)
print(json.dumps({"timing": rows, "largest_error": max(errors.values())}, indent=2))
