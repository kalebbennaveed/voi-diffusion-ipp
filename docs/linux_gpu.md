# Linux/NVIDIA portability

No Linux GPU is available in this session. CUDA execution is **not validated**.
The numerical code uses portable JAX operations and has no Metal/CUDA-specific
branches or sibling-repository imports. Do not infer GPU correctness from CPU tests.

## Conservative CUDA 12 environment

On supported Linux NVIDIA hardware, inspect `nvidia-smi` and the driver first.
JAX's current installation guide requires driver >=525 for CUDA 12. Keep the
macOS CPU lock intact and create a separate environment inside this repository:

```sh
UV_CACHE_DIR=.uv-cache uv venv .venv-cuda12 --python 3.11
UV_CACHE_DIR=.uv-cache uv pip install --python .venv-cuda12/bin/python 'jax[cuda12]==0.4.38' 'numpy<3' 'scipy<1.17' matplotlib pyyaml pytest
UV_CACHE_DIR=.uv-cache uv pip install --python .venv-cuda12/bin/python --no-deps -e .
JAX_PLATFORMS=cuda .venv-cuda12/bin/python scripts/check_environment.py
JAX_PLATFORMS=cuda .venv-cuda12/bin/python -m pytest -q
JAX_PLATFORMS=cuda .venv-cuda12/bin/python scripts/benchmark_backend.py
```

Then run a full smoke mission and compare its physics, posterior, spline, QP and
objective values with CPU. Record `jax.devices()`, output placement, driver,
versions, compile/steady times and peak process/device memory. Freeze this tested
CUDA environment separately; do not overwrite `uv.lock` with platform plugins.

## CUDA 13 alternative

The current JAX documentation requires NVIDIA driver >=580 and newer hardware
for CUDA 13. The CPU reference here pins JAX 0.4.38, which predates this extra.
A CUDA 13 migration therefore needs a separate environment with a compatible
modern JAX/Python pair, the complete numerical test suite, and cross-backend
comparison before freezing versions. Do not install a CUDA 13 extra on the old
pin or call an untested resolver result GPU validation.

Source (checked 2026-10-16): https://docs.jax.dev/en/latest/installation.html
