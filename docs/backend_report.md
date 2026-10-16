# Backend validation on the supplied M3 Mac

Host: ARM64 Apple M3, 16 GB, macOS 15.7.4, Xcode command-line tools present.
UV 0.10.8. Python environments, package caches and downloaded interpreters all
live under this repository.
Copies of the environment, timing and failure logs are committed in
[validation/](validation/), alongside the 19-test CPU log.

## CPU: validated

`uv.lock`: Python 3.11.15, JAX/jaxlib 0.4.38. Actual device TFRT_CPU_0.
The environment workload tests JIT matrix multiplication, Cholesky, automatic
differentiation, vmap and scan. Component tests additionally cover the GP,
spline extrema/gradients, QP, MBD and full two-stage rollout. CPU missions run
without globally disabling JIT. Raw evidence is in `results/environment_cpu.json`
and `results/backend_benchmark.json`.

A small complete two-stage kernel compiled in 2.047 seconds and executed in
0.00212-0.00266 seconds; peak process RSS was 376.6 MB. These numbers describe
that small kernel, not a full replan. Mission JSON records the actual larger
optimizer timing and process peak RSS separately.

## Official Apple jax-metal: workload unsupported

Reviewed Apple's guidance and PyPI metadata before installation. Isolated
`.venv-metal`: jax-metal 0.1.1, JAX/jaxlib 0.4.34, Python 3.11.15, NumPy 1.26.4.
Exact resolved versions are in `scripts/metal-requirements.txt`.

Sandboxed execution reported zero visible devices. With device access, the
plugin selected **Apple M3 / METAL** but failed compiling `mhlo.cholesky` in the
representative workload. No full planner GPU execution or GPU speedup is claimed.
`ENABLE_PJRT_COMPATIBILITY` was not necessary for this matching version pair.
JIT was not disabled. Raw failure: `results/environment_metal_unsandboxed.log`.

## Third-party metaljax: incompatible shader library on this OS

Reviewed upstream project/PyPI description and beta status. Isolated
`.venv-metaljax`: metaljax 0.11.9, JAX/jaxlib 0.11.2, Python 3.12.13. This is a
third-party PJRT plugin, **not official Apple/JAX support**. With GPU access, its
bundled metallib failed: language version 4.0 is unsupported on this OS.
Raw failure: `results/environment_metaljax.log`. It is not a package dependency.
No operating-system upgrade or global package changes were made.

## Reproduction

```sh
UV_CACHE_DIR=.uv-cache uv sync --frozen
JAX_PLATFORMS=cpu UV_CACHE_DIR=.uv-cache uv run python scripts/check_environment.py
JAX_PLATFORMS=cpu UV_CACHE_DIR=.uv-cache uv run python scripts/benchmark_backend.py
# Isolated official plugin experiment (known failure on the tested workload):
UV_CACHE_DIR=.uv-cache uv venv .venv-metal --python 3.11
UV_CACHE_DIR=.uv-cache uv pip install --python .venv-metal/bin/python -r scripts/metal-requirements.txt
JAX_PLATFORMS=METAL .venv-metal/bin/python scripts/check_environment.py
```

References checked 2026-10-16:
- https://developer.apple.com/metal/jax/
- https://pypi.org/project/jax-metal/
- https://github.com/eterevsky/metaljax
- https://pypi.org/project/metaljax/
