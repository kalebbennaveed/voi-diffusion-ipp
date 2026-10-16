#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.uv-cache"
export MPLCONFIGDIR="$PWD/.mplconfig"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
uv sync --frozen
uv run python scripts/check_environment.py
uv run pytest -q
uv run python -m voi_diffusion_ipp.experiments.run --config configs/smoke.yaml
uv run python -m voi_diffusion_ipp.viz.report results/runs
