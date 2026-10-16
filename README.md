# Value of Information Informative path planning using model based diffusion

A self-contained JAX research implementation of causal two-stage value-of-information
planning for persistent monitoring under a fixed unknown spatial wind field.
It includes paper-equation model-based diffusion (MBD), sampled spline RPCBF with
a constrained QP, correlated GP inference, an optional POAM adapter, baselines,
paired experiments, plots, and offline HTML replays.

**Research status:** the software and component mechanisms are tested on the M3
CPU. The original formulation did not establish a reliable mission-level advantage.
A revised feedback/recourse variant reduced mean cost by 20.9% on ten fresh paired
seeds in a controlled localized-wind scenario (eight wins, two losses, no collisions,
one infeasible executed QP). Read the [new findings](docs/recourse_findings.md),
[research workflow](docs/research_workflow.md), and
[original measured findings](docs/experimental_findings.md) for the scope of this
evidence. Generality and diffusion superiority remain unestablished. Sampled RPCBF is approximate;
no formal safety or optimality guarantee is claimed.

## Setup on Apple Silicon

```sh
cd voi-diffusion-ipp
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export MPLCONFIGDIR="$PWD/.mplconfig"
export JAX_PLATFORMS=cpu
uv sync --frozen
uv run python scripts/check_environment.py
uv run pytest -q
uv run python -m voi_diffusion_ipp.experiments.run --config configs/smoke.yaml
uv run python -m voi_diffusion_ipp.viz.report results/runs
```

`uv.toml` keeps the UV package cache local. The Python pin is native ARM64 3.11;
`uv.lock` pins the tested CPU stack. `scripts/smoke_test.sh` runs setup, component
tests and the integrated smoke mission. No reference checkout is required at
runtime. Nothing imports from or edits the sibling repositories.

The official Apple plugin was tested in an isolated environment and failed on
Cholesky. The third-party metaljax build failed on this macOS version's supported
shader language. **The full GPU planner is unsupported on the tested stacks.**
See [backend evidence](docs/backend_report.md) and [Linux/CUDA portability](docs/linux_gpu.md).
CUDA is not installed on this Mac. Keep plugin environments separate from the CPU lock.

## Experiments

```sh
# Small paired suite: 3 scenarios, 3 physical-field seeds, all primary baselines.
uv run python -m voi_diffusion_ipp.experiments.suite --output results/evaluation
# Focused scenario, with one common simulator and fixed physical fields.
uv run python -m voi_diffusion_ipp.experiments.run --config configs/corridor.yaml
# Larger-budget, explicitly exploratory corridor diagnostic.
uv run python -m voi_diffusion_ipp.experiments.run --config configs/extended.yaml --output results/extended
# Model and solver ablations.
uv run python -m voi_diffusion_ipp.experiments.suite --ablations --methods proposed --scenarios corridor --output results/ablations
# Original reset/replay selection structure, current-belief-only, fixed continuation.
uv run python -m voi_diffusion_ipp.experiments.suite --methods draft_selection current_belief fixed --output results/additional
# Optional POAM integration smoke (not a scientific POAM mission evaluation).
uv run python -m voi_diffusion_ipp.experiments.run --config configs/poam.yaml --output results/poam_smoke
# Timing and independent nonconvex solver comparison.
uv run python scripts/benchmark_backend.py
uv run python scripts/benchmark_optimizers.py
uv run python scripts/diagnose_learning.py
uv run python scripts/compare_rpcbf_variants.py
# Longer-mission check on fresh seeds, with the same small planning budget.
uv run python scripts/long_mission_diagnostic.py
# Render plots, distribution summary, and offline replay files.
uv run python -m voi_diffusion_ipp.viz.report results/evaluation
```

The suite deliberately uses 4 diffusion levels, 12 proposals, 4 worlds, 4 inner
fields and 10 backup integration steps for affordable broad diagnostics. This is
less expensive than the default smoke profile. Every saved JSON contains the
resolved configuration. `dev` and `research` profiles increase the budgets; the
research profile is provided for larger follow-up evaluations and is **not claimed
to have completed on this laptop**. Larger tests are not real-time on this M3.
Suite resume requires matching configurations and both JSON/NPZ outputs; use a
new output directory or explicit `--overwrite` to change an existing experiment.
After numerical code changes, also use a new directory or `--overwrite`: resume
checks configurations, not source-code changes.

Available methods:

| Name | Evaluator / solver |
|---|---|
| `proposed` | Joint causal two-stage controls, updated future belief, MBD |
| `cem`, `mppi` | Exactly the same joint objective and evaluation budget, different solver |
| `passive` | Current-horizon task cost; real belief updates, no anticipated learning benefit |
| `weighted_ig` | Current-belief task rollouts minus configurable GP information gain |
| `heuristic` | MBD optimization of the draft's noncausal, positive-clipped reset/replay score |
| `draft_selection` | Passive MBD candidate generation followed by draft reset/replay selection |
| `oracle` | Known physical wind by explicit assumption |
| `current_belief` | Two-period task cost with belief frozen in simulated continuation |
| `fixed` | First-stage MBD with fixed clarity-feedback continuation |

The original draft used Stein-generated candidates. `draft_selection` retains
its selection score but explicitly uses MBD to generate candidates; it does not
claim to reproduce Stein. CEM and MPPI exclude infeasible samples. All-invalid
optimization invokes a labeled clarity-feedback fallback through RPCBF; these
missions remain in aggregate statistics and their fallback count is reported.

Generated data stay in ignored `results/`: JSON metrics, compressed NPZ arrays,
PNG figures, and self-contained HTML replays. A run records actual clarity cost,
ROI access, probe occupancy, posterior error/calibration, clearance, collisions,
QP feasibility/slack, interventions, signed matched continuation benefits,
optimizer traces, compile/planning times and peak process memory. Probe occupancy
alone does not establish deliberate learning. Full raw outputs are retained locally;
concise reviewable evidence is also saved under `docs/`.

## Method and code

- [Mathematical formulation](docs/method.md)
- [Research workflow and literature-informed revisions](docs/research_workflow.md)
- [Feedback/recourse findings and all 37 new missions](docs/recourse_findings.md)
- [Measured findings and remaining research limitations](docs/experimental_findings.md)
- [Complete archived result table](docs/validation/summary.md)
- [Source-to-implementation mapping and licenses](docs/reference_mapping.md)
- [Approximations, failure handling and POAM restrictions](docs/implementation_choices.md)
- [Implementation checkpoints](docs/progress.md)

`wind/` holds beliefs and coherent fields; `safety/` holds spline RPCBF and QP;
`planning/` holds MBD, observation routing, rollout and MPC; `experiments/` runs
paired worlds; `viz/` makes scientific plots and replay artifacts. All package
code lives under `src/voi_diffusion_ipp/`. The original repository license and
initial README title are preserved.
