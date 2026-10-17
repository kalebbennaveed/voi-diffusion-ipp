# Value of Information Informative path planning using model based diffusion

A self-contained JAX implementation of causal value-of-information planning for
persistent monitoring under an unknown spatial wind field. It includes actual
MBD reverse diffusion, sampled spline RPCBF with a soft constrained QP, full POAM
wind inference, whole-grid clarity dynamics and interactive recorded comparisons.

The current development case monitors **two large regions A and B for 120 seconds**.
A narrow windy passage and two outer routes connect them. All 1,353 clarity cells
evolve with Q=0.16; the 180 cells in A/B have positive targets. Without sensing,
clarity 0.8 falls to 0.6 in 2.60 seconds. Both planners learn from real measurements;
only the proposed planner anticipates how later measurements affect future motion.

The compact causal feedback-policy variant generates a **10-second trajectory**.
On the M3 CPU, the latest proposed development mission measured 1.41 s median,
1.46 s 95th percentile and 1.68 s maximum steady planning time, including belief
update, cache preparation and a full uncached check. Startup compilation is
reported separately. These are synchronized solver timings, not a deployed
real-time controller: this recording replans every simulated second, and a
wall-clock-consistent two-second execution cadence remains to be tested.

| 120-second development pair, seed 8101 | Proposed | No anticipated learning |
|---|---:|---:|
| Integrated whole-grid clarity deficit | 2.8538 | 3.0218 |
| Completed A/B transfers | 20 | 15 |
| Physical contacts, independent dense audit | 0 | 0 |
| Median / 95th percentile planning time | 1.41 / 1.46 s | 0.80 / 0.85 s |

This is **5.56% lower cost in one development pair**, not a validated substantial
learning advantage. Both arms receive the same declared noisy historical survey
of the outer routes. The passage is initially unobserved. The first 6-proposal
long comparison failed numerically after 75 seconds in its comparator; its raw
outputs remain available. Equivalent square-root POAM statistics repaired the
float32 failure and passed independent algebra and observation-replay checks.
See [the frozen protocol](docs/two_region_design/persistent_root_protocol.json),
[the research worklog](docs/persistent_research_worklog.md), and
[reference mapping](docs/reference_mapping.md).

The budgeted planner uses a restricted continuous posterior-feedback policy and
two outer planning worlds. It retains the full POAM conditional residual, actual
MBD updates and sampled spline RPCBF. Candidate field evaluation uses a tested
continuous interpolation cache; the winning 10-second plan is checked uncached.
Larger scenario-bank checks, substantial held-out learning benefit, a low-value
control and a matched wind-ignored safety case remain research work.

The older [18-second full-POAM comparisons](docs/field_poam_findings.md) and
[fixed-policy wind-safety study](docs/wind_safety_validation/findings.md) remain
available, including all regressions. The earlier expensive 122-second study was
stopped with unfinished recordings preserved. The 90-second regime screen was
also stopped; **do not resume those old queues**. Earlier finite-feature ROI gains
apply only to that historical model, not the current full-POAM method.

## Interactive research demo

```sh
JAX_PLATFORMS=cpu uv run python -m voi_diffusion_ipp.viz.demo_server --port 8765
```

Open <http://127.0.0.1:8765>. **Compare & play** starts an animated, looping
side-by-side replay immediately. Pause, restart, adjust speed or scrub the shared
timeline above the moving robots. **Jump to largest cost gap** locates the largest
accumulated monitoring-cost difference. **Final result** shows completed trajectories,
whole-field clarity, costs and allowed safety slack. Switch between deficit, clarity
and target layers, or click any cell to inspect its clarity history. The wind section
shows full POAM estimates, uncertainty, estimation error and the hidden physical field.
Both methods learn during execution; only the proposed method anticipates learning
while choosing current motion. Inspect either planner's actual diffusion proposals,
weights, score updates and selected trajectories, then play the executed mission with
clarity, wind belief, safety diagnostics and a paired comparator. Full-field POAM
recordings are listed first. Ten legacy recordings remain explicitly labeled for
historical reproduction, including regressions and a no-measurement control.
The interface can launch new JAX runs with different seeds and settings. It has
no frontend dependencies or external assets. See [demo instructions and
provenance](docs/interactive_demo.md). The original MBD playground was used only
as an interaction reference; numerical planning remains in this repository.

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
# Full-model validation missions are offline computations, taking tens of minutes.
uv run python scripts/run_field_suite.py
uv run python scripts/report_field_poam.py
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

Use `scripts/run_field_suite.py` for the current three whole-field paired cases;
`--scenario field_shear` (or `field_vortices`, `field_mixed`) restricts the run.
Use `scripts/validate_full_poam.py` for the independent estimator survey and
posterior-sampler covariance checks. `scripts/report_field_poam.py` validates
recorded costs and provenance before bundling the interactive replay.

The commands below reproduce the **legacy ROI/finite-feature experiments**;
they are separate from the current full-field model.

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
- [Repeated belief updates, 49 refinement missions and demo validation](docs/multistage_findings.md)
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
