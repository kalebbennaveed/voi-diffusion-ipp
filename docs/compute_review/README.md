# Current method and compute review

Revision: 21 October 2026. Runtime tuning and new scenario development were
stopped at the user's request. This review documents the current implementation;
it does not claim a 10-second horizon or a two-second planning deadline has been
achieved.

The editable presentation is
`output/presentation/wind_aware_mbd/slides.qmd`. Open `slides.html` in the same
directory for the rendered presentation and its manual POAM/MBD explanations.
The earlier anticipated-learning method and its results remain at the end under
**Archived method**.

## Current decision rule

The decision vector specifies the nominal reference trajectory. Each performance
field produces controlled motion, sensing, integrated monitoring deficit and
soft constraint penalties. The score is the lower partial mean of monitoring
costs plus the full mean constraint penalty. Filter infeasibility invokes its
declared minimum-slack fallback; slack alone does not reject a trajectory.
The disturbance posterior is frozen inside planning and updated by actual
observations during nonlinear execution.

MBD uses a cosine schedule and the original score/reverse update. The archive
mixture changes the proposal and includes the full importance-density ratio.
RRT trajectories initialize the entire independent acceleration sequence;
the demonstration-likelihood term is disabled in this configuration.

## Measurement record

`current45_dynamic.json` records synchronized wall times for the unchanged
45-second configuration, using the forward-wind flight's attained state and
posterior at 100 seconds. It contains all warm repetitions, first-invocation
times, configuration values, source hashes, work counts and device metadata.

Dynamic state/field/trajectory inputs are passed into the compiled diagnostic
kernels. The script checks that separate search and selection produce the same
selected parameters and score as the complete optimizer.

| Operation | Warm median |
|---|---:|
| Complete planning call, including fields and RRT | 10.752 s |
| Fused optimizer, search plus selection | 9.298 s |
| Search, compiled separately | 5.712 s |
| Final selection, compiled separately | 3.644 s |
| RRT | 1.119 s |

See the JSON for unrounded values. Separately compiled stages can have different
fusion and scheduling, and nested timings overlap: do not add all table rows.
Three warm repetitions of the large stages describe this state on this machine,
not a mission-wide worst case. First invocation of the optimizer took about
25.19 seconds and includes compilation.

The dominant nested work is controller-aware prediction: 136 training trajectory
vectors in four fields plus 46 selection entries in eight fields = 912 rollouts.
At 225 steps each, those imply 205,200 forecast filter calls. Each builds two
eight-step braking continuations and their state derivatives. POAM updates and
field construction are much smaller than the optimizer cost.

The raw runner's `plans[].seconds` times only the optimizer. Its
`total_planning_seconds` adds RRT but omits field-bank generation, diagnostics,
audit compression, JSON serialization and physical execution. The complete-call
profile includes field generation and RRT, but deliberately excludes reporting,
plotting and flight simulation.

## Reproduce profiling

Use Python 3.11 and the dependencies in `pyproject.toml`. The checked experiment
configuration is in `current45_dynamic.json`. A portable saved profiling state
is included in `profile_input.json` and `profile_input.npz` (about 800 KB). It
contains the attained state, clarity, preceding trajectory continuation, sampled
fields and the 100 actual observations needed to reconstruct the posterior.

To profile that exact state from a fresh clone:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/profile_current_quarter.py \
  --fixture docs/compute_review/profile_input.json --repeats 3 \
  --output output/compute_review_reproduced.json
```

The full local raw flight archives are preserved at the record path in the
measurement file; they are not duplicated into Git.

To reproduce the same measured state when those archives are present:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/profile_current_quarter.py \
  --record output/european_quarter/planner_investigation/supported_forward_reverse300/two_forward_proposed_8101.json \
  --plan-index 20 --repeats 3 \
  --output output/european_quarter/planner_investigation/compute_profile/reproduced45.json
```

To generate a new local recording using the exact configuration without needing
the existing five-minute recording:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/run_compute_review_recording.py --seconds 10 --output output/compute_review_fresh
```

Then profile its initial state using `--record` for the resulting JSON and
`--plan-index 0`. That is a different state, so its timing need not equal the
100-second measurement. Use the portable fixture above for the measured state.

The shorter-horizon benchmark in `scripts/benchmark_short_horizon.py` is an
unfinished experiment. No warm benchmark or performance claim from that
experiment is included in these slides.

## Rebuild the presentation

With Quarto installed:

```sh
.venv/bin/python docs/presentation/build_current_review.py
cd output/presentation/wind_aware_mbd
quarto render slides.qmd
```

The full Quarto source is directly editable. The assembly script preserves the
existing preliminaries and final archive, and inserts `current_method.qmd` and
`current_implementation.qmd`. The implementation template takes its timing
values directly from `docs/compute_review/current45_dynamic.json`.

## Files to investigate

- `src/voi_diffusion_ipp/quarter/continuous.py`: reference integration, controller-aware rollouts, penalties, final comparison.
- `src/voi_diffusion_ipp/quarter/mbd_core.py`: independent trajectory parameters, proposal correction, score update.
- `src/voi_diffusion_ipp/quarter/rrt.py`: geometric diversity and stop-at-corner quintic initializations.
- `src/voi_diffusion_ipp/safety/flow_policy_value.py`: braking values, cubic-spline maxima, forward derivatives and filter assembly.
- `src/voi_diffusion_ipp/safety/qp.py`: 2D projection and common-slack LP fallback.
- `src/voi_diffusion_ipp/quarter/environment.py`: synthetic air-velocity field, geometry, posterior and field lookup.
- `src/voi_diffusion_ipp/quarter/execution.py`: nonlinear quadrotor execution with the runtime filter.
- `scripts/run_full_control_quarter.py`: field generation, actual observation timing, physical execution and audit recording.

Configuration entries `slack_weight`, `guidance_sigma`, and `jerk_limit` are
inactive in the selected hard-filter / guidance-off / zero-jerk-penalty path.
The nominal `safety_margin` is overridden by the execution margin for the hard
prediction filter. Prediction and physical control rates remain different.
