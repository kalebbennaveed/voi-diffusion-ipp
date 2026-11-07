# Spatial POAM and RPCBF audit — 2026-11-04

Scope: the separate RMSE-gated learning experiment in `run_rmse_learning.py`
and `rmse_monitoring.py`, with the saved `checked300` recordings. Older
known-wind and learning experiments use different adapters. No old recording,
webpage, wind field, robot equation, or safety parameter was changed in this audit.

## Wind and uncertainty

The physical field is a single fixed function

\[
w:\mathbb R^2\longrightarrow\mathbb R^2,\qquad w=w(x,y),\qquad
\partial_t w=0.
\]

It is the saved horizontal slice at height 0.75 m of the stationary CFD data.
The CFD preparation averaged its flow over a time window; robot execution
uses the resulting fixed map, not a changing CFD solution. Animation advects
particles through that map. Robot motion can change the experienced wind
because position changes, without any temporal dependence of the field.

The sensor is `y_i = w(p_i) + epsilon_i`, with independent Gaussian measurement
noise of 0.1 m/s per component. Only `(x,y)` enters POAM. Timestamps identify
observations and schedule updates; they are not kernel features. There is no
wind forgetting factor, time kernel, temporal process noise, or wind decay.
The monitoring clarity process noise is a separate model.

POAM's covariance is epistemic uncertainty about this fixed unknown function.
Revisiting a location can reduce uncertainty through another noisy observation.
Changing the fitted kernel can also change reported variance. “Nonstationary”
in the attentive kernel means spatial correlations can differ by location; it
does not mean the wind changes with time.

The active model has zero prior mean, prior standard deviation 1.8 m/s per
component, 128 inducing locations, and four spatial lengthscales
`[0.3, 0.8, 1.3, 1.8]` m. Two independent component GPs share an attentive
kernel and covariance. This is a statistical wind estimator, without
cross-component incompressibility or obstacle boundary constraints. Correlation
is based on Euclidean spatial inputs, not fluid connectivity around obstacles.
These are material assumptions, not properties guaranteed by POAM.

## POAM implementation checks

Compared with the [POAM paper](https://arxiv.org/html/2406.03669v1), the adapter
implements the attentive kernel, pivoted-Cholesky inducing selection,
unweighted transported statistics (19–21), variational updates (7–8), and
the full predictive conditional residual (4). Its scalar likelihood is extended
to two independent vector components. Kernel training uses the variational EM
objective (10), with actual observation history only.

Implementation choices differ from the authors' example: SI coordinates and
wind units instead of data normalization; four fixed physical lengthscales;
128 geometry-initialized inducing positions before any measurements; known
fixed sensor noise; four Adam steps every two seconds; and an observation-free
transported E step after kernel training to refresh all prediction caches.
They should not be described as an untouched copy of the authors' code.

Independent tests now check both unchanged/permuted and relocated inducing
statistics against dense NumPy calculations, posterior covariance including
its residual, kernel-training cache consistency, and rejection of time columns.
Replaying only the saved actual measurements reproduces both final maps within
their five-decimal recording precision. The replay took 11.9 seconds total for
both scenarios, including JAX compilation; no robot mission was rerun.

The continuous field sampler retains spatial coherence, but is numerical:
32 spectral nodes and 697 support locations, then a 0.2 m cubic cache. At 99
off-grid diagnostic points using the reconstructed final posteriors:

| Numerical comparison | Scenario 1 | Scenario 2 |
|---|---:|---:|
| Sampler covariance relative Frobenius error | 0.874% | 0.468% |
| Maximum sampler variance error, (m/s)^2 | 0.00442 | 0.00285 |
| Maximum cache mean-vector error, m/s | 0.0727 | 0.0262 |
| Maximum cache standard-deviation error, m/s | 0.00894 | 0.0138 |

These are sampled numerical checks, not uniform bounds. Neither sparse
inference nor cached sampling should be called exact full-GP simulation.

## Exact active safety settings

The plant remains the nominal double integrator with quadratic relative-air
force:

\[
\dot p=v,\qquad \dot v=u+d(w(p),v),\qquad
d(w,v)=\frac{0.1}{0.680}\|w-v\|_2(w-v).
\]

| Setting | Active value |
|---|---|
| Actuator bounds | Each component of u in [-1.6,1.6] m/s² |
| Geometric margin | 0.06 m, for the simulated point robot |
| RPCBF backup gain | 2 /s |
| Backup horizon / timestep | 1.2 s / 0.15 s; eight RK4 steps |
| Backup wind fields | Two coherent antithetic posterior samples |
| Policy value | Maximum over fields and full cubic-spline extrema in time |
| Barrier alpha | 1 /s |
| Extra wall HOCBF | Rate 1.5 /s; separate from the original RPCBF construction |
| Execution control / integration | 0.05 s / 0.01 s |
| Planning relaxation | Quadratic hinge-slack weight 15; hard actuator box |
| Execution optimization | Hard CBF QP; minimum common CBF slack when infeasible |

The local wind set used in the robust derivative is

\[
\mathcal W(p)=\{w:\|w-\mu(p)\|_2\le R(p)\},\qquad
R(p)=3\sqrt{\sigma_{post}(p)^2+0.1^2}\quad\mathrm{m/s}.
\]

Here sigma is the per-component posterior standard deviation, read from the
cubic cache, not uncertainty in wind speed magnitude. The 0.1 term is the
existing observation-noise floor, not temporal wind variation. There is
currently **no independently certified physical wind envelope**.

For each barrier gradient, the QP enforces

\[
\nabla_v B^T u \le -\alpha B-\nabla_p B^T v
-\max\left\{\max_s\nabla_v B^T d(w_s(p),v),\;
\overline{\sup}_{w\in\mathcal W(p)}\nabla_v B^T d(w,v)\right\}.
\]

The force supremum is conservatively bounded using the exact quadratic law:
an analytic Lipschitz bound and a 32-angle enclosure with a Lipschitz remainder.
Taking the smaller of these two upper bounds preserves enclosure of the
declared disk. Tests cover 25 random configurations, 9 gradients, 3001 angles
and 5 radial levels. This validates the implementation of the enclosure; it
does **not** establish that the actual wind lies in the disk.

The active nonlinear policy-value gradient passed finite differences. The
hard QP and minimum-slack result passed independent SciPy QP/LP comparisons
on 60 cases with 13 constraints, including infeasible and degenerate problems.
All 26 selected regression tests passed in 51.03 s. Passing solver tests does
not turn an infeasible state or invalid disturbance envelope into a safe one.

## Bounds fail against the recorded field

The following uses simulator truth strictly for offline diagnosis, not for
fitting, planning, or control. Coverage is measured on 2361 free grid cell
centers at the saved belief frames, with rounding uncertainty about 1e-5 m/s.

| Check | Scenario 1 | Scenario 2 |
|---|---:|---:|
| Maximum true speed on free grid, m/s | 9.826 | 16.709 |
| Initial disk coverage | 97.16% | 94.62% |
| Final disk coverage | 51.97% | 50.70% |
| Worst final error / local radius | 21.75 | 23.11 |
| Whole-map RMSE: initial → final, m/s | 1.643 → 2.231 | 2.523 → 4.110 |
| Mean posterior variance reduction | 94.4% | 92.9% |
| Recorded infeasible execution commands | 32 | 17 |
| First infeasible recorded command, s | 101.80 | 13.65 |
| Contact | Lower wall at 103.310 s | None in 300 s |

This is strong evidence of overconfidence and spatial model/coverage mismatch.
The bounds fail before learning, then shrink despite substantial error.
An isotropic 2D Gaussian disk of radius 3 sigma would contain 98.89% of draws
at a single point under a correctly specified model (without the added noise
floor); it is not a 99.7% vector bound, a hard bound, or a whole-mission bound.

The initial radius is 5.408 m/s. At zero velocity and zero mean, its maximum
quadratic-force magnitude is 4.301 m/s², while the maximum actuator vector
norm is 2.263 m/s². Thus even the currently undersized prior envelope admits
disturbances stronger than control authority. This does not make every motion
infeasible, but it rules out an unconditional claim that the filter can hold
position or stop everywhere.

## Safety conclusion and required corrections

The code follows the tested finite-sample RPCBF calculation, but the current
experiment does **not** satisfy sufficient conditions for a general safety
guarantee. The [RPCBF paper](https://arxiv.org/html/2410.11157v3) explicitly
distinguishes a robust infinite-horizon value from finite-horizon and
finite-disturbance-sample approximations.

Outstanding conditions are:

1. A wind envelope justified independently of an overconfident GP; the present
   three-sigma disks demonstrably exclude the truth. Calibration must be checked
   on withheld spatial locations, not only measured-path fit or shrinking variance.
2. Adequate backup disturbance coverage and a justified horizon/terminal
   condition. Two fields do not exhaust the disk or all plausible spatial fields,
   and the 1.2 s horizon drops the future tail without a verified terminal set.
3. Account for interpolation, integration, and sample-and-hold errors in any
   claimed certificate; the current implementation has no proved error margin.
4. Handle changes to the barrier when POAM is updated. A static true wind does
   not make the estimated barrier static; safety-set compatibility at updates
   has not been established.
5. Remain in a robustly feasible region with the actual actuator limits. Positive
   CBF slack is a violation, not a safety certificate. The fallback mixes wall
   and squared-obstacle barrier scales and cannot recover a guarantee by itself.

Increasing a multiplier alone is not a complete repair: it can make the QP
infeasible without improving the wind estimate. Turning off uncertainty because
the wind is static would hide the estimation error. No such changes were made.

## Changes and reproduction

This audit added spatial-only API guards and explicit stationary-data validation,
removed an int64-to-int32 pivot-index warning, and added independent regression
tests plus offline coverage and replay tools. Physical and planning results are
unchanged. Existing recordings and webpage assets are preserved.

Run from the `voi-diffusion-ipp` repository, using fresh output filenames:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_spatial_wind_safety_audit.py tests/test_poam.py tests/test_rpcbf.py \
  tests/test_rmse_learning.py tests/test_implicit_monitoring_v2.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/airflow/audit_spatial_wind_safety.py --output /tmp/spatial-safety-audit.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/airflow/audit_spatial_poam_replay.py --output /tmp/spatial-poam-replay.json
```

Evidence is saved in `output/rmse_gate_20261104/spatial-safety-audit.json`,
`spatial-poam-replay.json`, and `spatial-safety-tests.log`.
