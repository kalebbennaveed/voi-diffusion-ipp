# RMSE-gated wind learning and separate MBD

This additional diagnostic preserves all earlier recordings. It uses the same
5-obstacle geometry, stationary CFD wind, robot and clarity equations.

The user authorized simulation truth for a diagnostic handoff condition. After
requesting a lower threshold than the initially proposed 0.5 m/s, the new value
was stated as **0.2 m/s**. At a two-second checkpoint, compute

\[
E=\sqrt{\frac1{N_{free}}\sum_{j\in free}\|\mu(p_j)-w_{true}(p_j)\|_2^2}.
\]

Switch only if E <= 0.2. This is whole-map **vector** RMSE over 2361 free cells,
not component RMSE, posterior variance, trajectory-only error or a safety
certificate. The true field is not used to fit POAM, choose a waypoint, evaluate
MBD candidates or construct RPCBF. Only the diagnostic phase decision sees it.
State, velocity, clarity, clock and posterior carry across the phase boundary.
The webpage shows the two phases separately. No MBD trajectory is invented if
learning fails the accuracy condition.

## Diagnosis

The previous baseline switched at 60 seconds regardless of accuracy. It used the
V5 performance distribution r*mu + sqrt(r)*(posterior_draw-mu), which attenuated
both mean and spread in poorly supported regions. The new baseline evaluates
unmodified POAM posterior draws. Its inference remains the original sparse
variational POAM with transported unweighted sufficient statistics, attentive
kernel EM and retained full predictive residual.

An actual-data replay found low measurement-path residual but poor extrapolation.
At the end of the old Scenario 2 survey, residual to its observed data was 0.227
m/s while whole-map RMSE was 4.339 m/s. Recomputing sufficient statistics from all
recorded data gave 4.096 m/s; a full GP with the same kernel and data gave 3.596
m/s. Thus online compression contributes error but does not explain the failure
alone. This comparison is offline diagnosis, not a replacement estimator.

The safety replay also found a numerical defect in ExtendedTableBasis: clipping
to exactly lower+dx can evaluate to grid coordinate 0.9999999999999998. The
underlying cubic cache treats floor(0.999...)=0 as out of range and returns NaN.
At the old S2 baseline t=185 s, backup trajectories crossed the padded lower
edge; wind tables were finite, but every backup barrier and gradient became NaN.
The fix puts a 0.001-cell guard on the lower extension, matching the existing
upper guard. All physical workspace queries are unchanged. A regression test
checks all four outside edges, float32/64, gradients and interior equality.

## Additional safety constraints

Four wall constraints are added to the same hard execution QP and relaxed
planning filter. For each inward unit normal n, h=distance-to-wall-margin,

\[
n^T u + \min_{w\in\mathcal W} n^T d(w,v)
+2\lambda n^T v+\lambda^2 h\geq0,\qquad\lambda=1.5/s.
\]

The existing certified force enclosure evaluates the quadratic drag on the same
3-sigma wind disk. No force law is linearized. Actuator bounds remain hard. The
common-slack fallback remains explicitly infeasible, not certified safe. This
added condition depends on valid disturbance bounds and initial barrier
conditions; sampled implementation and changing beliefs preclude a general
safety claim. These constraints did **not** eliminate the S1 collision.

## MBD

Original independent MBD reverse chains: 15 s horizon, 60 control intervals,
120 acceleration coordinates, six geometric initializations plus hold and a
separate shifted previous incumbent. Two reverse levels and 14 proposals give
43 original search evaluations. Seven optimized mode leaders are rescored to
choose an alternative for validation; the global winner and best distinct
optimized leader receive independent hard-filter checks. Hold is additionally
checked if neither passes. The runner ends the attempted MBD experiment rather
than execute an unvalidated failed fallback. Ending a recording is not claimed
as a physically safe stop.

Two antithetic coherent fields evaluate expected monitoring cost, two independent
fields validate, and two full posterior fields construct RPCBF. Finite Monte
Carlo evaluation, 32-node support-corrected prior spectral quadrature, 0.2 m cubic
wind tables, 0.25 s search integration, and the finite 1.2 s backup horizon are
existing approximations. Validation/execution retain .05 s control and .01 s
physical integration. No new robot, wind or clarity approximation is introduced.

The separate module timing check (a prior belief, **not an accepted handoff or
successful monitoring run**) measured warm total times 0.767 and 0.740 s. This
includes field generation, geometry and optimization; the search plus seven
rescores took ~0.38 s, validation including failed hold ~0.32 s. First compilation
and call took 20.27 s. Only two warm samples: no runtime bound is established.

## Executed results

| Scenario | Duration | Final wind RMSE | Contact | Infeasible commands | MBD starts |
|---|---:|---:|---|---:|---|
| 1 | 103.4 of 300 s | 2.231 m/s | 103.310 s, lower boundary | 32 | No |
| 2 | 300 s | 4.110 m/s | None recorded | 17 | No |

All saved state, control and slack values are finite in these runs. All actual
measurements correspond to executed locations. Controls stay within ±1.6 m/s².
RMSE is independently recomputed from saved means and truth to <1e-5 m/s.
Neither run achieves the learning requirement. A/B visits do not establish a
successful learned-wind monitoring method; no such success is claimed.

Reproduce from the repository root, choosing a fresh output directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/airflow/run_rmse_learning.py \
  --output results/airflow/rmse_learning_new --seconds 300 --survey-seconds 300 \
  --rmse-threshold .2 --train-kernel --wall-limit 180
```

The new page is `/revised/#wind-learning-rmse`; MBD is `/revised/#mbd-rmse`.
All prior recordings are preserved. The packaging audit checks 145 preexisting
files byte-for-byte and verifies that removing the four added HTML blocks
recovers the previous index exactly.
