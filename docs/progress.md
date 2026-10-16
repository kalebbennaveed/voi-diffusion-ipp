# Implementation checkpoints

## Phases 0-1

Native ARM64 macOS 15.7.4, Xcode command-line tools present. UV 0.10.8,
Python 3.11.15, JAX/jaxlib 0.4.38 CPU. Environment workload passes JIT,
Cholesky, grad, vmap, scan; four physical/GP tests pass. See generated
`results/environment_cpu.json`. Metal assessment is pending separately.

The finite-feature GP is exact for its explicitly declared kernel; it is not POAM.
Next: spline RPCBF and constrained QP, then diffusion and sequential integration.

## Phases 2-3

Nine CPU tests pass, covering spline maxima against SciPy, the paper's exact
stopping gradient, RPCBF finite differences, QP solutions against independent
SLSQP, explicit infeasibility, MBD's analytic Gaussian-convolution score (with
Monte Carlo standard-error tolerance), multi-level optimization and equal budgets.
Full and local-window spline variants retain policy-value and QP logic. No safety
guarantee is inferred. Next: causal two-stage rollout and held-out missions.

## Phase 4

Ten CPU tests pass, including sequential terminal-state continuity, conditioning
on executed-position measurement values, unused-branch invariance, and explicit
rejection of truth-field objects at the RPCBF interface. Full sequential scoring,
fixed clarity continuation, passive, IG, reset/replay heuristic, oracle, CEM and
MPPI share the same rollout. First coupled mission is running; behavioral success
has not yet been established.

## Phase 5 and ongoing Phase 6

Sixteen CPU tests pass, including learned-belief RPCBF admissibility, empirical
wall-collision prevention, no-truth-leakage first-action comparison, finite-field
consistency, one/two-dimensional MBD and POAM variational statistics. Full MBD
missions and POAM integration smoke runs execute. Both Metal options fail on
this host's required workload; CPU remains the validated backend.

The posterior-to-filter mechanism is demonstrated at matched states. A robust
mission-level benefit from deliberate sensing is NOT yet established. Small-budget
paired trials and a higher-budget initialization/horizon diagnostic are running.
An equal-budget nonconvex trajectory benchmark favors MPPI over MBD on ten seeds;
this negative solver result is retained. All-invalid optimization is now an
explicit fallback event, not a discarded mission.
