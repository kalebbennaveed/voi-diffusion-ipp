# Implementation checkpoints

Current validation: 30 passing CPU tests, 7 passing frontend tests, 120 original
archived missions, 37 feedback/recourse missions, and 49 repeated-update refinement
missions (including six explicitly superseded development runs).
See [latest findings](multistage_findings.md), [recourse findings](recourse_findings.md)
and [original measured findings](experimental_findings.md); the
entries below preserve the sequence of implementation checkpoints.

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

## Phase 6 — measured baseline and ablation report

Completed 90 baseline missions (ten methods, three scenarios, three seeds),
18 ablation missions, three larger-budget paired diagnostics, and nine longer
missions on fresh seeds. All use the actual spline RPCBF; no penalty substitutes
are used. MBD/CEM/MPPI share objective-evaluation budgets. CEM invalid elites,
fallback reporting and fixed-continuation diagnostics were corrected and affected
experiments rerun. Config-safe resume and numerical observation-routing tests pass.

The larger-budget single-seed case improves cost by 17.28% over the frozen-future
belief comparator. Fresh-seed longer missions show learning is useful but the
proposed objective is 3.36% worse on average than that comparator. The intended
robust VoI advantage is therefore not established. All collisions, infeasible QPs
and all-invalid optimization fallbacks remain in the evidence. The source, lock,
configurations, metrics, distribution tables, selected figures and backend/test
logs are preserved. CPU is validated; both tested Metal stacks are unsupported
for this workload/OS, and Linux CUDA remains untested.

## Research iteration 1 — feedback policies and posterior task values

Added optional waypoint feedback parameters, independent posterior-predictive
continuation values and a localized uncertainty mechanism scenario. Full MBD
and spline RPCBF are preserved. Development seeds 20–22 gave a 7.68% mean cost
reduction. Source commit `cfa1820` and configuration were frozen before seeds
100–109: mean cost reduction was 20.93%, with eight wins and two losses. No
validation collisions occurred; one proposed step used the infeasible-QP fallback.
Task routing alone was not consistently better in the development ablations.
No-measurement execution arrays match exactly. All 37 new missions, tests,
the pre-run protocol, hashes, paired bootstrap intervals and every loss/failure
are archived separately. The result supports a controlled example, not a general
method or solver-superiority claim. See the research workflow for decision gates.

## Research iteration 2 — repeated belief updates and interactive inspection

Repeated causal conditioning inside the continuation improves mean actual cost
by 28.07% versus frozen future belief on predeclared seeds 200–209 (10/10 wins),
and 6.57% versus the preceding two-stage variant (8/10 wins). All 30 fresh-seed
missions have no collisions, but the repeated-update arm has nine infeasible
executed QPs versus one frozen-belief and two two-stage events. Independent LP
reconstruction confirms the failures; stricter simulated-candidate feasibility
did not fix seed 204. All 49 attempts and the superseded comparator warm-start
bug are retained. Numerical source hashes remained frozen during validation.

The local interactive demo records genuine MBD telemetry and separately replays
sampled policies through the full causal evaluator. It displays diffusion,
executed trajectories, beliefs, clarity, comparisons and failure events, and
launches new serialized JAX runs. Eight bundled recordings include losses and
QP failures. Browser verification included a completed live experiment; 30
Python tests and four frontend tests pass. Source and dependencies remain
inside this repository; all five reference repositories remain read-only.

## Demo clarity and soft-constraint interpretation

The user clarified that barrier relaxation is allowed. The numerical minimum-slack
filter is unchanged: positive slack is now assessed separately from physical
collisions and optimizer failures. A report of peak, integrated and duration-based
slack metrics reuses the unchanged 30 validation missions.

The demo now defaults to a paired final-result view with completed trajectories,
initial/later/total task costs and arrival times. Both methods learn during real
execution; only the proposed method anticipates future learning during planning.
Either planner's denoising can be inspected, including its final selected plan,
and mission playback returns to the final comparison. Ten recordings include
the new validation seed 200 pair, which exactly reproduces archived states,
clarity, controls, slack and feasibility. Seven frontend tests validate costs,
comparison orientation, matching worlds and soft-slack accounting. The previous
30 Python tests remain applicable; numerical planner source is unchanged.
See [the precise soft-constraint policy and research stage](soft_constraint_interpretation.md).
