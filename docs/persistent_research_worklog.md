# Active fast two-region development — 2026-10-16, 23:15 UTC

The user explicitly asked why work stopped at task calibration: continue through
implementation and experiments, not another task-design-only final response.
Production source is now being extended on main after old jobs stopped. Do not
resume old frozen experiments; their exact source revisions remain in Git.

Optional linear drag is implemented consistently in simulator, nominal and
backup controllers, RPCBF drift and dense collision checks. Eleven related tests
passed. Full details/retained failures are in `docs/two_region_physics/`.
The old acceleration-only dynamics are time-reversible for rest-to-rest paths.
A restricted quintic timing screen exaggerated the initial gamma=.6 route gap;
flexible speed optimization reduced it from 49% to 2%, so it was rejected.
The corrected 144-case screen (40/80 path intervals) found gamma=1, along-wind
acceleration=1.8: middle outbound 2.832 s, outer return 4.954 s versus middle
return 7.047 s. All 144 optimizations converged; dense control bounds were checked.
These are known-wind path-family physics diagnostics, NOT a learning advantage.

Experimental `planning/budgeted.py` now uses actual MBD and actual spline RPCBF
with tabulated full-POAM fields, a compact observation-conditioned future policy,
and no inner recourse optimizer. `configs/two_region_budgeted.yaml` has a
1-second first stage plus 9-second future stage. Cache code moved to `wind/cache.py`;
original profiler imports it. Full independent uncached checker is available.
`docs/budgeted_planner/profile.json` has three calls: steady-state cache+MBD+full
selected-plan verification 0.940 and 0.966 seconds; maximum control discrepancy
0.00051 on these initial-state plans only. Real online POAM update and execution
are not included yet. Startup tracing/compilation is explicitly separate. Do not
claim the complete one-second cycle or learning acceptance.

A unit test initially failed because its 0.6-second reference saturated all
controls and could not reveal a posterior effect; original output is retained.
The 3-second nonsaturated fixture and cache checks now pass (5 tests). Next is a bounded paired
closed-loop diagnostic with actual online POAM and actual safety controls. No
long sweep until full-cycle timing, causal tests and mismatch checks pass. Keep
one heavy worker at a time; inspect results/budgeted_tests.log and subsequent
results/budgeted_missions before launching any duplicate.

A pure-functional reusable POAM M-step gradient eliminates per-update JIT rebuilds.
The mathematical ELBO, ten training steps, real inducing relocation and sufficient
statistics are unchanged. A comparison initially exposed near-tied pivot selection
amplifying 1e-7 floating-point differences. `select_inducing` now restores the
analytically constant kernel diagonal and uses deterministic roundoff-scale ties.
`check_poam_refactor.py` compares old/new M-steps with the SAME stabilized pivot
policy: four batches agree within atol=3e-6/rtol=3e-5; biggest absolute error is
4.77e-6 in a loss. Steady update timings are ~0.013 vs ~0.25 s for four-inducing
component fixture, not yet the 24-inducing mission. Raw pre-fix snapshots and
all outcomes are retained. Release tests are in results/budgeted_release_tests.log.

---

# Two-region shuttle — latest user steering, 2026-10-16

The user has clarified the scientific scenario and authorized grounded method
changes: two LARGE regions A and B, rapid QoI clarity decay, repeated transfers,
two obstacles with a short windy passage and outer bypasses. Assume wind assists
A→B, with B→A possibly choosing a bypass; this is a hypothesis to verify, never
an imposed route. The earlier generic island scenarios are secondary diagnostics.
The one-second/10-second-trajectory requirement below remains in force.

Task feasibility is now established separately from any proposed-method outcome.
Read `configs/two_region_design.yaml`, `scripts/design_two_region_task.py`, and
`docs/two_region_design/{screen,protocol}.json`. The actual clarity ODE runs on
1353 cells, with 90 cells in each 1.4×1.2 m ROI. Positive targets are confined to
A/B; all cells still evolve. The selected development case uses target 0.70,
sigma 0.50, R=0.10, Q=0.16. Unsensed q=0.8 reaches 0.6 in 2.60 seconds and 0.4
in 7.81 seconds. No clarity resets or alternation rewards are applied to paths.

Over 120 seconds a prescribed symmetric shuttle (2-second service, 4-second
rest-to-rest transfer) costs 11.2481. The best of four tested one-region
hover/sweep policies costs 32.9710. More strongly, an independent analytical
covariance calculation gives an optimistic lower bound 31.4847 for ANY policy
confined to one ROI after the initial four seconds. That bound grants zero
initial cost, perfect clarity at four seconds, perfect visited-ROI service and
simultaneously maximal Gaussian cross-sensing of every neglected cell. The
shuttle is 64.27% below that bound. This establishes that returning to both
regions is beneficial for this task; it is NOT a learning benefit or MBD result.
All eight Q/footprint combinations and all dwell/reference outcomes are retained,
including the initial screen before the stronger bound was introduced.

The prescribed central transfer is geometrically collision free in zero wind
(clearance 0.22 m); its analytic peak axis acceleration is 1.083 < 1.6 m/s².
This does not establish wind feasibility or safety-filter behavior. The screen
uses sensor reference paths and does not pretend to run POAM or a planner.
Tests check the whole-region geometry, analytic acceleration bound, convergence
under a fourfold refined integration step, the cross-sensing bound, and zero-Q
limit. Task-design figures are labeled explicitly; do not publish them as a
proposed-method demo. Three targeted tests pass. The initial 0.1-second screen
failed the declared 0.002 cost convergence tolerance against 0.025 seconds
(difference 0.0057583). Reducing the screen step to 0.05 seconds and comparing
against 0.0125 seconds passes the unchanged tolerance; the original screen is
retained as `screen_dt_0p1.json`.

Next establish the directional route-choice physics with a known-wind reference,
then implement the budgeted planner using this task. Inspect whether bounded
acceleration alone captures the intended sustained headwind/tailwind effect;
if an air-relative drag or airspeed model is needed, justify it and update the
RPCBF drift and backup dynamics consistently. Never fabricate direction-specific
route costs. Keep wind inference as full POAM and QoI Q distinct from wind
uncertainty (wind initially remains a fixed unknown spatial field).

The key comparison remains proposed vs a wind-aware comparator that ALSO updates
POAM during actual execution. Do not freeze that comparator to create a benefit.
Use a shared bank of middle/outer route initializations if needed for optimization,
with MBD choosing/refining motion. No forced shuttling or route. The user accepts
small regressions in low-value cases: the declared target is now one-sided 5%
noninferiority, replacing the old symmetric ±5% equivalence test. Eight fresh
paired seeds per regime remain necessary once development stabilizes. The
current expected-cost method is not mathematically an optimism/UCB objective;
do not add a clipped positive VoI term based on that conversational description.

Do not resume expensive old sweeps. Preserve frozen checkpoints; proceed with
bounded single-worker component/short-transfer tests and the runtime gate.

---

# Runtime redesign — latest user steering, 2026-10-16

The user now requires **a 10-second trajectory in approximately 1 second on the
M3 Air**. This supersedes continuing the expensive experiment sweeps below.
Both coordinators were interrupted, all heavy workers were confirmed stopped,
and numeric checkpoints/partial telemetry remain intact. Status snapshots before
stopping are retained as `status_before_runtime_redesign.json` beside each status.
Do not restart either sweep. The existing heartbeat now prioritizes bounded,
single-worker latency research. No long validation study until timing and
full-model mismatch checks pass. Existing outcomes remain valid for their old
protocol; do not mix a new implementation into those frozen recordings.

The old main planner made 53 objective evaluations, each with four outer worlds
and 144 safety-filter steps, or 30,528 RPCBF calls per replan. Four inner fields
and eight backup steps yield about 976,896 backup integration steps, plus state
derivatives. Recent planning alone took a median 137.6 seconds excluding the
first call during the three-worker batch. Telemetry generation/execution added
roughly 35–45 seconds per period. These are observed concurrent wall times, not
an isolated component profile. POAM hyperparameter training occurs between
real execution periods, not inside every candidate evaluation.

Investigate shared spatial evaluation of coherent wind samples first, preserving
full POAM inference and actual spline RPCBF/MBD. Numerical interpolation must
be labeled and checked for field, gradient, filter and task error. If insufficient,
develop a clearly labeled multi-fidelity candidate evaluator with full RPCBF
checking of selected trajectories and applied controls. Preserve causal future
learning and the monitoring objective; do not introduce an information bonus.
Measure synchronized steady-state p50/p95, compilation, belief refresh, cache
refresh, optimization and final checks; report all deadline misses. A component
speedup is not a one-second planner. Preview data is recorded execution, not a
live real-time planner and not a completed superiority study.

---

## Bounded runtime investigation completed on 2026-10-16

`scripts/profile_runtime_redesign.py` contains three explicitly experimental
interpolators, outside the production source tree. CPU JAX 0.4.38 measurements
use 32 predeclared random states and eight synchronized repetitions; compilation
is separate. Raw samples and errors are in `docs/runtime_redesign/`.

| Variant | 32 RPCBF calls, median | Cache build | Posterior field refresh | Largest control difference |
|---|---:|---:|---:|---:|
| Original POAM field evaluation | 90–93 ms | — | — | reference |
| Entire draw and mean tabulated, 129 columns | 4.84 ms | 63.9 ms per posterior | requires rebuild | 0.0240 |
| Fixed prior tabulated, exact posterior kernels | 13.08 ms | 65.7 ms per bank | 0.256 ms | 0.0232 |
| Fixed prior and kernel tabulated | 12.93 ms | 62.6 ms per bank | 0.295 ms | 0.0240 |

The coarse 65-column complete-field cache changed a control by 0.336 and is
rejected. Fine-cache tests still observed barrier-gradient errors around 0.12;
zero feasibility disagreements in 32 states is not a safety validation. NaN
outside the interpolation interior makes domain exits explicit; a deployable
planner must reject/refresh such cases. Kernel/inducing changes invalidate the
cache. Covariance inference remains full POAM; interpolated fields are only a
numerical sampling/evaluation approximation. Cache identities are independent
draw indices, never a hidden-world feature available to a branch router.

Decoupling uses the existing Matheron sampler: precompute the unchanged prior
draw, then change only its exact small inducing correction when q(u) changes.
This is based on Wilson et al., ICML 2020,
https://proceedings.mlr.press/v119/wilson20a.html . The experiment keeps the actual
backup rollout, cubic-spline extremum, state derivative and QP. Catmull–Rom field
interpolation is additional to the RPCBF time spline and is labeled separately.
The MBD reference remains Pan et al., NeurIPS 2024,
https://proceedings.neurips.cc/paper_files/paper/2024/file/6a5c38a730b17f3827c09cf6b192be04-Paper-Conference.pdf .

Four actual recorded observation batches were replayed through online POAM with
10 training steps and inducing relocation. Update/construction times were
2.056/0.658 s initially, 0.488/0.0105 s next, then 0.277/0.00884 and 0.277/0.00917 s.
These include tracing costs; only four batches were measured. Full online POAM
is therefore not the main multiplicative bottleneck. This one-off measurement
used the first four execution batches of the preserved archipelago proposed
partial record and `new_poam`, `update`, `planning_belief` with its frozen config.

**No end-to-end one-second result exists yet.** Even a sevenfold component
speedup leaves too many nested calls. Next implement and benchmark a separately
labeled budgeted approximation, rather than resuming the old objective sweep:

1. A warm-started 10-second receding-horizon MBD plan, with a coarse control-knot
   representation and finer executed integration. Do not execute ten seconds
   open-loop. Keep full-grid Q clarity dynamics.
2. Replace the inner Monte Carlo branch-selection optimization with a compact
   observation-conditioned future feedback policy jointly optimized by MBD.
   Its inputs may contain only first-stage terminal state, clarity and posterior
   summaries. Both posterior mean and uncertainty may affect future controls.
   The candidate objective remains causal task cost, with no information bonus.
3. Compare cached full RPCBF against a local approximate candidate filter under
   a declared trust region. Use full RPCBF on selected trajectories and actual
   applied controls; report disagreement and rejection. A frozen local model
   must not silently pretend to account for changed posterior backup values.
   Retain the full evaluator as a development reference, not in every proposal.
4. Allocate an explicit ~1-second total budget including required belief/cache
   refresh, optimization and selected-plan verification. Initial compilation is
   a reported startup cost. Measure full-cycle p50/p95 and deadline misses over
   at least 100 varying calls, not repeated identical data only. Reusing a prior
   plan on a miss must be reported as a missed fresh-plan deadline.
5. Compare candidate ranking, first-action discrepancy, empirical clearance,
   physical collisions and executed clarity against the full model before
   long missions. A fast result with no learning benefit is not acceptance.

Only bounded single-worker profiling continues. No heavy worker remains after
these measurements. Two interpolation/conditioning tests initially passed;
the conditioning test was then expanded to both kernel modes. Ten frontend
tests and the syntax/lint checks passed. Production src and frozen configs were
not edited. Existing previews contain 97.2 seconds of actual execution for both
archipelago arms and proposed channels, explicitly marked stopped/unfinished.
The archived A/B example is directly reachable and remains labeled a legacy
feature-GP experiment; it is not evidence for full POAM superiority.

## Historical study instructions (superseded where conflicting)

# Persistent research continuation — 2026-10-16

The requested end result is an evidence-based demonstration of three regimes:
(1) materially better persistent clarity from anticipating learning, (2) similar
performance when learning has negligible task value, and (3) contact caused by
ignoring wind in safety. The user requested 1–2 minute physical missions,
stronger Q-driven decay, interesting multi-obstacle environments, full POAM and
whole-field clarity, and no interim completion claims. **The research acceptance
criteria are not yet met.** Do not notify completion merely because code runs.

## Active and queued experiments

The main checkout's 122.4-second, six-mission batch is running from
`scripts/run_persistent_suite.py`; consult
`results/persistent_poam/recordings/status.json`. Three CPU JAX workers are
allowed, with numeric checkpoints. Keep **all main src files and declared YAMLs
unchanged** until the frozen study completes. Every source file, including the
demo server, is hashed. Continue using the existing UV environment with
`uv run --no-sync` if dependency synchronization is unavailable.

A separate Git worktree is **inside this implementation repository** at
`tmp/learning-regimes`, branch `codex/learning-regimes`, latest development
commit `08ae94b`. Its queued coordinator is `scripts/run_regime_screen.py` and
its authoritative status is `results/regime_screen/status.json` inside that
worktree. It waits for capacity in the main study and caps the combined heavy
worker count at three. Do not start duplicate coordinators. Each coordinator
has a process-bound caffeinate assertion; neither changes permanent settings.

The worktree uses its own source and the main UV-managed environment:

```sh
JAX_PLATFORMS=cpu PYTHONPATH=src UV_PROJECT_ENVIRONMENT=../../.venv \
  uv run --no-sync python -u scripts/run_regime_screen.py
```

Both new cases execute 25 periods of 3.6 s = **90 s**, with a 7.2 s future
lookahead and repeated causal hypothetical belief updates. Each case has two
development seeds and both methods: wind gates 710/711, quiet basin 720/721.
Both use the full 475-cell POAM model, actual spline RPCBF, and six MBD reverse
levels with 16 candidates per level. Spatial Q has base 0.08 and range near
0.04–0.12: at Q=0.08, unsensed q=0.8 falls to 0.6 in **5.21 s**. Initial QoI
clarity is target+0.04 (capped), an explicit maintained-field initial condition.
Wind-gate uncertainty is 0.55; quiet-basin uncertainty is nonzero 0.03 with
observation standard deviation 0.1. Both methods actually learn in both cases.

Before any new screen worker executed, task-only initial policies were expanded
from three local policies to ten: retained previous/initial policy plus nine
spatial sectors. Future branches offer remote/return goals. Both methods receive
the same bank, and all evaluations count against their equal budgets. This
helps the optimizer consider monitoring across constrictions; it adds no
uncertainty reward, oracle information or change to the clarity objective.
The unchanged two-stage evaluator selects a seed, followed by actual MBD.
The seed-independence/geometry and recourse/demo tests passed: **11 tests**.
Earlier scenario/RPCBF checks passed: **9 tests**. Protocol revisions before
execution and exact hashes are retained in `docs/regime_screen/` in the worktree.
Keep that worktree's src, configs and runner frozen once execution starts.

Inspect whole-field time-integrated deficit, revisit counts, target recoveries,
physical clearances, interventions, POAM errors, branch choices and denoising
convergence. A longer mission does not guarantee learning value; a low-cost
trajectory that repeatedly hovers in one spot is not evidence of broad
persistent monitoring. Check attainable sensing throughput as Q increases.

An internal partial audit at 43.2–46.8 s confirmed actual repeat sensing (37–135
revisited cells across the three active missions) and target recoveries. It is
saved in `results/persistent_poam/partial_behavior_audit.json`. The arms had
different completed lengths at collection, so those partial costs must not be
presented as a completed or paired performance result. This audit checks that
Q-driven persistence is occurring while the full recordings finish.

After the development design is fixed, preregister at least eight fresh seeds
per regime. The current protocol targets >=10% mean reduction with a paired
95% interval excluding zero for useful learning; a paired 90% interval wholly
inside ±5% for negligible value. These are future validation gates, not current
findings. Retain all regressions and collisions. Review the limited sample,
lookahead and optimization budgets before drawing scientific conclusions.

## Completed safety study and demo

All 27 fixed-policy safety pairs were validated and bundled; 24 are declared
fresh seeds. Wind-aware collisions: 1/24; wind-ignored: 2/24; paired exact p=1.0.
Mixed seed 634 is an independently sampled wind-ignored-only contact; shear
623 is a failure of both filters. The previously observed mixed 302 contact
is disclosed as a positive control and excluded from inference. See
`docs/wind_safety_validation/findings.md`, protocol, summary and raw manifest.
Investigate shear623 before suggesting a reliable safety advantage. A subsequent
fully replanned zero-wind-filter baseline is supported in the worktree but has
not been run. The fixed-policy ablation cannot stand in for that comparison.

The demo exposes eight environments with readiness state, actual full-grid
clarity/deficit histories, per-cell Q decay, POAM estimates and denoising. It
also contains all27 safety replays, first-contact playback and failures.
Ten frontend tests passed; the two new safety-isolation tests passed.
All safety bundles passed cost, update-count, source-hash and dense-integration
validation. No reference repository was modified or imported.

The new regime status is served through a catalog entry of kind `study_status`
referencing the nested worktree's status file. Keep that path until results are
preserved. The regime coordinator does not automatically publish finished pairs.
Run `uv run --no-sync python scripts/report_regime_screen.py` from the main
checkout when new results complete. This validates frozen source hashes, declared
configuration overrides, identities, duration and complete execution, then uses
the shared telemetry reporter to publish only completed pairs. It preserves all
seeds, uses separate per-seed figures and shows the full high-Q range. The
translated archival protocol retains the hash of the unchanged original
protocol. Pair outcomes remain explicitly developmental. Six publication guards
were tested against actual full-POAM telemetry (including rejected source,
configuration, identity, duration and truncation errors); the shared reporter
also reproduced the six original full-POAM runs and ten frontend tests passed.
No new long-run outcome was available during this publication preparation.
After frozen runs finish, merge the development branch into the main checkout,
preserve its protocols/results, update the catalog path and remove the worktree
only after its active jobs have finished. The existing heartbeat
`validate-persistent-poam-monitoring` continues this work and should stay quiet
until validated completion or a genuine blocker.

## Active continuation after 7e01f35 (no completion claim)

The 12-second paired `results/budgeted_missions/dev_01` diagnostic completed.
Both methods mostly remained in A; costs 0.201145 proposed and 0.201676 current
belief are essentially equal, NOT a useful learning result. No backup or contact
was reported in that short run. Warmed component-sum planning cycles including
previous real POAM update/snapshot were ~0.93–1.00 s proposed and ~0.89–0.93 s
current belief. These sums exclude input preparation, so future runs must also
record inclusive generation wall time. Actual execution used 20 Hz safety AND
20 Hz wind assimilation while hypothetical learning used 5 Hz observations.
Fix the observation-rate mismatch to 5 Hz assimilation before any accepted study;
retain dev_01 as a diagnostic. Physical controls should still run at 20 Hz.

`diagnose_budgeted_seeds.py` replays the first eight real POAM batches from dev_01
and inspects shared slow/rapid initializations. Rapid transfers help with known
wind (cost to B 0.224 vs staying 0.272), but uncertain forecasts still stall.
`wind/oracle.py` is an explicitly privileged analytic known-wind reference, never
a learning backend; it exists only to isolate physical/task/search limitations.
The duplicate initial `diagnose_budgeted_oracle.py` can be consolidated later.

`check_poam_learning_resolution.py` established a material hypothetical-update
approximation error: with the 6x4 initial inducing grid, ten center observations
leave sigma=0.480 in the frozen-Z forecast versus 0.0316 after actual POAM inducing
relocation. 5x5/7x5 grids have a center point and eliminate that particular floor,
but do not remove floors at arbitrary other positions. Prefer fixing the causal
update to reflect full POAM relocation over selecting a representation solely
for a winning outcome. Raw table: docs/budgeted_planner/learning_resolution.json.

Uncommitted current changes implement optional hypothetical POAM relocation via
JAX pivoted Cholesky and full sufficient-statistic transport; cache relocation
refreshes inducing kernels/prior values without regenerating fixed prior draws.
Config flag `hypothetical_relocation` defaults false, so old configurations retain
fixed-Z behavior. `tests/test_hypothetical_poam.py` compares the full relocated
posterior/statistics against real POAM update with training disabled (same frozen
kernel); cache sampling checks also pass. Ten POAM/budgeted tests passed in
results/hypothetical_poam_tests.log. Full POAM kernel training remains real-only.

Currently running bounded profiler: session 43642, command
scripts/profile_budgeted_planner.py --calls 3 --relocate; output
results/budgeted_relocation_profile.log and docs/budgeted_planner/profile_relocation.json.
Do not start a duplicate. After timings, run the shared-seed diagnostic with
--first-stage 1/2 --relocate; do not assume success. A 2-second stage with fixed
inducing locations did NOT resolve the blocked predicted transfers. Keep the
10-second total horizon and one-second-ish full cycle requirement. No generic
information reward or forced visiting. Current configs still use the original
1+9 seconds until a bounded revision is actually justified and tested.


## Most recent multi-stage checks (still no successful mission)

The single-update relocated profiler completed at ~1.02–1.07 s for cache+MBD+
uncached verification, excluding actual POAM update/input prep. Relocation alone
did not make the recorded-state transfers attractive. A bounded multi-stage
variant now updates the hypothetical full POAM at future period boundaries and
recomputes the shared feedback reference only for future controls. Config field
`budgeted_update_steps=0` preserves the one-stage future; 15 gives updates after
1, 4 and 7 s in a 1+9 s horizon. Four causality/POAM tests pass, including changing
future observations without affecting any earlier controls. Full physical wind
remains the same outer draw across all stages.

`profile_relocation_chunk2_updates15.json` reports ~1.53 s cache+optimization+
full checking, with candidates batched in pairs. This exceeds the one-second
target. The multi-stage shared-seed check improves some bypass predictions, but
still prefers staying in A (roughly 0.272 vs 0.295 for the better transfer). Thus
more causality alone is not yet sufficient. All diagnostics retained. No second
mission has run yet. scripts/run_budgeted_mission.py is being improved to use
5 Hz actual wind assimilation matching the forecast (20 Hz physical safety),
and inclusive generation wall-time rather than only component sums. It accepts
explicit --rapid/--relocate/--future-update-steps/--first-stage/--chunk flags.

Potential next diagnosis: the current zero-mean, globally uncertain initial wind
prior also makes the OUTER routes unpredictable, whereas the user's motivating
case has a localized windy passage with usable outer alternatives. A declared,
common archived survey of outer routes could provide physically justified prior
knowledge without leaking unobserved passage wind; compare this as a separate
initial-information regime, retain the cold-prior failures, and label all such
prior data explicitly. Do not assert this will succeed before measuring.


## Continuation after premature stop: measured A/B missions and active work

The user asked why work stopped. Work resumed; no acceptance claim. A 24-second
known-wind MBD+RPCBF reference completed A/B/A/B, dense minimum clearance 0.127 m
and zero contacts. This is a privileged oracle only. A matched full-POAM pair
with the declared common archived outer-route survey also completed A/B/A/B:
proposed cost 0.496185 versus 0.512685 current-belief (3.2% lower, one development
seed, not substantial/held-out evidence). Both had no dense 32-substep contact.
Raw runs: results/budgeted_missions/outer_survey_01. Auditor independently
reintegrates the actual held controls in NumPy and recomputes whole-grid deficit.

Found an initialization issue: reference POAM initializes inducing inputs from
training data (POAM/src/models/poam_model.py::_init_inducing). Uniform unobserved
initial Z spends sparse capacity away from surveyed paths. New POAM.from_data
selects initial Z from the available data via pivoted Cholesky, then uses the
ordinary full statistic transport/variational update/training. Both priors 0.8
and 1.5 were retained; 0.8 remained the existing configuration, not selected on
proposed wins. High 1.5 uncertainty still blocks useful motion. Diagnostics in
docs/budgeted_planner/outer_survey_{screen,data_supported}.json. No passage truth
is supplied to the estimator. Initial survey is an explicitly different prior
regime; cold-prior failures remain.

Optional exact kernel-column reuse for hypothetical relocation passed numerical
tests, but carrying the large cache through lax.scan increased planner latency
to ~2 s/cycle. It was reverted in the planner, while the tested optional cache
API is retained. Omitting zero-target cells from candidate scoring preserves
the original full-grid normalization and is mathematically exact for this
policy. Actual execution and selected-plan verification still evolve all cells.
This alone did not materially improve timing; retain profile_active_cells.json.
Current profiler reduces MBD proposals from eight to six at all four genuine
reverse levels, keeps two outer worlds and three RPCBF samples, and restores the
faster non-carried cache. Session 13294; log results/budgeted_6_proposal_profile.log,
output docs/budgeted_planner/profile_6_proposals.json. Do not start duplicates.

New scripts/audit_budgeted_mission.py independently audits saved controls/costs.
New scripts/export_budgeted_demo.py reconstructs POAM from saved real data, checks
every snapshot, and replays stored MBD vectors for genuine candidate-path
visualization. It has not yet been run. Export time is separate from planning.
Next: finish latency measurement/tests, declare/run a 120-second development pair
(single heavy worker), audit it, export both arms and verify the interactive UI.
Then diagnose whether the central passage is actually learned/used: repeated
outer-route shuttling alone is not the requested positive learning case. Do not
launch held-out seed sweeps before that mechanism works.


### Latest user steering during the 120-second run

The user explicitly accepts **about two seconds per generated trajectory, as
long as its horizon is ten seconds or longer**. The earlier one-second target
is superseded. Prioritize quality/learning behavior within ~2 s; do not keep
reducing proposals to force a hard one-second deadline. The frozen 120-second
6-proposal study is already running (session 34419, results/persistent_two_region_01.log);
finish it unchanged and retain its outcome. Its 10-second horizon remains valid.
The four-call benchmark was ~1.14–1.16 s cache+MBD+full check; actual mission
cycles including POAM/input preparation are ~1.25 s. Startup compilation is
separate and remains substantial. Relevant 21 numerical/causality/cache/drag
tests and 10 frontend model tests passed.


### Square-root POAM repair and fresh 120-second pair

The original 6-proposal proposed run completed 120 s (cost 2.77085, 17 A/B
transfers, dense clearance 0.06293 m, zero contacts). The comparator failed after
75 s: accumulated B lost PSD under repeated float32 transport (min eig -3.9e-5),
causing K+B/noise² Cholesky failure. Raw partial run and numeric failure fixtures
remain in results/budgeted_missions/persistent_two_region_01 and
docs/budgeted_planner/poam_failure_period_*.npz. This is not a completed pair.

New exact square-root form transports B with QR of [P.T L_B, K_uf]. Variational
updates whiten by Cholesky(K+jitter I) and solve I+F F.T; S is formed as V.T V.
No covariance reset, forgetting, or information reward. Both real and hypothetical
updates use this algebra. Thirteen POAM/hypothetical/planner tests and an additional
independent NumPy float64 algebra test passed. Replaying all retained 75 comparator
and 120 proposed observation updates is finite with positive variational spectra.
Evidence: docs/budgeted_planner/poam_square_root_long_replay.json. First replay
script hit a relative-path output bug after successfully checking 75 updates;
retry log retained separately. Driver now records finite-belief status and stops
immediately after saving any invalid posterior, rather than proceeding another cycle.

Fresh frozen protocol: docs/two_region_design/persistent_root_protocol.json.
Same task, seed 8101, same prior survey and 10 s horizon; restore 8 proposals
because the user now accepts ~2 s. Run both full 120 s arms serially into
results/budgeted_missions/persistent_two_region_root_01. Preserve src/ and configs
while active. Still no validated substantial learning benefit. Exporter and UI
are being completed independently; publish only complete numeric runs with
posterior-reconstruction checks and genuine recorded-MBD replay.


### Fresh root-statistic pair completed; export and causal audit

Session 32295 finished both 120 s arms. Proposed cost 2.85379934, comparator
3.02178144: 5.559% lower, one development seed. Dense independent 32-substep
audits found zero contacts in both, proposed minimum clearance 0.12452 m.
Proposed 20 transfers versus comparator 15. Steady proposed planning median
1.410 s, p95 1.462 s, maximum 1.681 s; no >2 s steady cycles. Startup separate.
This satisfies the solver-latency target but the recording replans every one
simulated second, so it is NOT a wall-clock-real-time control deployment. Next
test should use --first-stage 2 --future-update-steps 10 for a 2+8 s horizon with
updates at 2,4,6,8 s, preserving actual 20 Hz control and 5 Hz observations.
Config.__post_init__ already rejects a future-update interval that does not
divide future_steps; this was verified before any new cadence experiment.

Exporter first attempt checked eager rather than jitted posterior predictions,
failing a 3e-5 std tolerance on one cell (3.445e-5 difference) at period 5. It now
uses exactly the mission's jitted snapshot; first forecasts and log are retained
under results/demo/failed_forecasts/nonjitted_snapshot_persistent_two_region_root_01.
Retry session 69803, results/export_persistent_two_region_root_01_retry.log, has
completed the proposed export and is exporting the comparator. Do not duplicate.
All source remains frozen until export and same-source diagnostic finish.

New scripts/diagnose_budgeted_choice.py (not yet run) reconstructs actual POAM
and evaluates identical recorded candidates at the same state/clarity under both
causal objectives, using three independent banks of eight outer fields. It checks
that first-stage states agree and logs signed continuation effects, not a clipped
reward. Default periods 18 and 26 examine passage approach/return in the proposed
record. Use this before launching fresh-seed sweeps; 5.6% is below the declared
substantial-benefit threshold. UI adds A/B regional clarity curves and measured
planning times; still needs browser verification after both exports complete.


### Same-state causal diagnostic and interactive verification

Session 12599 completed docs/budgeted_planner/causal_choice_root_01.json. At
periods 18 and 26, three independent eight-world banks selected the same candidate
under anticipated and frozen-future-belief objectives. Signed future-cost changes
for the executed selection were roughly -0.00047 to +0.00053 at period 18 and
-0.00012 to +0.000084 at period 26, against costs around 0.23–0.26. First-stage
states agree. Thus these two decision points DO NOT establish a robust deliberate
learning mechanism; the 5.6% mission difference alone is insufficient. Do not
launch a superiority sweep based solely on this example.

Both 120 s records are now exported and served at
http://127.0.0.1:8765/?run=budgeted_persistent_two_region_root_01_proposed_8101 .
Every real POAM snapshot was reconstructed and checked. Execution playback,
four actual diffusion levels, final selected path, final results table, all-grid
clarity and wind diagnostics were inspected in the browser. UI has A/B regional
clarity curves; older galleries/provenance are collapsed and the new pair is the
default for a fresh page. Final long paths are faded and update markers thinned
with an explicit legend, while all numeric samples remain available.

Scientific next diagnosis: unknown zero-mean GP versus a known windy-passage
motivation, controller authority relative to componentwise disturbance,
short-lookahead value and sampling error. A common imperfect coarse weather
forecast as a declared GP prior mean, with full POAM learning its residual, is a
possible justified refinement, not yet implemented or validated. Any such mean
must be specified independently of physical truth and used consistently in
prediction, coherent sampling, conditioning, RPCBF and cache checks. Never leak
truth or discard the conditional residual. Larger inner/outer sample diagnostics
and a two-second execution cadence should precede stronger acceptance claims.


### Active two-second cadence check

User's two-second/ten-second request is being checked at a matching simulated
replanning cadence as well: session 15458 runs the two full 120 s arms serially
(60 replans), --first-stage 2 --future-update-steps 10, same eight proposals/two
outer worlds and same task/prior. Protocol: persistent_2s_protocol.json; log:
results/persistent_two_region_2s_01.log. Early steady cycles are ~1.5 s. Do not
start duplicates or change src/configs. Existing root_01 1 s-cadence demo stays
available; publish this new pair only after completion/audit/POAM reconstruction.
Exporter now derives belief-update times from config and writes publication time;
new records are distinguished by their replanning cadence. Final display still
uses original uncached selected paths. Previous exporter source checks prevent
accidental reuse of incompatible forecast checkpoints; completed old replays
need not be regenerated after a label-only exporter update.


Do not compare jet amplitude 1.8 directly with componentwise actuation bound 1.6:
the center disturbance is approximately (-1.36,-1.23), so known wind can be
cancelled by the box-bounded controls. The more relevant prior issue is that
the current initially zero-mean passage posterior often predicts a milder wind
than reality; learning a strong mean can offset the benefit of smaller variance.
A declared common imperfect prior mean for a KNOWN windy passage is a scientifically
reasonable next diagnostic. Reference mapping now records Hewing et al. nominal
model + GP residual (Eqs. 1–2) and Klenske/Hennig's causal dual-control example
(Eqs. 7–9). These sources ground the investigation; no performance/safety
guarantees are imported and no forecast extension is implemented yet.


### Two-second cadence completed and published — continuation entry point

All heavy jobs in this turn finished (15458, 56390); no experiment source freeze
remains. Both full 120 s arms at 2 s cadence completed, were independently audited
at 32 physical substeps and exported with every POAM snapshot reconstructed.
Current playable pair:
http://127.0.0.1:8765/?run=budgeted_persistent_two_region_2s_01_proposed_8101
The earlier one-second-cadence pair remains separately selectable.

Proposed cost 2.83932948 versus comparator 2.95490885: 3.911% lower in one
development seed. Transfers 16 versus 14, dense minimum clearance 0.08991 m
versus 0.12724 m, zero contacts. Proposed steady generation cycles: median
1.72538 s, p95 1.98404 s, max 2.03845 s; 2/58 exceeded 2 s. Startup cycles
25.84 s and 3.11 s plus initial POAM setup 3.19 s, reported separately. Actual
40-control simulation per 2 s period averaged 0.121 s CPU after startup. This is
a solver/cadence benchmark, not deployed asynchronous real-time control.

Important retained limitation: the comparator's period 6 winning cached plan
failed the full uncached check and used the backup. Actual execution remained
contact-free. The demo explicitly shows this rejected forecast in red with a
backup explanation. Post-export display_annotation.json records the annotation
change and before/after replay hashes; numeric data are unchanged. Maximum
cached replay versus full selected-path displacement: 0.202 m proposed, 0.137 m
comparator over 10 s; finite objective disagreement max 0.601% / 0.278%, p95
0.0289% / 0.0163%. Retain uncached selected-plan checks; do not call the cache
exact. Three current cache/gradient/conditioning tests passed in 7.58 s.

Browser verification: 120 s paired playback, 10 s forecast/2 s executed label,
four denoising levels, final paths/costs/timing, period-6 rejection label, POAM
wind plots, regional clarity oscillations and whole-grid histories. No browser
console warnings/errors. UI caches the full deficit history to avoid repeatedly
processing millions of clarity samples per animation frame. Ten frontend tests
pass; both live exports also passed independent duration/array/cost checks.
Screenshots: results/demo/persistent_two_region_2s_demo.jpg (current result),
results/demo/persistent_two_region_clarity.jpg (earlier cadence clarity histories).

Research is NOT complete: 3.9%/5.6% development gains and the weak same-state
causal result do not establish substantial learning value. The 2 s pair's final
whole-field POAM RMSE is ~0.219/0.201 while mean sigma ~0.068/0.072, suggesting
miscalibration to investigate spatially (global averages alone are not a formal
coverage test). Next bounded work should inspect posterior calibration, common
imperfect passage-weather prior means with full POAM residual learning, and
independent-bank decision stability. Read the primary references just added in
reference_mapping.md. Preserve every prior/cadence regime and failure. Do not
launch fresh-seed superiority sweeps until an observable causal decision benefit
is robust to sampling. The original three acceptance claims remain open.


### Additional comparator requested, 2026-10-17 UTC

User clarified and explicitly approved ADDING a wind-blind nominal planner plus wind-aware execution safety, alongside the existing comparator. Preserve both existing arms. New `planning/wind_blind.py` accepts no wind belief; uses actual MBD on still-air unfiltered dynamics, known drag/obstacles and original clarity objective. Real execution continuously uses full POAM/RPCBF, but no nominal wind compensation. This comparison combines wind-aware planning and learning; it must not be labeled an isolated anticipation gain. Six component/causality tests pass. Protocol `docs/two_region_design/safety_only_protocol.json`; launch single 120 s job in `results/budgeted_missions/persistent_two_region_safety_only_01`, log `results/persistent_two_region_safety_only_01.log`. Keep src/configs frozen until run/export complete.

Offline calibration diagnostic completed in `docs/wind_calibration_01`: both original 2 s posterior histories reconstructed exactly within tolerance. At final passage grid points, production 24-point model has only 31.9%/43.3% component coverage within +/-1.96 sigma (proposed/comparator), descriptive on one field. Forty inducing points improve RMSE but coverage only 44.3%/52.9%. Frozen-after-survey kernel diagnostic gives 98.1%/99.0% coverage with much wider uncertainty; NOT a replacement for full POAM. Learned effective passage length grows from ~1.06 to 1.63/1.70 m, while amplitude falls modestly .634 to .442/.419. Prequential observation coverage 93.3%/91.8% is substantially better than spatial passage coverage. Thus calibration across unvisited space and kernel adaptation need diagnosis; adding sparse capacity alone does not resolve it. No estimator changes made. Source used for this diagnosis predates the added baseline; all raw outcomes retained.


### Wind-blind nominal comparator complete and published — latest continuation

Session 41342 (mission) and 15091 (export) completed; session 83615 tests passed.
No heavy worker or source freeze remains. User explicitly chose to ADD this
comparator alongside the existing wind-aware/current-belief comparison.
120 s, same seed 8101/task/initial survey/10 s horizon/2 s cadence:
wind-blind nominal MBD + continuous full-POAM RPCBF cost 3.49864674, eight A/B
transfers, five fallback periods, no dense 32-substep contacts, minimum clearance
0.110872 m. Median/p95 planning cycles 0.06747/0.07557 s. Proposed cost 2.83932948
is 18.8449% lower; current-belief cost 2.95490885 is 15.5414% lower. These contrasts
combine wind-aware planning/filter prediction with anticipation and MUST NOT be
presented as isolated learning gains. Same four MBD levels/eight proposals, but
not equal wall time; a higher-budget nominal optimizer is a future fairness check.

Protocol/results: docs/two_region_design/safety_only_{protocol,results}.json.
Raw data: results/budgeted_missions/persistent_two_region_safety_only_01.
Live record: budgeted_persistent_two_region_safety_only_01_wind_blind_planner_8101.
All 61 POAM snapshots reconstructed within tolerance. Replay candidate/selected
nominal paths agree to 4.34e-7 m; UI integrated cost differs by 8.57e-8. Original
proposed/current-belief exports unchanged. Explicit cross-revision pairing checks
config, task hash, seed, duration, initial observations/state/clarity, physical
wind and geometry and records both source hashes. Added branch defaults preserve
existing planner behavior; original causal tests pass.

Demo selector is now explicit: Wind-aware · no anticipated learning versus
Wind-blind plan + safety filter. Playback, final outcome/fallback table, both
MBD histories and wind-blind forecast labels were checked in the browser without
console warnings/errors. Screenshot results/demo/wind_blind_comparison.jpg.
The URL baseline=wind_blind_planner selects the new comparison; current_belief
retains the original. Seven distinct Python component/causal checks and eleven
frontend checks passed, plus source lint and independent mission/array audit.
Nominal-control test explicitly verifies no wind compensation, and the execution
filter still changes controls to brake near a boundary. No estimator was replaced.

Next research remains calibration and a robust causal learning mechanism. The
calibration findings above rule out treating more inducing points alone as a
solution. A useful next offline check is a dense GP posterior on the SAME earned
observations and final learned kernel, comparing transport/compression error with
kernel misspecification. This is diagnostic only, not a replacement for POAM.
Then investigate a common independently declared coarse weather prior + learned
POAM residual if justified. Preserve fixed-kernel diagnostic as diagnostic, not
production. Do not launch positive held-out sweeps based on the 18.8% contrast;
original anticipation-only development gain is still 3.9%, with weak same-state
causal evidence. The three original research acceptance cases remain open.
