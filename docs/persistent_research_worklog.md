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


### Inference diagnostics and active optimizer-budget check, October 17 UTC

The dense/posterior, coarse-forecast residual and half-length-scale diagnostic replays completed; all results and executed-driver provenance are retained in docs/wind_dense_diagnosis_01, docs/forecast_residual_diagnosis_01 and docs/kernel_range_diagnosis_01. Main src/ and configs are unchanged. Detailed findings will follow the expanded-budget audit below.

One new 120 s development wind-blind nominal mission is now declared in docs/two_region_design/safety_only_expanded_budget_protocol.json. It increases only proposals per MBD level from 8 to 256 (four levels, seven starts), testing optimizer effort as an explanation of the 18.8% combined comparison. Output results/budgeted_missions/persistent_two_region_safety_only_256_01; log results/persistent_two_region_safety_only_256_01.log. Preserve src/configs and driver while running; no duplicates or other heavy worker. This is expanded optimization budget, not assumed equal wall time and not an anticipation-only test. The previously published 120 s comparisons remain available.

Completed inference diagnosis (same earned data, no changed physical trajectory):

- Dense GP inference with the **same learned attentive kernel** does not repair the
  spatial undercoverage. At 120 s, passage RMSE is 0.15139/0.16595 and mean sigma
  0.03335/0.02486 for the proposed/current-belief histories; descriptive coverage
  within ±1.96 sigma is 32.38%/23.81%. Online POAM was 31.90%/43.33%. Rebuilding
  batch sparse statistics with the same kernel and inducing locations gives
  36.67%/37.14%. Thus sparse capacity and old-statistic transport cannot by
  themselves explain the confidence problem. These are correlated grid values,
  not independent coverage trials or a posterior safety guarantee.
- Dense inference with the INITIAL post-survey kernel, a diagnostic only, gives
  100%/98.10% passage coverage at 120 s. This does not justify freezing real
  production kernel learning or substituting a dense GP. It points to the learned
  correlation model and adaptive spatial data coverage as remaining issues.
- One prospectively recorded, geometry-based coarse forecast (elliptic Gaussian,
  amplitude 1.0, widths 1.5/0.7, A-to-B direction, zero cross component) was tested
  offline by subtracting it from every earned observation and adding it back to
  predictive means. Full POAM training/relocation/residual covariance remained.
  This is a DEVELOPMENT hypothesis after inspecting this scenario, not independently
  obtained weather data. It improves initial passage RMSE from 0.787 to 0.534,
  but final RMSE is 0.21228/0.15805 and coverage only 47.14%/38.57%. It does not
  solve the calibration problem, so no forecast extension was promoted into src/.
- A single declared half-length-scale POAM prior range (0.15, 0.4, 0.65, 0.9),
  otherwise unchanged, improves final passage coverage to 69.05%/68.10%, with
  RMSE 0.16337/0.17780 and sigma 0.08930/0.11077. The learned length weights
  saturate near the upper limit 0.899. Both improvement and regressions remain;
  this exploratory range is not a calibrated production setting.
- All 600 next-batch sensor predictions are also retained. Mean marginal
  prequential negative log density (lower is better): original -0.37189/-0.26850,
  coarse-forecast residual -0.52564/-0.37462, half-range -0.30347/-0.28122.
  The mixed pattern is another reason not to select a replacement solely from
  the hidden-field grid. These are adaptive-policy histories, not held-out missions.

The independent float64 attentive kernel agrees with production at selected
snapshot queries to at most 1.71e-7; all eight original POAM snapshots replayed
within 3e-5. Every new prediction and parameter remained finite, and diagnostic
training sets were checked against exactly the earned data (minus the declared
forecast in that diagnostic). Numeric arrays, protocols, old/new driver hashes
and original executed driver text are saved. `scripts/report_wind_diagnosis.py`
reproduces the combined report and the visually inspected figure:
[calibration diagnosis](wind_dense_diagnosis_01/calibration_diagnosis.png).

Do not add the coarse forecast merely to obtain an apparent learning gain.
Further bounded investigation should separate correlation/geometry assumptions,
limited posterior-contingent policy expressivity (current shift capped at 0.6 m),
and sampling/optimizer noise. The seven geometric warm starts currently contain
single transfers and staying, not a complete return trip; test improved common
initializations as optimizer diagnostics rather than imposing alternation. The
original three scientific acceptance cases still are not established.

### Expanded nominal-MBD budget completed — current continuation entry point

Session 46256 completed all 60 replans / 120 s. No heavy worker or source freeze
remains. `safety_only_expanded_budget_results.json` contains matched-input checks,
full dense audit, comparisons and raw SHA-256 manifests. The intended change was
256 instead of 8 proposals at each of four MBD levels. A post-run matching check
caught an additional candidate_chunk change (2 to 1). This unintended numerical
batching deviation is retained in safety_only_expanded_budget_deviation.json;
do NOT call this a strict single-parameter comparison. All other config fields,
source hashes, seed, initial observations/state/clarity, physical field and
geometry match. A same-state chunk-equivalence diagnostic follows below.

Cost 3.46489096 versus original nominal 3.49864674: 0.965% improvement with 32 times
as many sampled proposals. Seven A/B transfers, one fallback (versus eight/five
originally), zero dense contacts, minimum clearance 0.101564 m. Independent state
reintegration disagrees by at most 1.73e-7; all-grid clarity cost by 6.34e-8.
Median/p95/max steady planning cycle 0.26395/0.27056/0.27691 s, no 2 s misses;
first two cycles 2.11145/1.35961 s. This is still **not equal wall time** to the
proposed 1.725 s median. A much larger proposal budget alone did not close the
combined contrast in this development seed: proposed cost is 18.05% lower.
This does NOT enlarge the isolated anticipation effect (still 3.9%). No new
success claim or held-out sweep is justified. The original three interactive
comparisons stay published; this additional optimizer sensitivity and all raw
candidate histories are preserved separately rather than replacing the prior
baseline or relabeling its provenance.

The host restart had stopped the local demo server. It was restored at loopback
port 8765 (exec session 98580) and its catalog verified to contain all three
120 s records. No UI or mission export changed in this continuation. Git main
had already been committed/pushed at 382ca22 in response to the user; the new
work in this continuation consists of diagnostics, reports and one bounded
optimizer sensitivity. Old stopped persistent/regime jobs must stay stopped.

The batching diagnostic (session 27080, complete) compares chunk 1 and 2 on the
same saved states at 0, 30 and 60 s, same 256 proposals, common keys and starts.
Selected objective values are identical at all three states. Maximum selected
vector discrepancy is 3.58e-7; all candidate-vector discrepancy 4.77e-7; finite
candidate-cost discrepancy 8.95e-8; zero valid/invalid classification changes.
This supports numerical equivalence at those states but does not erase the
protocol deviation or prove identical full closed-loop missions. Both histories
are retained in safety_only_expanded_budget_batching.npz.

The heartbeat prompt now points to these completed findings and preserves the
user's quiet-notification preference. No held-out success claim has been made.

### User-directed wind reversal study — October 17, active

The user requested opposite wind and route-relevant/irrelevant wind cases, then
explicitly limited this work to the 2D demo and required retaining the original.
Research automations remain paused; none were resumed. The account meter started
at 37% used (63% remaining); stop bounded investigation by 55% used (45% remaining).

`docs/wind_route_study/protocol_01.json` prospectively declares 120 s seed-8101
reversal (three arms), crosswind (two wind-aware arms), and off-route wind (two
wind-aware arms). Only physical truth changes; the planner receives earned noisy
measurements, never the case specification. Original is exactly preserved and
reversed wind is its exact negative. Eight independent-field/equivalence tests
passed. Source, task config, driver, exporter and audit hashes are frozen in that
protocol. Do not change src/configs or those scripts until missions and forecasts
finish. New numerical files contain the resolved physical field and task specs.
Session 72281 runs the reversal sequentially; log results/reversed_wind_01.log.
No duplicate or parallel heavy JAX workers. Old stopped missions stay stopped.

Only demo/app.js and demo/index.html are being enhanced for explicit wind-case
selection, direction-specific route summaries and physical/estimated wind arrows.
Existing 3D files and original recordings are untouched. Additional analysis
scripts do not alter the frozen production source. All outcomes, including
negative results, will be kept; this is development, not held-out validation.

First wind-route screen and all seven 120 s replay exports completed successfully
(coordinator 59752; results/wind_route_study_status.json complete). Freeze for that
screen is lifted; src/ remains unchanged. Original replay retained. Reversal gives
1.285% proposed cost improvement versus current belief, 21.063% versus wind-blind
nominal planning; proposed's one middle transfer reverses to B→A, with 15 bypasses.
Zero dense contacts. Crosswind: proposed cost 3.04879 versus 2.85073, but comparator
contacts at 75.90 s, minimum -0.06261 m (26 control intervals). Proposed has zero
contacts. Its higher raw cost is retained; cost after physical contact is only a
no-impact-response simulation diagnostic. The 2D UI now warns and pauses at first
contact; this was visibly checked at 75.95 s. This is NOT a no-wind-safety ablation.
Off-route field: costs 2.64237 / 2.86059, 7.629% development improvement, zero contacts.
This is not a clean no-learning-value case: the quiet middle route starts unknown.
All independent whole-field clarity replays pass; numeric field checks and
32-substep contact audits are in docs/wind_route_study/results_01.json.

A separate low-value common historical main-route survey is prospectively declared
in low_value_prior_protocol_01.json. The mission driver/exporter are being extended
with routes-survey (the original 42 noisy locations plus 21 noisy passage locations),
with no production method changes. Earlier driver text and all first-screen hashes
remain archived. Do not rerun the original coordinator after this driver revision:
its frozen hash check should now reject it.

Active session 57041 runs scripts/diagnose_observation_latency.py, log
results/observation_latency_01.log. It reproduces 74–80 s of the crosswind comparator
before testing additional earned 5 Hz conditioning within the existing 2 s kernel-
adaptation intervals. This is a fixed nominal-policy filter diagnostic, not replanning.
One heavy worker only. The declared same-state multi-bank round-trip-seed diagnosis
in causal_protocol_01.json has NOT started yet. Automations remain paused; 3D untouched.

Follow-up coordinator session 83287 is active, status in
results/wind_route_followup_status.json, with a new frozen manifest in
followup_source_manifest.json. Source/configs, mission driver, exporter and the two
diagnostic drivers must stay frozen until this coordinator finishes. It sequentially
runs observation-latency diagnosis, the 120 s surveyed-main-route pair plus exports,
and the declared four-bank/four-world causal candidate diagnoses. Nine physical-
field/survey tests pass; the original 42 survey locations are exactly preserved.

The initial observation-latency diagnostic failed before any counterfactual
execution because an unjitted snapshot differed from the archived jitted standard
deviation by 4.37e-5 at two near-zero cells. Failure log, executed driver and protocol
are retained in observation_latency_01. Rerunning with the original jitted snapshot
path (same tolerance) in observation_latency_02 reproduces the original 74–80 s
trajectory **exactly**. Original boundary updates: minimum dense clearance -0.06261,
26 contact intervals. Additional earned 5 Hz conditioning: +0.04241 m, zero contact.
Independent local physical reintegration errors <6.1e-8. Full POAM kernel/inducing
updates still occur at 2 s boundaries; only fixed-kernel earned-data conditioning is
added between them. This is one fixed-policy local diagnostic, NOT a replanned
mission, general safety result, or newly validated proposed method. It supports
investigating shorter observation latency before changing path optimization.

The POAM primary paper was rechecked at https://arxiv.org/html/2406.03669v1,
Sec. IV-C (incremental variational updates) and IV-D (minibatches from all data and
slower hyperparameter adaptation). It supports separating inference and parameter
adaptation; it does not validate this control cadence or establish safety here.

Scope clarification for the latency result: the archived **nominal reference
policies** are fixed, but their wind-mean compensation and RPCBF both receive the
updated belief. Thus this diagnoses controller observation latency; it does NOT
isolate RPCBF alone or hold actuator commands fixed. scope_clarification.json is
stored alongside the result. Any promotion must evaluate both components and
match the forecasting approximation to the faster execution updates.

Follow-up coordinator 83287 is complete; its source freeze is lifted. All five
wind-case 2D comparisons are published, including the original. The surveyed-main-
route case gives costs 2.43553 / 2.67378 (8.91% development gain), so it too fails
to establish negligible value. Its initial passage sigma remains 0.183 m/s².
Four-bank causal checks at six saved states show the same best averaged candidate
in five states, and no robust held-bank advantage in the remaining state. None of
14 added round-trip seeds wins; do not promote those seeds. See the full study
README and retained numeric outputs. Next bounded refinement should address the
compact policy's limited route-conditioned representation (0.6 m per-coordinate
knot shift) and separately the demonstrated observation latency. Do not call
single-seed development contrasts validation. All three original acceptance
cases remain open in their strong scientific form.

### Optional posterior-gated two-route policy — active development

The original compact policy remains default. Branch policy protocol 01 prospectively
declares one reversed-wind 120 s pair, seed 8101, with the same real POAM, sampled
spline RPCBF, MBD budget and objective. The new 24-dimensional representation
optimizes two future routes and an observable-posterior affine gate. Only reference
curves switch at causal belief boundaries; physical and clarity states continue.
23 component tests passed, including timing/causality and default compact tests.
Freeze src/, configs, mission driver, exporter and audit against the protocol until
this pair and its genuine forecasts finish. Keep both prior screens and all failures.
No held-out success claim; automations remain paused; 3D is untouched.

The reversed two-route pair and actual forecast exports finished. Its proposed cost
3.08719 exceeds current-belief 2.89869 by 6.50%; both pass the dense no-contact
audit. The original compact method stays default; this regression is retained.
Session 41157 runs the declared same-state multi-bank diagnosis; source/configs
stay unchanged. A new open-passage low-value protocol is declared but not yet
started: same large regions/Q/off-route wind and common historical route survey,
with radius 0.25 and normal offset 0.95. The mission driver/exporter now support
an archived per-run task path/hash without modifying the original task file.

### Open-passage control, informed diagnostic and horizon diagnosis completed

Open passage: 120 s cost 2.045994 proposed vs 1.996063 current belief (2.50% regret),
zero dense contacts, common direct route. This is development seed 8101 only.
The route classifier initially retained the original gap thresholds. Its original
audits and driver are archived in open_passage_classification_correction. It now
derives middle/bypass boundaries from archived geometry; all genuine forecasts
and trajectories are unchanged, and the open-passage replay was refreshed.

The explicitly privileged reversed-wind compact planner achieves 2.711526 cost,
zero contacts. This is a finite-budget fully informed diagnostic, not an optimal
lower bound or a full-POAM method. It still selects both middle and outer routes.

Optional branch-policy independent-bank checks have the same averaged winner in
all three tested states (one held bank at 30 s favors active selection). Its
executed 6.50% regret remains a negative development result. Four-bank horizon
diagnosis compares 10 versus 14 s on earned reversed states: a round-trip seed
becomes best at 60 s with 14 s lookahead for BOTH objectives. No reliable isolated
anticipation mechanism is established. Actual MBD with all 21 seeds takes about
1.80 s for 10 s and 2.54 s for 14 s at one warm saved state; cold startup ~19 s.
This is not a mission latency claim and the longer horizon is not promoted.

Eight fresh low-value pairs are prospectively declared in
low_value_confirmation_protocol_01.json (seeds 18101–18108, exclude 8101).
The coordinator is scripts/run_low_value_confirmation.py; status at
results/low_value_confirmation_status_01.json, main log low_value_confirmation_01.log.
Do not duplicate it. One heavy worker; arm order alternates; all 120 s per arm.
Primary one-sided paired-t 95% upper regret <5%, with no contacts and all eight
pairs complete; secondary 90% equivalence interval ±5%. This covers a fixed task
and field under stochastic seeds, not a population of environments. All source,
configs, mission/audit drivers, report and coordinator are frozen against that
protocol while active. All original scenarios and 3D remain unchanged.
Automations stay paused; this is a bounded user-requested foreground study.

Additional read-only diagnoses are prepared but NOT started while the eight-seed
study runs: scripts/diagnose_poam_adaptation.py with adaptation_protocol_01.json
compares baseline, 10x slower kernel learning and larger minibatches on frozen
earned histories. The crosswind history stops before physical contact.
The diagnostic scripts/diagnose_planning_horizon.py now accepts --protocol and
--output; its earlier executed revision is preserved in commit 4b5df56. A new
dual_split_protocol_01.json declares 10/14 s lookahead with a single first-stage
hypothetical update followed by fixed-posterior exploitation, motivated by
Arcari et al. (https://arxiv.org/abs/1911.03728). This remains an approximation
and can overvalue early learning by omitting later opportunities; no promotion
or new mission yet. Also run scripts/report_wind_route_extensions.py after
heavy timing measurements finish. Production source/configs remain frozen.

### Fresh-seed result and bounded variance-reduction screen (2026-10-17)

The eight low-value pairs finished, all 120 s with no dense contact. Mean paired
regret is +2.736%, 90% paired-t interval [-3.549%, +9.022%]. The +5% one-sided
noninferiority target FAILED; equivalence also failed. All eight outcomes and
input hashes are retained in low_value_confirmation_results_01.json. All steady
proposed cycles were <2 s. The 2D page exposes every seed, without relabeling the
8101 development demo as fresh evidence. No source/config freeze from that
completed study remains active. Do not repeat or add seeds post hoc.

Post-confirmation coordinator completed: extended report, full-POAM adaptation
sensitivity, and dual/exploitation split. Slower adaptation improves coverage but
worsens next-batch marginal NLL; larger batches do not fix passage calibration.
Neither variant is promoted. Single-update 14 s lookahead took 1.54–1.63 s at a
saved state but changed no averaged route choices in six state/horizon checks.
This remains a diagnostic approximation. All raw predictions and timings saved.

Early-choice protocol completed at 2/4/6/8 s. Every original uncached selected
objective/path reproduces EXACTLY despite the optional branch source extension.
Reversed anticipation changes the averaged winner only at 2 s, with small mixed
held-bank differences. Surveyed narrow off-route case changes winner in three
states; nine held-bank comparisons favor active selection, seven tie, none oppose.
These are dependent development model scores, not independent missions. Main-route
wind can be weak while uncertainty in a narrow gap still has decision value.

Sampling protocol 01 completed on two fixed development states. Actual paired
MBD optimizers are identical in six of eight comparisons; the two different
active choices have smaller independently estimated active cost. Pairing opposite
Gaussian outer field AND observation-noise draws leaves each marginal unchanged,
with independent inner RPCBF draws unchanged. Across candidates finite in all
banks, mean variance is 0.297 and 0.266 of iid; every infeasibility is retained.
This suggests a variance-reduction option, not a demonstrated mission gain.

An optional --outer-sampling antithetic flag is now implemented in the mission
runner and genuine forecast exporter. Default iid is unchanged; source helper
planning/sampling.py does not alter MBD, POAM, RPCBF, the reward or actual noise.
26 component tests pass, including Gaussian second-moment and inner-sample
invariance checks. The declared two-pair development screen is NOW ACTIVE in
session 24191; coordinator scripts/run_antithetic_screen.py, status
results/antithetic_screen_status_01.json, main log antithetic_screen_01.log.
Each pair is 120 s, seed 8101: reversed original geometry and off-route open
passage; arm order alternates. Check per-job logs antithetic_reversed_01.log and
antithetic_open_01.log. One heavy JAX worker only. Freeze all source/configs and
mission/export/audit/coordinator drivers in antithetic_protocol_01.json until
both missions, audits and exports finish. Do not duplicate. Keep any regression.

No 3D changes. No automation resumed. The original and seven development cases
remain in the 2D catalog, with actual denoising and execution. New antithetic
cases should be separately labeled if published, never replace the iid originals.

### Bounded investigation completed; no active research worker

Both antithetic pairs, independent audits and genuine exports finished successfully.
Reversed: proposed 3.05430 vs current 3.03479 (+0.643% regret), no contacts;
open off-route: 1.95237 vs 2.03353 (-3.991%), no contacts. Steady proposed cycles
remain below 2 s. Both development cases are separate playable 2D buttons; the
nine cases include the original. IID remains default. No strong isolated learning
validation or low-value statistical noninferiority has been established.

One reversed antithetic forecast at 16 s passed the cache but failed the full
continuous checker (sampled forecast clearance -0.000717 m). The executed backup
had no physical contact. cache_rejection_01 reproduces it: enlarging the cache
129→257→513 does not recover the clearance sign, and the largest warm cache build
alone costs 2.46 s. Keep the full checker. rejection_alternatives_01 is a post hoc
same-state diagnostic: the next three finite, distinct cached-ranked MBD proposals
all pass full evaluation. The first has +0.039685 m minimum forecast clearance,
full task cost 0.280212, and a 0.292 s warm check; adding that to the originally
measured cycle gives 1.837 s. This is an ESTIMATED sum at one saved state, not a
new mission or implementation of fallback selection. A justified next bounded
change is to check one next-ranked actual MBD proposal before backing up, with
explicit provenance in the demo and a fully replanned comparison. No tolerance
was relaxed and no original trajectory changed.

All research jobs launched in this foreground request are complete, including
sessions 24965, 34048, 94939, 82883, 24191, 58812 and 36576. Source freezes from
these jobs are lifted. Do not resume any old stopped study or paused automation.
The user's usage-reserve limit is near: meter last read 54% consumed (46% left).
Do not launch additional experiments in this turn. Core tests: 26 pass; 2D model
playback tests: 11 pass; Ruff and JS syntax pass. Browser verified nine enabled
case buttons, paired actual MBD levels 1→4, executed playback advancing beyond
16 s, and no console errors. Screenshots and all numerical outputs are retained.
No 3D files changed. All failed criteria and regressions are documented in
README.md and JSON reports under docs/wind_route_study.

### User-requested Safe-MPD comparison (active foreground, October 17)

New user request authorizes testing Safe-MPD instead of RPCBF and careful JAX
porting. Read the reference checkout and paper; details/limitations in
docs/safe_mpd_comparison.md. Original scenario, RPCBF default and all demos remain.
No automations resumed. Five shield/score tests pass; the original budgeted and
MBD tests passed in the same run (16 passed, one initial new-test failure).
The failure was RK4 vs exact exponential at 0.2s: 5.16e-6 exceeds a 4e-6 assertion;
the repaired test uses the analytical fourth-order remainder, not a changed
integrator. Failure log retained. All five new tests now pass.

The completed 8-second pilot has four formulations. Original compact RPCBF
reproduces its archived states and cost EXACTLY over all four replans.
No pilot contacts. Median full cycles: compact RPCBF1.649s, compact sampled
shield1.714s, full-control RPCBF0.844s, full-control Safe-MPD adaptation1.412s.
Pilot durations are too short to judge persistent behavior.

Current ACTIVE coordinator session58813 runs scripts/run_safe_mpd_study.py with
docs/safe_mpd_study/mission_protocol_01.json. Six 120-second original-environment
development missions, one heavy JAX worker: compact RPCBF/proposed, compact
shield/proposed, full-control RPCBF/current-belief, full-control shield without
transformed score/current-belief, full Safe-MPD/current-belief, and discrete
monitor Safe-MPD/current-belief (1 vs4 monitor substeps). All forecast horizons10s,
actual replans2s, seed8101, same survey/physical field. Protocol frozen hashes
cover ALL src, configs, mission/audit/coordinator drivers. Do not edit them while
active. Status results/safe_mpd_mission_01_status.json; logs named by job.
No duplicates. All failures must be retained. Unknown-wind terminal low-speed
test is empirical and not the certified invariant set assumed by the paper.

After completion: report all jobs with scripts/report_safe_mpd_study.py, inspect
plots and dense physics audits. A prepared scripts/benchmark_safe_mpd_jax.py
does same-state chunk1/2/8 timing with explicit lower/compile and synchronized
warm calls. It is NOT started while missions run. Can then test justified
bounded sensitivities. Full-control Safe-MPD freezes hypothetical belief; it is
not an anticipated-learning implementation. The shield-only compact arm retains
causal hypothetical POAM but not the transformed-control score; label separately.

First six missions are now COMPLETE; source snapshot committed as61f8ade.
Results in docs/safe_mpd_study/mission_results_01. Direct compact shield contacted
at109.60s and made0 transfers; raw post-contact cost5.41443 is not mission success.
Full transformed Safe-MPD had no contacts but only1 transfer and cost4.41457;
endpoint versus dense monitoring produced identical physical data in this seed,
with median cycles0.567 versus1.764s. Original RPCBF cost2.83933,16 transfers.

Optional reference-style reconsideration is now implemented (default remains
paper latch).18 component/regression tests pass. Current benchmark session49830
runs scripts/benchmark_safe_mpd_jax.py; results docs/safe_mpd_study/jax_results_02,
log results/safe_mpd_jax_02.log. It is the only heavy worker. Predeclared batching
criterion is in jax_protocol_02.json. Wait for completion before further missions.
Do not restart the old coordinator. The next bounded original-task mission set
will compare reconsidering compact and full-control shields, a32-proposal
full-control screen, and a current-belief compact RPCBF control. Preserve all
failures and label any gain as developmental, not held-out validation.

JAX batching screen is COMPLETE: all three chunk settings produced exactly the
same selected vector in each formulation at the fixed initial state. Fastest
median optimizer calls: full RPCBF chunk8,0.327s; reconsidering endpoint Safe-MPD
chunk8,0.093s; reconsidering compact shield chunk1,0.742s. These exclude cache,
verification and real updates and are NOT full-cycle mission latency.

ACTIVE session12348 now runs mission_protocol_02.json, coordinator
scripts/run_safe_mpd_study.py. Status results/safe_mpd_mission_02_status.json.
Five120s jobs: compact-current RPCBF, compact anticipated-learning reconsidering
shield, full-control RPCBF chunk8, reconsidering Safe-MPD8 proposals, reconsidering
Safe-MPD32 proposals. Latest source/configs/mission drivers frozen by protocol.
No source changes or duplicate workers until complete. No old stopped study or
automation resumed. Source snapshot61f8ade preserves all first-screen behavior.

### Safe-MPD comparison completed; no active worker

All11 declared120-second missions, the4 short pilots, and the9-setting blocked
JAX batch screen are complete. Current source freezes are lifted. Do not restart
sessions58813/12348/49830 or old stopped studies. Automated research remains
paused.34 final component/regression tests pass, Ruff and diff checks pass.
See docs/safe_mpd_comparison.md for full interpretation and reproducible commands
in the JSON protocols; all-run numeric/2D evidence is docs/safe_mpd_study/all_results.

Final conclusion: retain RPCBF default. Full reference-style sampled-wind Safe-MPD
achieves median0.377s / p950.600s full planning cycles for10s trajectories, but
cost3.91687 and3 A/B transfers, versus compact current-belief RPCBF2.95491 and14
transfers, and original proposed2.83933 and16 transfers.32 proposals did not help:
cost3.94494,3 transfers, median0.541s. These full Safe-MPD runs had no contacts.
Direct shield replacements inside the anticipated-learning compact planner
contacted: both paper latch and reference-style reconsideration; preserve failures.
The unknown-wind terminal speed test is empirical, not an invariant-set proof.

The matched full-control RPCBF cost is3.58436 (chunk2) or3.84002 (chunk8), with
median0.827/0.800s. Same-initial-state batching equality does NOT guarantee the
same long mission: at period1 identical inputs give4.17e-7 control-vector change;
by period3 the selected-vector difference is8.78e-4 after causal feedback/POAM
updates. Both results are retained in reproducibility_checks.json. No seed sweeps
or general success claim were made. Original120s arrays (states,controls,clarity,
POAM mean/std) reproduce exactly. Original scenarios/demos/3D are unchanged.

No benchmark code imports from reference repositories, and no dependencies were
installed. Reference tracked files are untouched (an existing untracked .DS_Store
was left alone). CPU is the measured supported backend; earlier Metal limitations
remain. Safe-MPD replay data is not mislabeled as RPCBF: existing exporter rejects
new shield formats; comparisons are supplied as independently audited2D plots.
