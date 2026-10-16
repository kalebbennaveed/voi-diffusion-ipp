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
