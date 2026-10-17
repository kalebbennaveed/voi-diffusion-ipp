# Safe-MPD investigation (2026-10-17)

The original task/configurations and recordings are preserved. User authorized a
separate Safe-MPD comparison. No scheduled study was resumed.

## Completed result and recommendation

**Keep RPCBF as the default.** A reference-style Safe-MPD wind adaptation reduced
full planning time, but did not improve persistent monitoring in this development
case. The shield-only replacements that retain anticipated learning contacted
obstacles/boundaries. This is evidence about this adaptation and configuration,
not a conclusion that the original Safe-MPD algorithm cannot work.

All 11 missions below are120s, forecast10s and replan2s on the M3 CPU, original
geometry/wind, same seed8101 and noisy historical survey. Cost is the integrated
whole-grid clarity deficit. Timing is median **full cycle**, with first two
cycles/startup separately archived. No source, environment or reward was changed
while a declared batch ran. No fresh-seed statistical validation is claimed.

| Formulation | Cost | Median cycle, s | A/B transfers | Physical contact |
|---|---:|---:|---:|---|
| Original compact anticipated learning + RPCBF |2.83933|1.703|16|No|
| Compact current-belief + RPCBF |2.95491|0.837|14|No|
| Compact learning + paper-latched shield |5.41443*|2.007|0|Yes|
| Compact learning + reference reconsideration |6.16366*|1.023|1*|Yes|
| Full-control RPCBF, batch2 |3.58436|0.827|6|No|
| Full-control RPCBF, batch8 |3.84002|0.800|4|No|
| Full-control shield, untransformed score |4.95139*|1.184|0|Yes|
| Safe-MPD, latched, dense monitor |4.41457|1.764|1|No|
| Safe-MPD, latched, endpoint monitor |4.41457|0.567|1|No|
| Safe-MPD, reconsidering, endpoint monitor |3.91687|0.377|3|No|
| Same,32 proposals per level instead of8 |3.94494|0.541|3|No|

\*Raw post-contact simulation has no impact-response physics. Its full-duration
cost and transfers are **not successful mission performance**. Independent
collision audits use32 integration substeps per executed0.05s control interval.
The plots mark the post-contact continuation with a dashed red line.

The fastest contact-free Safe-MPD adaptation has p95 full-cycle time0.600s and
no steady2s deadline misses. Its cost is32.6% above the compact wind-aware
current-belief baseline and38.0% above the original proposed method. Against the
matched full-control RPCBF baseline, it is roughly2x faster with2–9% higher cost,
depending on RPCBF batching. This distinction prevents attributing representation
or anticipation changes to the safety mechanism alone. Full-control Safe-MPD
does not anticipate learning; all arms still learn real wind with full POAM.

**What the failures diagnose:** the binary backup switch can be much more
restrictive than RPCBF, and a low-speed endpoint under finite wind samples is
not a robust terminal holding set. Compact paper-latched shielding incurred12
failed uncached selected checks; reference reconsideration incurred26. Increasing
Safe-MPD proposals8→32 did not resolve persistent monitoring quality. This is
not evidence that a still larger budget could never help, nor an isolated proof
of a single failure cause. A defensible next design needs a wind-aware recovery
and terminal-set construction, and independent calibration/closed-loop checks,
before claiming that the paper's safety argument carries over.

**JAX verification:** fixed-shape scans, batched field evaluation and candidate
maps, no host callbacks/QPs in the shield, blocked timing and explicit compilation
separation. Same-state batch1/2/8 gave identical selected vectors; fastest
optimizer calls were0.093s for reconsidering Safe-MPD and0.327s for full-control
RPCBF. These are not full-cycle numbers. Full RPCBF missions diverged beginning
at period3 after a batching change, despite initial-state equality; all outcomes
are retained. This exposes sensitivity of the finite-budget closed loop to
floating-point batching and requires multi-seed validation before promotion.

34 component/regression tests pass. The original 120s recording reproduces
**exactly** in position, controls, whole-field clarity, POAM mean and standard
deviation. Dense/endpoint latched Safe-MPD physical arrays also match exactly in
this seed. The reference repositories and original demo recordings were not
modified. No new GPU support is claimed; see [backend evidence](backend_report.md).

Evidence: [all-run results](safe_mpd_study/all_results/results.json),
[cost/runtime comparison](safe_mpd_study/all_results/comparison.png),
[executed 2D trajectories](safe_mpd_study/all_results/trajectories.png),
[batching measurements](safe_mpd_study/jax_results_02/results.json),
[reproducibility checks](safe_mpd_study/reproducibility_checks.json).
The actual-execution shield-active fraction excludes backup controls already
encoded by the offline projection; it must not be interpreted as the total
amount of backup behavior in Safe-MPD.

## Reference audit before implementation

Read-only checkout `safe-mpd`, commit ff8d2b7ae433afcde7a8b203e0fcad47d46f4879.
Paper: Kim et al., *Safe Model Predictive Diffusion with Shielding*,
https://arxiv.org/html/2512.06261v1, Algorithms 2–3, Definition 4.
Independent implementation; no source copied/imported. No LICENSE file was found
in the reference checkout; its dependencies were not installed.

Relevant reference locations:

- `mbd/robots/acc_tt2d.py:shielded_rollout`: prospective step followed by a
  braking rollout, switch to backup if any checked constraint fails.
- `mbd/planners/mbd_planner.py:reverse_once`: replace proposed controls with
  actually applied controls before weighted score averaging; final shield.
- `mbd/utils.py:rollout_us_with_terminal`: state/control/cost scan.

Differences identified, not silently copied:

1. Paper Algorithm 2 latches backup for the remaining horizon. Reference
   `_step_internal` computes `backup_active_next` but returns
   `in_backup_mode=backup_active_prev`, explicitly relaxing this persistence.
   Our shield carries the latch through the full first/future horizon and resets
   it only on a new actual replan. No zero-velocity state reset is allowed.
2. Reference utilities take an extra zero-control step before the rollout. This
   is not a no-op with initial velocity/acceleration. Our scan starts at exactly
   the observed initial state.
3. Reference proposal denominator uses `alphas_bar[i-1]` and clips Gaussian
   samples. We use paper Eq. (10) with `alphas_bar[i]`; full-control actuator
   bounding is an explicit transform AFTER sampling. A transformed-proposal
   score is not claimed to be an exact untransformed Gaussian score.
4. Reference JIT cache keys omit captured environment/configuration values.
   Our factories close over immutable Config and accept beliefs/environment/key
   as dynamic pytrees; no global cross-environment compiled-function reuse.
5. Reference acceleration backup has coarse 1s checking. Here both prospective
   and backup steps use RK4 with four substeps, plus an independent 32-substep
   physical mission audit. Control is held through each interval.
6. The published examples use known deterministic dynamics and an invariant
   stopped set. Unknown spatial acceleration wind does not satisfy that premise.

## What is being compared

`sampled_mpd_shield` accepts a nominal step only if all three independent
belief-derived wind samples pass collision checking and reach speed <=0.2 m/s
after the same 8 x 0.15s backup used by RPCBF. Backup uses the same bounded
belief-compensated braking policy. This low-speed test is an **empirical recovery
condition, not a certified robust invariant set**. The paper's theorem is not
claimed. Shielding has no barrier gradient, spline maximization or QP. It is a
binary switch rather than RPCBF's minimally modifying control. No additional
penalty/reward or hidden-wind input is supplied. Infeasible forecast rejection
and the full selected-forecast check remain enabled.

Two distinct experiments prevent conflating changes:

- **Shield-only swap:** original compact causal learning policy, actual MBD and
  POAM retained, but RPCBF replaced by the sampled shield during prediction AND
  actual execution. Raw policy coordinates still enter MBD's score. This is a
  controlled shield ablation, **not the complete Safe-MPD algorithm**.
- **Full-control Safe-MPD adaptation:** 50 two-axis controls covering 10s;
  project through a posterior-mean shielded rollout, use those effective control
  vectors in the score, evaluate their task costs across the same outer wind
  samples, and shield the final sequence. Projection uses only the earned
  belief, never outer-world controls. Future belief is frozen; real full online
  POAM still learns after each 2s execution. A matching full-control RPCBF arm
  isolates representation changes from comparison with the original method.

The full-control experiment is a wind-aware baseline, not an implementation of
anticipated learning. Transformed hypothetical controls generally cannot be
encoded exactly back into the original 18-parameter causal policy. Averaging
outer-world-specific controls would introduce noncausal information; we do not
do that. Original anticipation runs remain available for context.

## JAX and measurement

Fixed-shape `lax.scan` over time and reverse levels, `vmap` over inner fields,
`lax.map(batch_size=...)` over candidates to bound memory on 16GB M3. No Python
callbacks, host QP, gradients through the shield, Torch or CUDA-specific imports.
Every timed return uses `jax.block_until_ready`. Report startup and first online
POAM compilation separately from steady full cycles, including cache, optimization,
uncached verification and the previous real POAM update. Actual 20Hz execution
step time (filter plus physics, sensing and clarity, not isolated filter latency)
is also reported; this offline simulator is not a real-time flight
stack. CPU is the supported measured backend; no new Metal success is claimed.

Protocols, all failed tests/pilots and mission results are retained under
`docs/safe_mpd_study/` and `results/safe_mpd_*`.

## First 120-second screen

Six complete missions, same original task and development seed8101. Original
RPCBF: cost2.83933,16 transfers,no contact, median full cycle1.703s. Direct
learning-aware shield swap: no transfers, contact109.60s,12 failed selected
checks; its raw cost5.41443 includes post-contact simulation and is not a successful
monitoring cost. Matched full-control RPCBF: cost3.58436,6 transfers,no contact,
0.827s median. Full-control untransformed shield also contacts. Transformed
Safe-MPD: cost4.41457,1 transfer,no contact,1.764s (dense monitor),0.567s
(endpoint monitor). Dense and endpoint versions produced identical physical
trajectories in this seed. This does not establish equivalence in other cases.

The default paper-style latch was tested first. The next predeclared bounded
check adds `shield_latch=False` / `--shield-reconsider` to reproduce the supplied
repository's per-step reconsideration behavior. It retains the empirical
terminal speed check and the independent physical audit. This variant does not
inherit the paper's invariant-set proof. A same-state JAX chunk1/2/8 screen
selects the fastest batching only if its decision vector matches chunk1 within
1e-5. A 32-proposal full-control run tests whether the small8-proposal budget
under-solves the100-dimensional sequence. All results will remain separate.

These experiments are currently assessed with independently audited 2D plots.
The existing interactive RPCBF demos are untouched; their exporter explicitly
rejects new shield formats to avoid labeling them as RPCBF denoising.
