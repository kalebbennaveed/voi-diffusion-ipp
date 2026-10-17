# Safe-MPD investigation (2026-10-17)

The original task/configurations and recordings are preserved. User authorized a
separate Safe-MPD comparison. No scheduled study was resumed.

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
