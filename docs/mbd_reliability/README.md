# Original MBD–RPCBF reliability investigation

This continues the user-approved plan without replacing the original scenarios,
realization-based planner, covariance experiment, or archived demos. Scheduled
research remains paused. One foreground heavy JAX worker runs at a time.

## Mathematical audit and tested repair

The decision variable is the nominal feedback-policy vector theta. MBD scores
its RPCBF-filtered monitoring cost, including coherent physical wind, earned
measurements, future POAM updates and continuous robot/clarity states. It does
not average scenario-specific executed controls or require an invertible filter.
The objective averages costs before exponentiation.

The deployed experimental target is more precisely

`p(theta | S) proportional to 1_box(theta) 1_valid,S(theta) exp(-Jhat_S(theta)/T)`.

The box is [-6,6]^18 in new tests. `valid,S` excludes nonfinite and sampled-contact
forecasts when rejection is enabled. It is a proper density when its finite-cost
valid subset has positive volume. The sampled admissibility set itself depends
on the outer and inner banks. Increasing world count therefore changes both
cost-estimation accuracy and which sampled contacts are discovered. Report both;
do not interpret it as only a reduction in Monte Carlo variance. This is not an
empirical-safety guarantee. The historical zero-bound default remains available
for archival reproduction, without a normalizable unbounded-target claim.

The all-invalid diffusion branch previously returned zero weights but still
rescaled the noisy iterate, and reported effective sample size near 1e30. The
new explicit branch holds that iterate and reports zero effective samples. The
incumbent objective remains infinity if no valid candidate exists. A failing
regression was recorded before the repair; 19 MBD, causal-budgeted and runtime
cache tests pass afterward. This is a failure-handling correction, not evidence
of improved monitoring. Finite warm-started reverse diffusion remains an
approximate optimizer; no finite-compute convergence guarantee is claimed.

## Completed numerical audit

Nine event-selected states cover departure and first outbound/return midplane
crossings in original, reversed and open-passage low-value development missions.
Each checks the stored selected policy and three distinct next-ranked finite
proposals. Two separately labeled historical numerical failures are stress tests.
Selection uses recorded geometry and events, not favorable method outcomes.

For both anticipated and frozen forecasts, compare the original 129-wide cache,
the same cache with exact outer physical fields, a 257-wide cache, and uncached
evaluation. The physical-field override is only a simulator input: it never
supplies the controller or inner RPCBF bank.

At all nine event-selected states, the cached and uncached evaluations select
the same shortlist winner for both belief modes. Yet anticipated forecast paths
differ by up to 0.433m across the four policies and two sampled worlds. Replacing
only the physical interpolation does not remove that discrepancy. The finer
cache reduces the maximum to 0.386m, with nonuniform improvement. Maximum
anticipated mean-cost errors are 0.002144 (129) and 0.000698 (257). These small
shortlists do not establish global ranking stability.

The previous worst selected-path stress case reproduces about 0.959m divergence,
reduced to 0.027m with the finer cache. In the separate rejected-forecast case,
the coarse cache predicts +0.0291m clearance while uncached evaluation finds
approximately -0.00073m. The 257-wide cache still misses the sign (+0.00143m).
The next-ranked proposal passes the uncached check (+0.0397m). Keep final
uncached verification; merely increasing cache width is not a sufficient repair.
Median warm cache-build time here is 0.077s (129) versus 0.294s (257), excluding
the rollout, replanning and real POAM work.

Two diagnostic failures are retained. The first driver did not decode archived
JSON null costs as invalid floats and failed before any evaluator ran. The second
completed ten states, then detected up to 0.000232 state difference in its batched,
reduced-grid uncached evaluation of a historical rejection. The original full-grid
single-policy evaluation reproduces that archive exactly (zero difference).
Verification now uses the original evaluation shape with the original tolerance;
batching drift is reported separately. No archived values or tolerance were edited.
All three executed driver versions are preserved with matching protocol hashes.

Evidence: `numerical_summary.json`, `numerical_protocol_01/02/03.json`, and numeric
arrays under `numerical_02/` and `numerical_03/`. The last directory completes the
one failed state; it does not replace the earlier ten outcomes.

## Declared search and sampling experiment

`search_protocol_03.json` freezes 84 optimizer calls at the first outbound
midplane state in each of the three environments, paired anticipation/frozen
objectives and two optimizer seeds. It compares 4x8, 8x16, 16x32; 2/4/8 nested
outer worlds; and equal-37-evaluation one versus three chains. Initial/final
diffusion scales, temperature and shared starts are held fixed. Counts of outer
rollouts and wall-clock cost are reported as well as objective-call counts.

Each chosen policy is evaluated under 16 independent posterior physical worlds,
first holding the inner bank fixed and then using independent inner banks with
the same outer worlds/noise. Evaluation uses original uncached fields and causal
hypothetical POAM updates for every selected policy: frozen describes planning,
not disabling real learning. These test banks do not rerank or choose the method.
Contact incidences remain separate from raw post-contact costs. They assess the
posterior model, not calibration to independent physical environments.

Results are pending. Do not launch held-out mission sweeps or claim a search
improvement until this diagnosis is complete. Extra chains, fresh-bank reranking,
Gaussian reference priors and covariance-only anticipation are not promoted.
