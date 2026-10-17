# Wind-route development study

The original two-region scenario and all earlier recordings are retained. This
study changes and verifies the **2D** demo only. Research automations remain paused.
All comparisons below use 120-second missions, 10-second forecasts, 2-second
planning intervals, full online POAM, actual MBD, and sampled spline RPCBF.
These are development results for seed 8101, not statistical validation.

## Executed results

| Physical field / common initial information | Proposed cost | Current-belief cost | Proposed reduction | Physical contact |
|---|---:|---:|---:|---|
| Original passage wind; outer survey | 2.83933 | 2.95491 | 3.91% | Neither |
| Exact reversed wind; outer survey | 3.02046 | 3.05978 | 1.28% | Neither |
| Crosswind in passage; outer survey | 3.04879 | 2.85073 | −6.95% raw cost | Comparator at 75.90 s |
| Wind away from main route; outer survey | 2.64237 | 2.86059 | 7.63% | Neither |
| Same off-route wind; outer + main-route survey | 2.43553 | 2.67378 | 8.91% | Neither |

Cost is the integrated mean positive clarity deficit across all 1,353 cells.
The crosswind comparator penetrates an obstacle by 0.06261 m in the independent
32-substep audit. Its trajectory after contact is integrated without impact
response: the raw full-duration cost is **not** a successful mission outcome.
The demo discloses this and pauses at the first contact. It can be resumed to
inspect the retained diagnostic continuation. This comparator is wind aware;
it is not the separate ablation that removes wind from safety.

The wind-blind nominal planner plus wind-aware execution safety has reversed-field
cost 3.82642; proposed cost is 21.06% lower than that comparator. This contrast
combines wind-aware planning, nominal wind compensation and anticipation. The
wind-blind comparator uses wind only in its execution safety filter. It does not
establish the isolated value of anticipating learning.

In the original proposed run, one of sixteen transfers uses the middle, toward B.
Under reversed wind, one of sixteen uses the middle, toward A. The other fifteen
transfers remain bypasses. Thus the direction flips, but systematic exploitation
of the favorable passage is not established. The restricted, privileged
rest-to-rest route-family calculation also reverses its favored direction:
3.855 s through the favorable middle versus an 8.600 s bypass in the opposite
direction. These family timings are not globally optimal trajectories and are
not directly comparable to the robot's region-boundary-to-boundary transit times.

The anticipated low-value control did **not** establish near parity. The extra
21 noisy main-route survey observations reduce initial passage mean sigma from
0.489 to 0.183 m/s², but substantial uncertainty remains around the route. A calm
physical passage is not equivalent to a belief that the passage is known. Both
negative-control attempts and their outcomes remain available; no claim of
negligible learning value is made from these results.

Reversed proposed planning cycles: median 1.59 s, p95 1.67 s, no 2 s misses.
Crosswind p95 is 2.35 s and the unsurveyed off-route p95 is 2.83 s; the surveyed
case has median 1.47 s and p95 1.52 s. These measurements include prior real POAM
work, input preparation and the full selected-plan check. Startup is reported
separately in every audit. They were measured sequentially on the same M3 under
ongoing desktop use, not a controlled thermal benchmark; a universal two-second
deadline has not been demonstrated.

## Mechanism and failure diagnosis

`reversed_causal_01` and `offroute_causal_01` reconstruct exactly earned POAM
histories at 0, 30 and 60 s. Four independent banks of four worlds evaluate the
same 40 original candidates plus 14 geometry-only round-trip initializations.
First-stage states agree between the two objectives. Five of six states have
the same best averaged candidate with and without anticipated future learning.
The remaining state does not show a robust advantage on held-out banks: two
held-bank differences are zero, and two favor the frozen-objective choice.
One reversed-field held-bank comparison is infeasible and retained as such.
No added round-trip seed wins either averaged objective or the privileged oracle
at these states. These particular seeds are therefore **not promoted**.
The executed single-seed gains cannot yet be attributed to a robust route-choice
VoI mechanism rather than sampling and closed-loop optimizer effects.

The crosswind contact begins 1.9 seconds after the filter's last belief update.
The nearby grid estimate understates the downward acceleration. A same-state
74–80 s diagnostic in `observation_latency_02` reproduces the original trajectory
exactly. Providing earned 5 Hz observations between the existing 2 s full POAM
adaptation boundaries changes minimum clearance from −0.06261 to +0.04241 m,
with no contact in that local replay. Both nominal wind compensation and RPCBF
use the updated belief; only the archived nominal reference policies are held
fixed. This is not an isolated RPCBF effect or a replanned mission. Independent
physical reintegration error is below 6.1e-8. The initial unjitted-snapshot
reproduction failure and its driver remain in `observation_latency_01`; using the
original jitted snapshot path fixes it without changing the tolerance.

Separating fast posterior inference from slower hyperparameter adaptation is
consistent with POAM's incremental variational updates and variational-EM
organization. The paper also discusses delaying hyperparameter adaptation when
early data is not representative. Neither observation establishes safety or
validates a new controller here. See [POAM, Sections IV-C and IV-D](https://arxiv.org/html/2406.03669v1).

A remaining policy-representation concern is the compact future feedback: it
moves its position knots by at most 0.6 m on each coordinate. The normal-component
knot displacement is bounded by approximately 0.823 m in this geometry, while
the common bypass initializations use a 1.5 m offset. This is a bound on knot
motion, not on a spline's complete trajectory or the safety filter's deflection.
It motivates checking a policy with separately parameterized middle/bypass
continuations, selected only from earned posterior information, rather than
increasing an arbitrary information reward or adding more of the failed seeds.

## Further bounded tests

The optional `budgeted_policy: branches` representation optimizes two independent
future routes and a four-coefficient affine gate on the passage posterior's mean
and standard deviation. Its first-stage controls remain common across worlds;
only earned hypothetical observations can change the gate. A switch changes the
reference curve, never the physical state or clarity. Full POAM, actual MBD and
sampled spline RPCBF are retained. The original compact policy remains default.
Twenty-three component tests passed, including causal observation timing.

On the reversed 120 s development mission, this representation costs **3.08719
versus 2.89869**, or **6.50% regret** from anticipation, with no dense contacts.
It is not promoted. Its paired arms share initializations; comparison with the
older compact experiment additionally changes the parameter dimension and seed
representation, so it is not an isolated parameter-count comparison. The
independent four-bank check again finds the same best averaged candidate at all
three tested states. One held-bank selection at 30 s favors anticipation, while
the other eleven comparisons are equal. All numerical outputs are retained in
`branch_causal_01` and `branch_policy_results_01`.

An additional open-passage control keeps the large regions, rapid Q decay,
off-route wind and common noisy route survey, but changes the obstacle radius
to 0.25 m and normal offset to 0.95 m. The middle half-gap is 0.70 m; the
historical outer survey still has at least 0.298 m geometric clearance. The
development pair costs **2.04599 versus 1.99606**, or **2.50% regret**. Proposed
uses the middle on all 18 transfers; the comparator uses it on 18 of 19. Neither
contacts an obstacle. Proposed cycle median/p95 are 1.48/1.51 s, with no steady
2 s misses. This is a candidate low-value case, not a statistical conclusion.

The first open-passage audit retained the original fixed route thresholds; it
misclassified one middle crossing as "other" and used the original narrow
passage for local wind summaries. The audit now derives these thresholds from
the archived geometry. Its previous output and driver are preserved in
`open_passage_classification_correction`. Physical clearance, contacts, executed
trajectories and all-grid cost were unaffected. The 2D replay was refreshed.

`low_value_confirmation_protocol_01.json` declares eight fresh paired seeds
18101–18108, excluding the development seed, before execution. The primary
criterion is a one-sided 95% paired-t upper regret bound below 5%, with all pairs
complete and no dense contacts; secondary equivalence requires the 90% interval
within ±5%. This tests randomness in one fixed physical environment and prior.
It does not establish equivalence over arbitrary wind fields or environments.

The privileged fully informed reversed-wind planner costs **2.71153**, with no
contacts. It knows the complete wind for planning, compensation and safety, so
it is a finite-budget information reference, not a deployable method, optimal
cost bound, or demonstration of what POAM will learn.

The ten-second horizon is shorter than the approximately 12.45 s favorable-
crossing/opposing-bypass round trip in the restricted rest-to-rest family. A
prospective 10 versus 14 s sensitivity at earned states in `horizon_diagnosis_01`
finds a round-trip seed becomes best at 60 s with the longer forecast—but for
**both** anticipated and frozen-belief objectives. No robust isolated learning
advantage emerges. Actual MBD with 21 common initializations takes about 1.80 s
for 10 s lookahead and 2.54 s for 14 s at the tested warm state; startup is about
19 s. These are saved-state measurements, not mission latency guarantees. The
longer horizon and extra seeds remain diagnostic options, not promoted defaults.

## Reproduction and interactive records

The first screen is declared in `protocol_01.json`; subsequent historical-survey
and causal checks have separate protocols. `results_01.json` contains dense
collision/clarity audits and data hashes. `overview_01.png` displays the paths
and cost curves, with post-contact cost dashed. `route_physics_01.json` retains
all restricted-family calculations and the temporal-resolution refinement.

The 2D homepage now has separate buttons for every completed wind case, physical
wind or POAM-estimate arrows, direction-specific route counts, genuine paired
MBD denoising, whole-field clarity and the original scenario. All new replay
means and standard deviations are checked against their original POAM snapshots.
The raw numeric archives remain under `results/budgeted_missions/`; published
replays remain under `results/demo/live/`. No 3D files were modified.
