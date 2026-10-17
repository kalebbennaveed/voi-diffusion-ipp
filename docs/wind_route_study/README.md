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
cost 3.82642 (21.06% above proposed when expressed as proposed's reduction from
that comparator). This contrast combines wind-aware planning and anticipation;
it does not establish the isolated value of anticipating learning.

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
