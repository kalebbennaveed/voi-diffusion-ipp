# Recourse study: complete evidence

All 37 missions are included. Cost is actual integrated clarity deficit (lower is better).
Development seeds: 20–22; frozen validation seeds: 100–109. These are now used validation data.
Timing includes concurrent host load. No-measurement trajectories and controls match exactly.

| Suite | Method | n | Mean cost | ROI access | Collisions | QP infeasible | Fallbacks | Plan s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| recourse_ablations/task_openloop | proposed | 3 | 4.3685 | 3/3 | 0 | 0.42% | 0 | 4.44 |
| recourse_ablations/threshold_feedback | proposed | 3 | 4.1289 | 2/3 | 0 | 0.00% | 0 | 1.09 |
| recourse_ablations/threshold_openloop | proposed | 3 | 5.0204 | 1/3 | 0 | 0.00% | 0 | 1.14 |
| recourse_no_measurements | current_belief | 1 | 5.5169 | 0/1 | 0 | 0.00% | 0 | 4.50 |
| recourse_no_measurements | proposed | 1 | 5.5169 | 0/1 | 0 | 0.00% | 0 | 4.30 |
| recourse_pilot | current_belief | 3 | 4.8158 | 2/3 | 0 | 0.00% | 0 | 4.21 |
| recourse_pilot | proposed | 3 | 4.4458 | 3/3 | 0 | 0.00% | 0 | 4.25 |
| recourse_validation | current_belief | 10 | 4.5743 | 9/10 | 0 | 0.00% | 0 | 4.23 |
| recourse_validation | proposed | 10 | 3.6169 | 10/10 | 0 | 0.12% | 0 | 4.41 |

## recourse_pilot

Mean paired benefit 0.3700 (7.68% relative mean cost reduction). Paired percentile-bootstrap 95% interval: [-0.1428, 0.7142]. 10,000 resamples, RNG seed 2026; small-sample descriptive uncertainty, not a guarantee.
Wins / ties / losses: 2 / 0 / 1.

| Seed | Proposed | Frozen-belief comparator | Benefit | Initial extra cost | Later benefit |
|---:|---:|---:|---:|---:|---:|
| 20 | 4.9774 | 5.5159 | 0.5385 | 0.0001 | 0.5386 |
| 21 | 4.1782 | 4.8925 | 0.7142 | 0.0979 | 0.8121 |
| 22 | 4.1818 | 4.0390 | -0.1428 | 0.1712 | 0.0284 |

## recourse_validation

Mean paired benefit 0.9574 (20.93% relative mean cost reduction). Paired percentile-bootstrap 95% interval: [0.4434, 1.4560]. 10,000 resamples, RNG seed 2026; small-sample descriptive uncertainty, not a guarantee.
Wins / ties / losses: 8 / 0 / 2.

| Seed | Proposed | Frozen-belief comparator | Benefit | Initial extra cost | Later benefit |
|---:|---:|---:|---:|---:|---:|
| 100 | 3.9147 | 3.7738 | -0.1409 | 0.1202 | -0.0207 |
| 101 | 3.8056 | 5.2923 | 1.4867 | 0.1355 | 1.6222 |
| 102 | 3.2378 | 4.2864 | 1.0486 | -0.0224 | 1.0262 |
| 103 | 3.8229 | 4.6211 | 0.7982 | 0.1106 | 0.9088 |
| 104 | 4.2327 | 3.7127 | -0.5200 | -0.0834 | -0.6035 |
| 105 | 3.3657 | 5.2677 | 1.9020 | 0.1466 | 2.0486 |
| 106 | 3.7643 | 4.5163 | 0.7520 | 0.1465 | 0.8985 |
| 107 | 3.3401 | 4.4319 | 1.0918 | 0.0543 | 1.1462 |
| 108 | 3.1305 | 5.5077 | 2.3772 | 0.0009 | 2.3780 |
| 109 | 3.5551 | 4.3331 | 0.7781 | 0.0857 | 0.8637 |
