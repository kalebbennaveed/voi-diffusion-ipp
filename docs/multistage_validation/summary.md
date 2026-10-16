# Complete repeated-update study

49 missions retained. Six initial development runs are superseded because their comparator warm start anticipated future updates. They are excluded from all scientific comparisons below.

| Suite | Method | n | Mean actual cost | ROI access | Collisions | Infeasible QP steps | Optimizer fallbacks | Plan s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| multistage_corrected | current_belief | 3 | 4.8158 | 2/3 | 0 | 0 | 0 | 4.56 |
| multistage_corrected | proposed | 3 | 3.8363 | 3/3 | 0 | 0 | 0 | 5.20 |
| multistage_dev | current_belief | 3 | 3.6887 | 3/3 | 0 | 0 | 0 | 4.32 |
| multistage_dev | proposed | 3 | 3.8363 | 3/3 | 0 | 0 | 0 | 4.23 |
| multistage_failure_controls | proposed | 1 | 3.2073 | 1/1 | 0 | 0 | 0 | 7.46 |
| multistage_validation/frozen_belief | current_belief | 10 | 4.3803 | 8/10 | 0 | 1 | 1 | 4.72 |
| multistage_validation/repeated_updates | proposed | 10 | 3.1507 | 10/10 | 0 | 9 | 0 | 4.53 |
| multistage_validation/two_stage | proposed | 10 | 3.3722 | 10/10 | 0 | 2 | 0 | 5.33 |
| strict_qp_dev | proposed | 4 | 4.0345 | 4/4 | 0 | 0 | 0 | 5.37 |
| strict_qp_failure_followup | proposed | 2 | 3.1630 | 2/2 | 0 | 8 | 0 | 5.94 |

## Repeated updates versus frozen_belief

Relative mean reduction: 28.07%. Mean paired benefit: 1.2297. Paired bootstrap 95% interval: [0.6867589604455406, 1.795724552238298]. Wins / ties / losses: 10 / 0 / 0.

| Seed | Repeated updates | Comparator | Benefit | Initial extra cost | Later benefit |
|---:|---:|---:|---:|---:|---:|
| 200 | 3.3777 | 5.4224 | 2.0447 | 0.0145 | 2.0592 |
| 201 | 3.3430 | 4.3032 | 0.9602 | 0.0255 | 0.9857 |
| 202 | 2.8996 | 3.2718 | 0.3723 | 0.1245 | 0.4967 |
| 203 | 2.7481 | 5.5177 | 2.7695 | 0.0501 | 2.8197 |
| 204 | 3.3393 | 3.5030 | 0.1637 | 0.0138 | 0.1776 |
| 205 | 2.9682 | 5.1066 | 2.1384 | 0.1107 | 2.2491 |
| 206 | 2.9429 | 3.3391 | 0.3962 | 0.0106 | 0.4067 |
| 207 | 3.1581 | 3.3108 | 0.1527 | 0.1515 | 0.3042 |
| 208 | 3.2548 | 4.5087 | 1.2539 | 0.0803 | 1.3342 |
| 209 | 3.4748 | 5.5201 | 2.0453 | 0.0914 | 2.1367 |

## Repeated updates versus two_stage

Relative mean reduction: 6.57%. Mean paired benefit: 0.2215. Paired bootstrap 95% interval: [0.044397484478540715, 0.46427928246092043]. Wins / ties / losses: 8 / 0 / 2.

| Seed | Repeated updates | Comparator | Benefit | Initial extra cost | Later benefit |
|---:|---:|---:|---:|---:|---:|
| 200 | 3.3777 | 3.6951 | 0.3173 | -0.0251 | 0.2922 |
| 201 | 3.3430 | 3.3916 | 0.0486 | -0.0184 | 0.0302 |
| 202 | 2.8996 | 2.9160 | 0.0164 | -0.0133 | 0.0031 |
| 203 | 2.7481 | 2.6202 | -0.1280 | -0.0627 | -0.1906 |
| 204 | 3.3393 | 3.6574 | 0.3181 | -0.1634 | 0.1547 |
| 205 | 2.9682 | 4.1250 | 1.1568 | 0.1108 | 1.2675 |
| 206 | 2.9429 | 2.8637 | -0.0792 | -0.0457 | -0.1249 |
| 207 | 3.1581 | 3.2274 | 0.0693 | 0.1515 | 0.2208 |
| 208 | 3.2548 | 3.5500 | 0.2952 | -0.0300 | 0.2652 |
| 209 | 3.4748 | 3.6753 | 0.2005 | 0.0915 | 0.2919 |

Positive benefit means lower actual monitoring cost with repeated updates. Intervals use 10,000 paired percentile-bootstrap resamples, RNG seed 2026. They describe a small sample in a selected scenario family, not broad generalization or a safety guarantee.
