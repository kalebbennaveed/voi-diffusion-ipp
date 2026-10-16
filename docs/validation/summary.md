# Archived measured results

120 missions. Costs are actual integrated clarity deficits; lower is better. ± is sample SD, not a confidence interval.

| Suite | Scenario | Method | n | Cost mean ± SD | ROI access | Collisions | QP infeasible | Optimizer fallbacks | Mean plan s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ablations/fast_spline | corridor | proposed | 3 | 4.1493 ± 0.0085 | 1/3 | 0/3 | 0.00% | 1 | 2.91 |
| ablations/independent_cells | corridor | proposed | 3 | 4.1535 ± 0.0051 | 0/3 | 0/3 | 1.67% | 0 | 2.39 |
| ablations/no_measurements | corridor | proposed | 3 | 4.1607 ± 0.0064 | 0/3 | 1/3 | 13.33% | 6 | 2.80 |
| ablations/shared_future | corridor | proposed | 3 | 4.0026 ± 0.2798 | 0/3 | 0/3 | 0.00% | 1 | 2.58 |
| ablations/short_probe | corridor | proposed | 3 | 4.1025 ± 0.0918 | 1/3 | 0/3 | 0.00% | 0 | 2.36 |
| ablations/terminal | corridor | proposed | 3 | 4.1649 ± 0.0125 | 0/3 | 0/3 | 0.00% | 1 | 2.75 |
| additional | corridor | current_belief | 3 | 4.1592 ± 0.0043 | 0/3 | 0/3 | 0.00% | 1 | 2.28 |
| additional | corridor | draft_selection | 3 | 4.1552 ± 0.0032 | 0/3 | 0/3 | 0.00% | 0 | 1.24 |
| additional | corridor | fixed | 3 | 4.1586 ± 0.0041 | 0/3 | 0/3 | 0.00% | 1 | 1.86 |
| additional | irrelevant_uncertainty | current_belief | 3 | 1.1047 ± 0.1346 | 3/3 | 0/3 | 0.00% | 0 | 2.91 |
| additional | irrelevant_uncertainty | draft_selection | 3 | 2.0563 ± 0.0295 | 2/3 | 0/3 | 0.00% | 0 | 1.08 |
| additional | irrelevant_uncertainty | fixed | 3 | 1.4255 ± 0.3468 | 3/3 | 0/3 | 0.00% | 0 | 1.93 |
| additional | no_worthwhile_learning | current_belief | 3 | 4.1602 ± 0.0056 | 0/3 | 1/3 | 13.89% | 6 | 2.86 |
| additional | no_worthwhile_learning | draft_selection | 3 | 4.1558 ± 0.0031 | 0/3 | 1/3 | 25.00% | 1 | 0.93 |
| additional | no_worthwhile_learning | fixed | 3 | 3.9365 ± 0.3873 | 1/3 | 1/3 | 13.89% | 7 | 1.91 |
| evaluation | corridor | cem | 3 | 4.1250 ± 0.0622 | 1/3 | 0/3 | 0.00% | 1 | 3.20 |
| evaluation | corridor | heuristic | 3 | 4.1552 ± 0.0032 | 0/3 | 0/3 | 0.00% | 0 | 1.05 |
| evaluation | corridor | mppi | 3 | 4.1501 ± 0.0086 | 0/3 | 0/3 | 0.00% | 1 | 2.17 |
| evaluation | corridor | oracle | 3 | 3.9292 ± 0.4077 | 1/3 | 0/3 | 0.00% | 0 | 2.14 |
| evaluation | corridor | passive | 3 | 4.1552 ± 0.0032 | 0/3 | 0/3 | 0.00% | 0 | 0.51 |
| evaluation | corridor | proposed | 3 | 4.1490 ± 0.0081 | 1/3 | 0/3 | 0.00% | 1 | 1.91 |
| evaluation | corridor | weighted_ig | 3 | 4.1608 ± 0.0070 | 0/3 | 0/3 | 0.00% | 1 | 2.41 |
| evaluation | irrelevant_uncertainty | cem | 3 | 1.1030 ± 0.1505 | 3/3 | 0/3 | 0.00% | 0 | 2.73 |
| evaluation | irrelevant_uncertainty | heuristic | 3 | 2.0446 ± 0.0490 | 2/3 | 0/3 | 0.00% | 0 | 1.06 |
| evaluation | irrelevant_uncertainty | mppi | 3 | 1.3040 ± 0.4885 | 3/3 | 0/3 | 0.00% | 0 | 2.55 |
| evaluation | irrelevant_uncertainty | oracle | 3 | 0.9698 ± 0.0396 | 3/3 | 0/3 | 0.00% | 0 | 2.53 |
| evaluation | irrelevant_uncertainty | passive | 3 | 2.0563 ± 0.0295 | 2/3 | 0/3 | 0.00% | 0 | 0.54 |
| evaluation | irrelevant_uncertainty | proposed | 3 | 1.1003 ± 0.1178 | 3/3 | 0/3 | 0.00% | 0 | 1.95 |
| evaluation | irrelevant_uncertainty | weighted_ig | 3 | 1.1180 ± 0.1125 | 3/3 | 0/3 | 0.00% | 0 | 2.11 |
| evaluation | no_worthwhile_learning | cem | 3 | 4.1623 ± 0.0111 | 0/3 | 1/3 | 24.44% | 4 | 2.10 |
| evaluation | no_worthwhile_learning | heuristic | 3 | 4.1553 ± 0.0088 | 0/3 | 1/3 | 25.00% | 2 | 1.31 |
| evaluation | no_worthwhile_learning | mppi | 3 | 4.1590 ± 0.0115 | 0/3 | 1/3 | 24.44% | 4 | 1.91 |
| evaluation | no_worthwhile_learning | oracle | 3 | 3.9292 ± 0.4077 | 1/3 | 0/3 | 0.00% | 0 | 3.09 |
| evaluation | no_worthwhile_learning | passive | 3 | 4.1558 ± 0.0031 | 0/3 | 1/3 | 25.00% | 2 | 0.78 |
| evaluation | no_worthwhile_learning | proposed | 3 | 4.1602 ± 0.0056 | 0/3 | 1/3 | 13.89% | 6 | 2.78 |
| evaluation | no_worthwhile_learning | weighted_ig | 3 | 4.1602 ± 0.0056 | 0/3 | 2/3 | 13.89% | 6 | 2.76 |
| extended | corridor | current_belief | 1 | 4.1450 | 0/1 | 0/1 | 0.00% | 0 | 80.04 |
| extended | corridor | proposed | 1 | 3.4288 | 1/1 | 0/1 | 0.00% | 0 | 58.78 |
| extended_no_measurements | corridor | proposed | 1 | 4.1450 | 0/1 | 0/1 | 20.00% | 0 | 79.63 |
| long_mission | corridor | current_belief | 3 | 5.3869 ± 1.2441 | 3/3 | 0/3 | 1.39% | 2 | 1.87 |
| long_mission | corridor | proposed | 3 | 5.5680 ± 1.0068 | 3/3 | 0/3 | 3.61% | 1 | 1.85 |
| long_no_measurements | corridor | proposed | 3 | 7.4404 ± 1.0340 | 1/3 | 2/3 | 17.78% | 12 | 1.90 |
