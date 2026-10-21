# RRT-guided MBD development evidence

Development tuning on seed 8101. These are not independent confirmation trials. Incomplete mission costs are excluded from percentages. Failed missions include the unsensed remainder. Planning computation is outside the physical simulation clock. No real-time or safety guarantee is claimed.

RRT supplies geometry-based initial trajectories. MBD optimizes every nominal acceleration interval; the nonlinear quadrotor executes the selected trajectory. All methods use the same RRT procedure and budget within each trial.

| Trial | Wind | Method | Status | Physical time (s) | Mission deficit integral | Reduction vs blind | Min. clearance (m) | Median planning (s) |
|---|---|---|---|---:|---:|---:|---:|---:|
| 300 s mission; 45 s horizon; 0.2 s planning step; 6 × 16 MBD | forward | blind | completed | 300.00 | 22.0200 | +0.00% | 0.404 | 0.89 |
| 300 s mission; 45 s horizon; 0.2 s planning step; 6 × 16 MBD | forward | blind_filter | completed | 300.00 | 24.5930 | -11.68% | 0.189 | 0.89 |
| 300 s mission; 45 s horizon; 0.2 s planning step; 6 × 16 MBD | forward | proposed | completed | 300.00 | 19.1271 | +13.14% | 0.335 | 5.13 |
| 300 s mission; 45 s horizon; 0.2 s planning step; 6 × 16 MBD | reverse | blind | completed | 300.00 | 23.0846 | +0.00% | 0.124 | 0.93 |
| 300 s mission; 45 s horizon; 0.2 s planning step; 6 × 16 MBD | reverse | blind_filter | completed | 300.00 | 23.8995 | -3.53% | 0.108 | 0.95 |
| 300 s mission; 45 s horizon; 0.2 s planning step; 6 × 16 MBD | reverse | proposed | completed | 300.00 | 22.2418 | +3.65% | 0.112 | 5.19 |
| 300 s mission; 45 s horizon; common construction draws | forward | blind | completed | 300.00 | 21.3783 | +0.00% | 0.252 | 11.82 |
| 300 s mission; 45 s horizon; common construction draws | forward | blind_filter | completed | 300.00 | 25.6429 | -19.95% | 0.176 | 12.45 |
| 300 s mission; 45 s horizon; common construction draws | forward | proposed | running | 230.00 | — | — | 0.277 | 97.58 |
| 30 s; smaller clearance penalty margin | forward | blind | failed | 9.07 | 41.7739 | +0.00% | -0.001 | — |
| 30 s; smaller clearance penalty margin | forward | blind_filter | failed | 38.65 | 25.9374 | +37.91% | -0.001 | 3.46 |
| 30 s; smaller clearance penalty margin | forward | proposed | interrupted | 80.00 | — | — | 0.029 | 50.58 |
| 30 s; 0.6 m clearance penalty margin | forward | blind | completed | 120.00 | 10.4404 | +0.00% | 0.169 | 5.45 |
| 30 s; 0.6 m clearance penalty margin | forward | blind_filter | completed | 120.00 | 9.0838 | +12.99% | 0.004 | 4.69 |
| 30 s; 0.6 m clearance penalty margin | forward | proposed | completed | 120.00 | 9.8184 | +5.96% | 0.108 | 51.15 |
| 30 s; 0.6 m clearance penalty margin | reverse | blind | completed | 120.00 | 12.2260 | +0.00% | 0.042 | 6.62 |
| 30 s; 0.6 m clearance penalty margin | reverse | blind_filter | completed | 120.00 | 12.5053 | -2.28% | 0.166 | 9.25 |
| 30 s; 0.6 m clearance penalty margin | reverse | proposed | completed | 120.00 | 11.9384 | +2.35% | 0.162 | 50.47 |
| 45 s; up to three RRT site visits | forward | blind | completed | 120.00 | 10.0629 | +0.00% | 0.328 | 6.82 |
| 45 s; up to three RRT site visits | forward | blind_filter | failed | 33.52 | 26.3018 | -161.38% | -0.000 | 6.95 |
| 45 s; up to three RRT site visits | forward | proposed | completed | 120.00 | 8.6369 | +14.17% | 0.140 | 109.76 |
| 45 s; common construction draws | forward | blind | completed | 120.00 | 10.0629 | +0.00% | 0.328 | 9.91 |
| 45 s; common construction draws | forward | blind_filter | completed | 120.00 | 13.5882 | -35.03% | 0.176 | 11.50 |
| 45 s; common construction draws | forward | proposed | completed | 120.00 | 8.6401 | +14.14% | 0.314 | 115.55 |
| 45 s; softer filter, smaller MBD budget | forward | blind | completed | 120.00 | 10.0299 | +0.00% | 0.362 | 5.33 |
| 45 s; softer filter, smaller MBD budget | forward | blind_filter | completed | 120.00 | 10.1862 | -1.56% | 0.301 | 5.56 |
| 45 s; softer filter, smaller MBD budget | forward | proposed | completed | 120.00 | 9.3145 | +7.13% | 0.332 | 53.06 |
| 30 s; stronger 3.4 m/s wind | forward | blind | completed | 120.00 | 12.5318 | +0.00% | 0.257 | 8.92 |
| 30 s; stronger 3.4 m/s wind | forward | blind_filter | failed | 69.57 | 26.1167 | -108.40% | -0.000 | 10.11 |
| 30 s; stronger 3.4 m/s wind | forward | proposed | running | 85.00 | — | — | 0.056 | 83.28 |

## Reproduction

The accompanying `rrt_study.json` gives every source path and complete configuration. Each source recording contains source-code checksums, per-call RRT paths, sampled control vectors, MBD traces and links to the 100 Hz execution records.

Run `scripts/audit_rrt_quarter.py --results <trial directory>` to reconstruct the replay and reported cost. Run `scripts/report_european_quarter.py --results <trial directory>` for measured passage crossings and sensing time.
