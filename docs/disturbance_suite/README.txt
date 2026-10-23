DISTURBANCE-AWARE MONITORING: CHECKPOINT BEFORE SENSING REVISION

This checkpoint records ten paired 300 s nonlinear quadrotor simulations, seed 9101. The original sensing cone has a 45 degree half-angle: radius = altitude, approximately 3.2 m. Both planners use this same sensing model and the same physical execution filter.

View output/disturbance_suite/final_report through an HTTP server. Its results.json contains exact configuration, measurement protocol, source hashes, audit outcomes, development selection and limitations. Playback data are archived as compressed JSON. Full 100 Hz execution/search logs and historical development cohorts remain in local output/disturbance_suite (several GB); they are deliberately not added to Git. Source snapshots and raw-run paths are listed in the report.

Run fresh paired simulations using scripts/run_disturbance_suite.py. Final configuration: --seconds 300 --seed 9101 --horizon 12 --dt .2 --iterations 3 --modes 13 --proposals 26 --worlds 2 --validation-worlds 4 --temperature .06 --prior-std 1.8 --survey-count 64 --inducing-count 64 --short-length .6 --long-length 1.8 --prediction-filter hard --prediction-substeps 2 --validation-batch-size 27 --execution-margin .7 --planning-clearance .6 --output FRESH_DIRECTORY. For distributed use execution-margin .9 and planning-clearance .9 for both planners. Select cases with --cases.

These are development-selected synthetic fields, one paired seed, with 64 shared noisy pre-mission measurements. No hardware, CFD, held-out robustness or real-time deadline claim. Planning pauses the simulated mission clock.
