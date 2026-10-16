# Persistent study status

Started 2026-10-16 at 17:05 America/Detroit. **Results are still being computed.**
The authoritative live status is `results/persistent_poam/recordings/status.json`;
this page is a launch record, not a claim that the runs have completed.

At 17:10, all three initial workers had completed their first actual 3.6-second
segment and saved two POAM belief snapshots. The island pair increased clarity
in 57 sensed cells while 418 cells decayed; the channel run increased clarity in
67 cells while 408 decayed. This checks physical execution and field evolution,
not successful long-term maintenance. The first optimizations took about 171–172
seconds each with three concurrent CPU workers, plus recording and update time.

All six 122.4-second comparisons are frozen in `protocol.json`. Three workers
run concurrently; proposed/current-belief archipelago and proposed channels run
first, followed by the remaining channel and two mosaic runs. The coordinator
log is `results/persistent_poam/suite.log`. Per-worker logs and recoverable
numeric plan checkpoints are in the recordings directory.

The complete persistent backend suite passed **46 tests**, and the additional
whole-field covariance check plus demo request validation passed after merging
the implementation. The browser model's **10 tests** passed with all six new
short recordings bundled. Ruff and JavaScript syntax checks passed. The local
demo server was restarted with the persistent-scenario API and progress endpoint.

## Continue safely

Keep `src/` and the declared YAML configurations unchanged while workers run.
The exact source/config manifest is checked before recovery. Do not launch a
second coordinator while the existing workers are active. The current command is:

```sh
JAX_PLATFORMS=cpu uv run python -u scripts/run_persistent_suite.py --workers 3
```

Once an interrupted coordinator and its workers are no longer running, this same
command skips completed missions and recovers unfinished ones. Recovery preserves
computed plans and replays physical dynamics and POAM updates; it never bypasses
the estimator or replaces RPCBF. A process-bound `caffeinate -i` assertion was
started for the coordinator; no persistent power settings were changed.

The coordinator automatically runs the following validator/publisher after
completions:

```sh
uv run python scripts/report_field_poam.py \
  --source results/persistent_poam/recordings \
  --output docs/persistent_poam_validation \
  --label-prefix 'Persistent POAM · 122 s' --paired-only
```

The demo must label these outcomes separately from the completed 18-second
integration comparisons. Never fill missing long-run outcomes using repeated
short trajectories or synthetic playback. Validate the recorded final costs,
revisit episodes, clarity target recoveries and sustained deficit after 30 s.
Inspect all-cell evolution, wind calibration and collisions/slack, including
regressions. Examine the small MBD improvement and Monte Carlo/lookahead budgets
if the learning benefit is inconsistent; the short-study findings do not prove
general superiority.

A thread follow-up named **Validate persistent POAM monitoring** checks this
study every 30 minutes and continues analysis/refinement. It should report
meaningful findings or failures, stay quiet for unchanged progress, and stop once
this persistent study and necessary bounded refinement have been delivered.
