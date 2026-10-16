# High-Q persistent development screen

Results are pending. `paired_outcomes.json` lists every declared pair and its
completion state; absence of a pair is not an experimental result. The original
frozen protocol is in the internal `tmp/learning-regimes` worktree. Its exact
hash and source hashes are retained in `archival_protocol.json`.

Publish and validate new completed pairs from the main checkout:

```sh
uv run --no-sync python scripts/report_regime_screen.py
```

The reporter checks the source and configuration of each completed mission,
rejects partial trajectories, recomputes clarity costs, verifies POAM telemetry
and initial-condition/physical-wind pairing, and publishes both arms together.
Figures include the seed so multiple runs cannot overwrite one another.
The four declared pairs are development examples; neither statistical superiority
nor equivalence can be concluded from this screen. Fresh-seed validation follows
only after the development design is fixed.
