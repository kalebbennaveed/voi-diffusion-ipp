# Interactive research demo

Run from the repository with the existing UV environment:

```sh
JAX_PLATFORMS=cpu uv run python -m voi_diffusion_ipp.viz.demo_server --port 8765
```

Open <http://127.0.0.1:8765>. The server binds only to the loopback interface.
The browser frontend has no external dependencies, model downloads, fonts or
CDNs. The numerical planner runs in Python/JAX on the tested CPU backend; it is
not a JavaScript approximation and does not claim browser GPU acceleration.

## Current whole-field views

New recordings use POAM and 475 equal-area clarity cells, with three distributed
monitoring scenarios (vortices, shear, mixed flow). Historical recordings are
explicitly marked **Legacy**. Never interpret their corridor percentages as
POAM/full-field results.

The default comparison includes a field-layer selector (deficit, clarity, target),
paired instantaneous and integrated deficit curves, and a shared animation clock.
Wind panels show each planner's actual full POAM mean with uncertainty or error,
and a separate physical-wind reference hidden from the planners. RMSE and posterior
standard-deviation histories change at the real update boundaries. Denoising
shows both planners' candidate populations, denoised means, and selected paths.
The final table reports task deficit, clearance, collisions and allowed slack;
ROI arrival is no longer a headline or scoring criterion.

## Playback controls

- **Compare & play (default):** an animated comparison starts automatically at
  half speed and loops after a short final-frame hold. Both robots share the same
  clock; their position markers, trails, clarity and cost so far change together.
  Play/Pause, Restart, speed, Loop and a time slider are directly above the maps.
  “Jump to largest cost gap” pauses where accumulated task costs differ most. Positions and
  clarity are visually interpolated between the recorded 0.15-second samples;
  this interpolation never affects the simulation or saved final metrics.
- **Final result:** compare completed, actually executed missions side
  by side. Both approaches learn during execution; only the proposed planner
  anticipates learning while choosing current motion. The table separates initial
  and later monitoring cost, clarity deficit, clearance, collisions and allowed
  barrier slack. Default seed 200 exactly reproduces archived validation states
  and costs. Pair identities stay consistent when either record is selected.
- **Denoising:** inspect either planner from its result card, select a planning
  period and scrub the reverse updates. Gray
  paths are sampled policies, weighted by their actual stochastic objective.
  Purple is the denoised policy mean; teal is the best evaluated policy. At the
  last level, teal and the cost card include the final denoised-policy evaluation.
  Hover over a weight bar to isolate a candidate and inspect its objective,
  importance weight and observation-selected branch.
- **Details:** scrub actual robot motion, clarity, measurements, corrections,
  clearance and allowed barrier slack. Proposed is teal; the comparator is amber.
  The wind posterior changes only at period
  boundaries. The physical-wind layer is explicitly diagnostic and never feeds
  the planner or safety filter.
- **Follow planning → execution:** play all recorded denoising levels for a
  period, execute that period, then proceed to the next plan and updated belief.
  Playback speed is independent of measured solve time. “Show final selected plan”
  jumps to the final evaluated selection; completed mission playback returns to
  the Final result view.
- **Paired comparator:** overlay a run using the same physical wind. The comparison
  uses actual integrated clarity cost. A no-measurement demonstration and losing
  cases are included alongside favorable cases.
- **Run a new experiment:** choose MBD, frozen-future-belief MBD, CEM or MPPI;
  toggle repeated future updates and measurements; change the seed, wind strength,
  observation noise and optimization budget. One job runs at a time. Progress
  remains visible while the page stays interactive. Completed runs are stored
  under `results/demo/live/` and their catalog entries survive server restarts.

The small-screen layout keeps the paired animation and its controls together;
experiment settings move beneath it. Space plays/pauses the active animation when
focus is outside an input control. Denoising starts when its tab is selected and
has its own Play button and level slider above the forecast map. New recordings
return the page to the top and start their comparison replay.

## Scientific provenance

`planning/mbd.py` records the actual proposed vectors, costs, normalized weights,
noisy iterates, score estimates, reverse updates, incumbent vectors and diffusion
schedule. Recording does not add objective calls or change the optimizer update.
Tests reconstruct the costs and reverse updates from this telemetry.

`viz/demo_data.py` separately replays those recorded policies through the same
full causal evaluator to obtain drawable paths. It reconstructs the exact first
outer planning world and random buffers used in that planning call. These extra
visualization evaluations cannot alter optimization or execution. The displayed
paths represent **one planning world**, while candidate weights and objectives
use **all outer worlds**. A path that looks clear in one world may have zero weight
because another world failed. The denoised mean need not be feasible.

The original forecasted physical world is distinct from the actual mission's
physical field. Neither is secretly substituted for the belief in a controller.
Recorded JSON includes full configuration, backend, timing and source hashes;
infeasible objective values are encoded as `null`, never non-standard JSON Infinity.
The executed-cost plots recompute trapezoidal clarity deficit from the saved state
arrays, with an automated check against the Python mission metric.

## Reproduce and bundle a recording

```sh
JAX_PLATFORMS=cpu uv run python -m voi_diffusion_ipp.viz.demo_data \
  --config configs/recourse_dev.yaml --seed 102 --method proposed \
  --output results/demo/reproduced
uv run python scripts/bundle_demo.py \
  results/demo/reproduced/localized_corridor_proposed_102.demo.json \
  --label 'Reproduced proposed · seed 102' --group reproduced-two-stage
```

Generate the comparator with `--method current_belief` and use the same group
label to pair it. Only pair identical configurations and physical seeds. Bundled
data live inside `demo/data/`; full numerical arrays remain in `results/`.

## Reference and limitations

The read-only `mbd-playground` repository informed the separation between
denoising playback and trajectory execution. Its Safe-MPD browser solver and
shield are different algorithms from this repository's Pan MBD and spline RPCBF.
No reference solver, frontend code, 3D asset, external module or dependency was
copied or imported. This implementation uses a restrained two-dimensional
scientific view appropriate to the double-integrator model.

The bundled scenarios are controlled examples with a known spatial wind shape.
They are not a general robot simulator or a proof of safety. Live experiments
are offline computations on the M3; changing controls creates a new experimental
result, not an independently validated method claim. The interface retains losses,
allowed barrier slack and optimizer failures rather than hiding them. A positive
slack is not a collision or an automatic mission failure; see the
[soft-constraint interpretation and research status](soft_constraint_interpretation.md).
