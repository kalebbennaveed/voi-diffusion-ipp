# Urban quadrotor experiment

This local demo implements the user-supplied four-panel layout at 40 × 28 m.
Passage A is centred at y=9 m, B at y=25 m, and C at y=1.5 m. The main building
occupies x=14–28 m, y=12–22 m, height 8.5 m. The nominal flight altitude is 3 m.
The physical quad has motor centres at (±0.08, ±0.08) m; the imported visual is
scaled to match those centres. Its rotor envelope is approximately 0.34 m in
maximum diameter. Collision checking conservatively uses radius 0.20 m.

## Sources and licenses

- Quad visual: `autonomous-drone-racing-local/web_demo/models.js`, repository
  commit `c642a9b588f5ebe7545d49956baeba3fe5bc30e8` (MIT, copy in licenses/drone.txt).
- Village rendering vocabulary: `fastpathplanning/examples/uav_visualization.py`,
  `Village`, commit `b2a720710bde19e56d98778f13d6cfe75666c432` (Apache 2.0).
  Building boxes, roof caps, framed windows and cuboid vegetation are implemented
  in Three.js. The deterministic layout follows the user's diagram, not the
  reference notebook's random layout. No fastpathplanning optimizer is substituted.
- Rigid body and geometric controller: `lab-experiment-sim-starter/src/lab_sim`,
  commit `dc78ff9a14736644c068013e0a9166e3d9acd2d7` (MIT, package LICENSE).
  Local extensions: quintic reference derivatives, nonzero jerk/snap flatness,
  compiled integration of the same equations, spatial force evaluated at each
  RK4 stage, and a shared inspection acceleration envelope.
- Three.js: existing vendored distribution; license included.

No reference repository imports or edits are required.

## Physical and planning model

The MBD solver is the existing `planning/mbd.py` implementation. Its optimization
variables describe inspection target, A/B/C passage, cruise speed, and dwell.
Connected geometric routes initialize this restricted policy class. The MBD
score and reverse updates are unchanged. Current budgets: 8 geometric starts,
4 reverse levels × 8 proposals, two outer coherent fields and two independent
inner controller fields. These are development budgets, not a convergence study.

Each route is evaluated with planar double-integrator prediction. A sampled
policy-value filter (the existing spline/RPCBF implementation) produces the
mean-field reference. Predictions include the translation component of the
same geometric tracker's position and velocity feedback; they do not predict
attitude or motor states. The reference is committed for two seconds. The full
quadrotor then tracks it at 100 Hz, using a C2 quintic interpolation and flatness
attitude/body-rate/angular-acceleration feedforward. The nonlinear plant includes
3D position, velocity, rotation, body rates and four motor speeds: 22 stored
states, with motor lag, aerodynamic drag and rotor angular momentum. The
rotation matrix is projected to SO(3) after each RK4 step.

The common operational command envelope is ±2.6 m/s² on each horizontal axis,
after subtracting known drag compensation. It applies to all three real trackers
and all three predictions. It is an inspection-flight restriction, not the
maximum thrust the motors could theoretically produce. Vertical control remains
active. This restriction is required for a meaningful comparison to a bounded
planar planner; it is not an additional obstacle or PLCBF controller.

Wind is a smooth, fixed spatial *equivalent acceleration load*, in m/s², applied
as mass times the load in the nonlinear plant. It is not CFD and is not labelled
as air velocity. Cases 1a/1b reverse the complete vector field. Case 2 has exactly
zero load in B, alternating crosswind jets in A, and adverse load in C. No calm
straight strip is inserted through A. Case 3 has eastward loading in all three
passages and a larger right inspection region. Drawn routes in the supplied
image are hypotheses, not hard-coded execution outcomes.

The final case 3 load is 2.15 m/s² eastward, within the shared command envelope.
An earlier 3.1 m/s² stress test exceeded that envelope and collided; shortening
the commitment to 0.8 s did not resolve it. Both development records are retained
under `output/urban_quad/unrecoverable_3p1_load` and `rapid_commitment`, outside
the final comparison. In the final case, proposed and disturbance-aware planning
cross to the right region: the requested turn-back/stay-left behaviour has not
been demonstrated. No route was forced to obtain that behaviour.

## Learning and comparisons

POAM maintains a full sparse predictive distribution, including the residual
covariance away from inducing points. This experiment freezes its kernel and
28 inducing locations. A padded 1 m field table accelerates the existing sampled
policy-value calculation. Tests check interpolation error and covariance.
Measurements occur every two seconds at actual quadrotor positions. They are
constructed from the simulated acceleration sensor residual after removing
known thrust, gravity and drag, with identical noise seeds across methods.
The physical field never enters the controller/filter belief directly.

- Proposed: samples coherent fields and anticipates chronological posterior
  updates, continuing from reached robot and clarity states.
- Disturbance-aware: the planning belief is held fixed over the prediction.
- Disturbance-blind: the planner predicts a zero disturbance field.

All three real trackers receive the same estimator interface and real sensor
updates. All have the same nonlinear plant and command envelope. Learning is
valued only by its effect on total monitoring cost; no entropy or positive-only
learning bonus is added. Future feedback uses observations only after they are
obtained. The policy does not contain an unrestricted future replanning tree.

Monitoring cost is the time integral of mean clarity deficit over the inspection
cells, using actual quad positions. The same cell definition is used by all
methods in an environment. The larger right region in case 3 has more cells and
therefore more weight in this uniform-cell metric. Collision failures stop
physical execution and charge the remaining horizon with no further sensing;
they are explicitly labelled in the replay.

## Reproduction

From `voi-diffusion-ipp`:

```sh
.venv/bin/python -m pytest tests/urban_quad -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_urban_quad.py --seconds 120
.venv/bin/python scripts/build_urban_quad_demo.py
.venv/bin/python scripts/validate_urban_quad_results.py
.venv/bin/python -m http.server 8890 --bind 127.0.0.1 --directory output/all_methods_demo
```

Open `http://127.0.0.1:8890/urban-quad/`. The builder excludes incomplete,
nonfinite-planning and superseded model revisions. It serves actual recorded
position, attitude, motors, clarity, estimator snapshots and planning references.
Playback never invents trajectories or derives performance from screen geometry.
Current result JSON stores configuration, vehicle parameters, wall time, planning
costs and traces, physical tracking errors, minimum clearance and completion.

Early development records with mismatched translation/control authority are
kept separately and are excluded from the final comparison. CPU execution is
used. No real-time control, formal safety, CFD validity, or statistical superiority
claim is made from these individual development missions.

The twelve 120 s final missions completed without collision. Proposed cost
relative to blind is 3.9% higher in forward wind, 6.7% lower in reversed wind,
13.2% lower in crosswind, and 5.3% lower in the crossing environment. In crosswind,
disturbance-aware planning is best (34.1% below blind). These mixed outcomes do
not establish a consistent anticipated-learning advantage. `validation.json`
records file hashes, independent cost integration, motor/attitude checks and
uncertainty reduction for every final run.

Verification on 2026-10-20: 28 tests passed on CPU across `tests/urban_quad`,
MBD, RPCBF, clarity, hypothetical POAM and learning-value tests. The compiled
nonlinear executor agrees with the NumPy plant/controller with and without the
operational acceleration envelope. Browser checks covered all four environments,
three-method visibility, play/pause, speed selection, scrubbing, uncertainty,
enlarged comparison and follow camera; the final browser log had no errors or
warnings. A screenshot is saved at `output/urban_quad/qa/three-method-comparison.png`.
