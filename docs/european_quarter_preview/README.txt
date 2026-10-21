CANAL QUARTER — MBD TRAJECTORY OPTIMIZATION

The earlier route-restricted recordings are archived and are not results of
the requested MBD trajectory optimizer. Their old method notes are retained in
README_ARCHIVED_v5.txt for reproducibility.

The current implementation is documented in MBD_CONFORMANCE.md.
Runner: scripts/run_full_control_quarter.py
Optimizer: src/voi_diffusion_ipp/quarter/mbd_core.py
Decision and prediction: src/voi_diffusion_ipp/quarter/continuous.py
Nonlinear execution: src/voi_diffusion_ipp/quarter/execution.py

A 30 s horizon has 600 independent acceleration parameters (dt=0.1 s).
RRT-Connect generates geometric initial paths, converted to control sequences.
MBD refines all time intervals using the paper's Section 4.3 / Eq. (13) guidance.
No fixed route library, dwell schedule or temporal basis truncation is used.
Raw diffusion samples, weights, field costs, score estimates, reverse states,
source hashes and measured computation times are retained with each experiment.
Only complete physical mission recordings qualify for mission comparisons.
Planning-only checks and brief execution checks are development evidence.

The viewer renders actual nonlinear quadrotor states. The square panels show
spatial wind belief and clarity deficit with fixed color scales. Camera follow
controls and thick flight trails are available when a recording is loaded.
Follow proposed / Follow blind use a low-elevation perspective chase camera
behind measured motion. The camera shortens its distance at building corners;
it never switches to a top-down follow view. Above is a separate manual view.
The optional "Show planned motion" overlay displays the logged prediction for
the selected method at the most recent replan. It is separate from the recorded
flight paths and uses the learned mean wind (zero wind for the blind planners).
The wind field is synthetic; the rendering is not CFD or real-world validation.

RRT-GUIDED DEVELOPMENT EXPERIMENT

Horizontal scale 0.35; inspection-region scale 0.55. Aircraft, flight height,
wind physics and building heights retain their physical dimensions.
Planning horizon 30 s; dt 0.1 s; replan every 5 s; mission duration 120 s.
16 diffusion iterations, 64 samples/iteration, cosine endpoint 0.08,
temperature 0.15; 4 planning fields and 8 independent validation fields.
Planning clearance penalty begins at 0.6 m from a building; the physical
contact check subtracts the 0.275 m aircraft radius. Soft-filter slack weight 50.
These settings apply to all three compared methods.

Both forward and reversed wind are available. The initial planned controls are
identical under wind reversal, before any observations are made. The subsequent
plans differ because the physical flight and resulting observations differ.
The 30 s development runs improve over unfiltered blind planning by 5.96% in
forward wind and 2.35% in reversed wind. The filtered blind baseline is better
than proposed in the forward case. These are mixed results, not a demonstration
of a large or general advantage. Longer-horizon tuning is documented separately
in output/european_quarter/RRT_STUDY.md and rrt_study.json in the source repository.

The performance table uses the time integral of mean positive clarity deficit.
It does not include the optimizer's soft constraint penalties. A failed flight
retains the remaining mission with no sensing. Positive reduction percentages
mean less deficit. Individual-seed results are not a statistical benchmark.

scripts/audit_rrt_quarter.py reconstructs every displayed position and attitude
from the saved 100 Hz nonlinear execution, verifies the selected control vector,
and recomputes the reported mission cost and minimum clearance. The local
data/execution_integrity.json file contains the audit summary when RRT runs
are published. Raw audit arrays remain with the source experiment outputs.

PREVIOUS FIVE-MINUTE EXTENSION (PAUSED)

The matched extension uses a 45 s horizon, 12 diffusion iterations, 64 proposals,
cosine endpoint 0.05, 4 planning fields, 8 validation fields, a 5 s commit interval,
and common base samples for constructing the state-dependent execution filter.
Each method continues its original 120 s physical recording to 300 s. The
continuation restores the unrounded state, clarity, learned belief, random streams
and shifted preceding control sequence. The first 120 s are preserved exactly.
The continuation test compares an interrupted run with an uninterrupted run,
including controls, physical state, observations, posterior and mission cost.

Only completed and audited five-minute comparisons are added to the Experiment
selector. The report separates the first minute from the remaining 240 s.
Source: output/european_quarter/rrt_mbd_common_300 in the repository.

TIMING REVISION AND FRESH FIVE-MINUTE COMPARISONS

The runtime300 experiment restarts every method from the same initial state.
It does not combine a new configuration with an older flight prefix.
Horizon 45 s; planning/control interval 0.2 s; nonlinear physics 0.01 s;
replan every 5 s; six diffusion iterations and 16 proposals per iteration.
There are 225 time intervals and 450 independent acceleration coordinates.
36 RRT initializations, two elementary initializations, the shifted prior
trajectory, 96 sampled proposals and the final denoised vector give 136
optimization evaluations. Independent validation is additional.
All three methods retain the same configuration, geometry and physical wind.
Cosine endpoint 0.05; temperature 0.15; four optimization fields, eight
independent validation fields, and two common construction fields.

The code already used a compiled outer planner, lax.scan and batched rollouts.
The 2D Newton linear solve now uses equivalent fused scalar algebra, checked
against the previous solver and double precision. This alone made only a small
runtime difference. Most savings come from the reduced sample budget and the
coarser planning interval. The eight-iteration soft projection remains an
approximate solve, not a guarantee of optimality or safety.

Fixed-state synchronized CPU timings: 63.60 s median for the earlier 12 x 64,
0.1 s configuration; 4.08 s for the selected 6 x 16, 0.2 s configuration.
Compilation is excluded and reported separately. On the three sampled states,
the independent predicted cost changed by -0.8%, +24.5%, and +0.3%.
These checks show the speed/optimization-quality tradeoff, not a mission
performance guarantee. Actual mission costs and per-call runtimes are displayed
in Performance. Runs are only selectable after completion and audit.
Source: output/european_quarter/rrt_mbd_runtime_300 and runtime_budget_review.

MEASURED RESULTS FOR THE TIMING REVISION

All six fresh 300 s flights completed without detected contact. Costs below
are integrals of mean positive clarity deficit, reconstructed from 100 Hz logs.

Wind        Proposed     Blind planning     Blind planning + filter
Forward     19.1271      22.0200            24.5930
Reversed    22.2418      23.0846            23.8995

Proposed reduces total deficit versus blind planning by 13.14% forward and
3.65% reversed. In forward wind, its recorded westward transfers use sheltered
passage C (9.6–13.7 s), while blind planning returns against the wind through
A (19.4–23.0 s). Proposed spends 64 s inside the eastern inspection region,
versus 37 s for blind planning. This is an observed route/execution mechanism.

The reversed-wind benefit is delayed: proposed is 5.2% worse cumulatively at
120 s, then 1.8% better at 180 s and 3.65% better at 300 s. Over 60–300 s alone,
its deficit is 6.13% lower than blind planning (14.88% lower in forward wind).
These are single-seed development outcomes; this comparison does not isolate
learning from optimism or demonstrate statistical generalization.

Median proposed planning including RRT is 5.12 s forward and 5.18 s reversed,
with compilation excluded. First planning calls including compilation are
21.97 s and 21.82 s. The reversed run's early calls overlapped replay publication;
all measured times retain that overhead. Physics runs outside the planning wall
clock, so these recordings do not establish real-time operation.

RRT initializations can remain the best evaluated choice at this smaller budget.
All acceleration intervals remain free during MBD; no route or trajectory is
substituted for the selected nonlinear execution. The audit records the origin
of every selected control vector, including when an initialization wins.
