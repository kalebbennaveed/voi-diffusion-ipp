CANAL QUARTER — SPATIAL MAPS AND RECORDED QUADROTOR FLIGHTS

Local demo: http://127.0.0.1:8890/european-quarter/
Four-site environment: add ?layout=multi
Reversed wind: add ?case=reverse

The upper and lower panels are square maps of this environment. The selectors
can place wind belief, clarity deficit, clarity or wind estimation error in
either position. Wind belief offers posterior standard deviation, posterior mean
speed, and physical wind speed. Each map has a fixed color scale and an Enlarge button. Displaying the
physical field is a diagnostic; it is not supplied to the planner.

The east inspection site is 18 m north of the west site. Passage A is 5 m wide;
passage B is 6 m wide. The central building block separates them by 25.5 m.
Inspection sites occupy 9 x 10 m. The visual/collision envelope is 0.55 m across and nominal flight
height is 3.2 m. The lab model retains its own exported motor-arm parameters. The extended environment has northwest and southeast sites.

PLAYBACK
Play/pause, restart, scrub and speed controls operate on recorded simulation
states. Proposed and disturbance-blind flights share the 3D environment. Select
which method supplies the spatial maps. Measured performance includes every
completed comparison available in the selected environment. A positive percentage
means a reduction of accumulated clarity deficit relative to the blind planner.
Only equal-duration completed missions receive a percentage comparison.
Passage crossing times are measured between x = -22 m and x = +22 m; they
exclude travel from an inspection square to the passage entrance. The full
mission score includes all travel, inspection, and remaining-time penalties.

MODEL AND METHOD
The planning model is planar translation with a double-integrator control input
and known quadratic relative-air-velocity drag. Planning integration is 0.1 s.
The common inspection command limit is 2.2 m/s^2 per horizontal axis. This
controller limit is separate from maximum motor thrust. It is an explicit shared
inspection-flight acceleration limit, chosen for this narrow-street experiment;
it is not presented as the airframe's ultimate wind resistance. An independent
uniform-wind calibration measured 3.00 m/s in calm air and 0.768 m/s against
3.1 m/s wind at a requested 3.0 m/s ground speed. The slowdown follows the
quadratic drag and shared command limit, rather than a playback speed rule.
Sensing assumes a nadir-stabilized camera: a finite 90-degree camera field of view (45-degree half-angle)
with Gaussian quality and building visibility masks. Clarity integrates the
observation/process-noise model along actual flight positions; it is not an
image-reconstruction experiment.

Execution uses the existing full nonlinear quadrotor, motors, differential-
flatness references and geometric feedback at 0.01 s. Wind is an unknown static
spatial air-velocity field. Its synthetic definition is an experiment, not CFD
or a measurement of a real neighborhood. Vehicle parameters are known.

The posterior remains fixed inside each planning call. Coherent posterior fields
are drawn separately for execution-filter construction, monitoring costs and
final selection validation. The proposed objective is the lower-tail mean of
predicted monitoring cost, with finite penalties for constraint violations.
The filter uses the spline maximum over design-policy rollouts with penalized
slack. This is constraint-aware planning, not a safety certificate. The wind table
has a constant extension outside its computational domain for speculative
soft-penalty and backup-policy evaluations; physical execution still terminates
at contact or the environment boundary. The interior field and its derivatives
are unchanged. If the final shortlist has no finite validation score, locally
inspecting policies are evaluated on that same validation bank before declaring
a planning failure.

MBD uses a cosine schedule over a warm-start interval [0, 0.75], with the exact
cumulative alphas saved in every recording. The proposal mixes the ordinary
Gaussian with archive-centered Gaussians and applies the full density ratio.
Every method starts from the same geometric route library. The optimized policy
parameters select inspection sites and passage classes and adjust speed, dwell
and crossing offsets. This is a restricted policy parameterization, not arbitrary
control-sequence optimization. The selected route controller retains its waypoint cursor and dwell timer across replans; choosing a different route starts its controller from the current state. A receding-horizon regression checks completion of all four inspections under a fixed cyclic policy.

Only actual simulated observations update POAM between planning calls. The
posterior uses sufficient-statistic updates with fixed kernel/inducing locations
and retains conditional residual covariance. It is not a prerecorded uncertainty
reduction. The old anticipated-observation method is not used in these runs.

COMPARISONS
Proposed: optimistic planning, archive mixture, actual learning and runtime filter.
Disturbance-blind planning: zero wind inside planning; actual learned-wind
compensation remains in the common low-level tracker, with no execution filter.
Blind planning + filter: blind planning, same actual learning and runtime filter.
Expected-cost planning: mean scenario cost, same actual learning and filter.
Optional ablations disable learning, remove the archive mixture, or fade optimism.

All primary comparisons share geometry, physical field, vehicle, initial state,
initial clarity, sensing model and observation-noise seed. The monitoring score
is the integral of mean positive clarity deficit. A detected collision or loss
of flight terminates sensing; the remaining mission deficit is retained.
Planning latency is measured outside the simulation clock. Recorded wall times
include competition with other local jobs and are not a controlled runtime
benchmark. These recordings do
not establish real-time operation. Individual seed runs are illustrative; they
do not establish a statistically general advantage of optimism.

REPRODUCTION
Source: src/voi_diffusion_ipp/quarter/
Runner: scripts/run_european_quarter.py
Data publisher: scripts/build_european_quarter_data.py
Viewer builder: scripts/build_european_quarter_preview.py
Numerical/regression checks: tests/quarter/test_method.py
Calibration: scripts/check_quarter_headwind.py
Fixed-state parameter study: scripts/screen_european_parameters.py
Selected experiment recordings: output/european_quarter/validated_v4/
Source checksums and common run conditions: output/european_quarter/validated_v4/source_manifest.json
Shorter-horizon exploratory records: output/european_quarter/pilot/
Implementation snapshot: output/european_quarter/source_snapshot/v4/

Run configuration, random seed, actual states, clarity, posterior maps,
observations, policy parameters, optimizer diagnostics and planning times are
saved in each JSON recording. Interrupted or superseded development runs are
retained separately and are not included as completed comparisons.

MATERIALS
Original quadrotor visual is retained. European architecture is procedurally
constructed with CC0 Poly Haven textures and daylight. See credits.html and
materials.json. No changes to the public website are made by this local build.
