# Urban quadrotor airflow experiment

This revision changes the physical model and environment; it is not a reskin of
`docs/urban_quad_demo`. The earlier demo and unsuccessful development records
are retained. Use the recorded model revision in each JSON to distinguish runs.

## Environment and physical scale

The flight domain is 40 × 28 m. The main building is 14 m high and occupies
x=14–28 m, y=10.2–23 m. The lower building is 10 m high and occupies
x=14–28 m, y=2.6–7.8 m. A is 2.4 m wide (centre y=9), C is 2.6 m wide
(centre y=1.3), and B is 5 m wide (centre y=25). Nominal flight altitude is 3 m,
so the quad cannot fly over these buildings. A/B centre lines are 16 m apart.
The quad's motor centres are at (±0.08, ±0.08) m; the visual is scaled to those
coordinates. Physical collision checks include a conservative 0.20 m radius.

The industrial rendering adds concrete, metal roofs, windows, service doors,
rooftop equipment, drainage joints and fencing outside the flight domain. The
recorded building geometry, not decorative assets, defines collision checks.
Orbit the view or use the actual-attitude inset to inspect vehicle motion.

## Actual wind, finite control authority and tracking

Wind is an air-velocity vector in m/s. The full nonlinear plant uses
`F_drag = c ||w-v|| (w-v)`, with c=0.1 and mass 0.680 kg, from the lab model.
The controller compensates using the learned posterior mean, not actual wind.
All methods bound the **total commanded horizontal thrust per mass** to ±2.6
m/s² on each axis. Compensation consumes this same budget. This is an inspection
operating restriction, not the motor's maximum theoretical thrust capability.

The supplied lab rigid-body equations and SO(3) geometric tracker run at 100 Hz,
including rotation, body rates, motor lag, aerodynamic drag and rotor momentum.
The planner still generates planar double-integrator references, lifted to 3 m
and interpolated with C2 quintics for flatness feedforward. Its rollout predicts
translation with the same wind drag and position/velocity feedback; it does not
optimize the attitude and motor states. No added PLCBF controller is used.

A controlled nonlinear-flight check (`scripts/check_urban_airspeed.py`) requests
2 m/s with known constant wind. Measured steady speeds are approximately 2.00
m/s in calm air, 2.00 m/s downwind, and 0.405 m/s upwind against 3.8 m/s air flow.
The latter agrees with `sqrt(m*2.6/c)-3.8`. This validates slowdown and positive
upwind progress in straight flight, not obstacle avoidance or mission success.

## Planning, belief and monitoring

A 120 s prediction is replanned every 2 s over a 360 s mission in cases 1a,
1b and 2. Case 3 uses a 240 s prediction: its known-wind upwind crossing takes
about 186 s, so a 120 s horizon would omit much of the return cost. The horizon
is identical across all three methods within each case. The original MBD
score/reverse update is retained: four levels, eight proposals, two outer fields,
two inner control-filter fields. Twenty geometric initializations cover separate
outward/return passages and local inspection. Policy parameters select the next
region, outward and return passage, speed and dwell time. Predictions repeat the
inspection cycle; the actual dwell time already spent at a region is carried
into subsequent planning calls. All methods slow their references before
geometric turns using the same distance-based braking profile. This is a
restricted policy class and a small optimization budget, not a global optimum.

The sampled policy-value construction is adapted to the known, velocity-dependent
air-drag law. Its design policy brakes under the posterior mean. Inner samples
come from the current wind posterior, never the physical field. Spline extrema
and automatic derivatives are retained. Finite prediction and nonlinear tracking
mismatch do not constitute a formal safety guarantee; failures remain visible.

**Initial knowledge is explicit:** ten noisy preflight anemometer observations
on B and C (five x-locations each), with noise standard deviation 0.25 m/s, are
supplied identically to all methods. A has no preflight observations. Raw survey
coordinates and measurements are exported in every run. No-survey diagnostic
runs are retained separately: they often stayed in one region because uncertain
unvisited routes had higher predicted cost. These are different experimental
initial conditions and must not be pooled.

POAM uses 28 fixed inducing locations, the full sparse predictive covariance,
and fixed kernel settings. Every two seconds, the actual simulated acceleration
residual is converted to an air-velocity observation by inverting the known drag
law; independent observation noise is added. All three estimators update:

- Proposed: predictions anticipate chronological observations and belief updates.
- Disturbance-aware: predictions hold the current posterior fixed.
- Disturbance-blind: predictions use zero wind; the actual tracker still uses
  the shared online estimator, as in the other methods.

Learning is credited only through predicted monitoring cost. There is no extra
information bonus or hand-coded route preference. Predicted state and clarity
continue after every hypothetical observation; they never restart.

Clarity obeys `dq/dt = C²/R (1-q)² - Q q²`, with `C=exp(-r²/(2 sigma²))`,
sigma=2 m, R=0.3, Q=0.025 and target 0.75. The physical run integrates it using
actual positions at 100 Hz. Cost is the time integral of mean positive clarity
deficit over inspection cells. The displayed footprint represents normalized
clarity sensing gain `C²`; the two contours are 50% and 10% of its peak. This is
a fixed-altitude downward sensing model, not a detailed optical camera model.

## Cases and honest comparisons

1a: 3.8 m/s eastward wind in A; B/C calm.
1b: exact reversal of the complete vector field in A.
2: adverse headwind and lateral wind in A/C; B exactly calm. The field is smooth,
spatial and synthetic, not a CFD prediction of building wakes.
3: strong eastward wind in all three passages, with a larger right inspection
region. Crossing, returning or staying local must emerge from the objective.
No prescribed turn-back trajectory is used.

The three methods use identical physical conditions, initial sensor data and
sensor-noise seeds. Recorded failures stop physical execution and charge the
remaining mission with no further sensing. Do not interpret a failed run's last
physical position as successful hovering. One seed is a development comparison,
not statistical evidence of superiority. A separate diagnostic varies kernel length scales on recorded observations. It
does not change these runs. The fixed sparse model can extrapolate inaccurately
in unvisited areas; reduced posterior variance alone is not proof of accuracy.
The full-map wind-error plot remains visible for that reason.

The original unconstrained-turn runs
that clipped a boundary are kept under `output/urban_quad_airflow/late_braking_v7`. Stronger lateral-wind
trials are kept under `strong_lateral_v8`; the current crosswind has a 0.7 m/s
lateral amplitude and retains 3.65/3.55 m/s axial headwinds in A/C.

## Replay and reproduction

```sh
.venv/bin/python -m pytest tests/urban_quad tests/test_mbd.py tests/test_clarity.py -q
.venv/bin/python scripts/check_urban_airspeed.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_urban_quad_airflow.py
.venv/bin/python scripts/validate_urban_airflow_results.py
.venv/bin/python scripts/build_urban_quad_airflow_demo.py
.venv/bin/python -m http.server 8890 --bind 127.0.0.1 --directory output/all_methods_demo
```

Open `http://127.0.0.1:8890/urban-quad-v2/`. It displays actual position, attitude,
motor speeds, reference, sensing footprint and time-synchronized diagnostics:
clarity deficit per region, cumulative cost, wind uncertainty, wind-estimation
error, ground speed, true wind and posterior maps. Wind tracers are advected
through the selected true/estimated velocity field; they are visualization,
not particles in the aerodynamic solver. Colour scales are fixed across methods.

`validation.json` independently integrates exported clarity cost, checks finite
states, quaternion norms, motor bounds, completion, survey equality, reversal
and calm B, and reports physical passage crossings and directional speeds.

## Recorded results (20 October 2026)

These are 360 s, seed-8101 development runs with the declared shared B/C preflight survey. They are not cold-start or multi-seed results. All twelve physical executions completed without collision. Lower monitoring cost is better; percentages use each case’s disturbance-blind cost as denominator.

| Environment | Proposed cost | Aware cost | Blind cost | Proposed cost reduction | Aware cost reduction |
|---|---:|---:|---:|---:|---:|
| 1a · forward A | 47.01 | 46.31 | 83.20 | 43.5% | 44.3% |
| 1b · reversed A | 47.11 | 46.58 | 86.63 | 45.6% | 46.2% |
| 2 · adverse A/C, calm B | 47.22 | 47.06 | 76.20 | 38.0% | 38.2% |
| 3 · strong eastward wind | 92.04 | 91.98 | 71.97 | -27.9% | -27.8% |

Negative reductions mean higher cost than disturbance-blind. Proposed does not outperform disturbance-aware in any of these four records. These experiments demonstrate a wind-awareness benefit in the first three environments, not an additional anticipated-learning benefit.

In 1a, the informed methods use A rightward and C leftward (17 crossings each), while blind uses A both ways (9 crossings). In 1b, the informed circulation reverses. In 2, the informed methods choose calm B outbound and the wind-assisted passages for return; blind persists through slow adverse wind. The time spent travelling delays revisits and increases clarity deficit.

**Case 3 remains an unsuccessful behavior/performance test.** The informed methods cross to the larger right region and stay near its centre, rather than turn back and stay left. Blind also services the right side but moves more, achieving lower monitoring cost. The right region is calm and more valuable; the supplied eastward field assists the outward crossing. This setup does not establish that staying left is optimal. The limited route policy and belief extrapolation also require further study. No conclusion of superiority is made for this case.

Wind uncertainty decreases, but full-map wind-estimation error increases in these records. The fixed, sparse kernel can fit flown locations and extrapolate poorly elsewhere. The replay includes the error curves alongside uncertainty; lower variance is not proof of an accurate field.

Verification: 22 tests passed (`tests/urban_quad`, `tests/test_mbd.py`, `tests/test_clarity.py`). The exported-run audit independently recomputes cost and checks finite states, quaternion norms, motor bounds, actual wind at recorded positions, identical initial conditions/configuration, and mission completion. Six known-wind passage checks reach the opposite side, including the strongest upwind case at about 0.10 m/s. Browser controls, all four environments, three simultaneous views and plots are checked separately.

Compact evidence: [mission audit](results/validation.json) and [physical passage checks](results/passage_validation.json). The audit includes SHA-256 hashes of the raw run files and the model/planner source files at audit time. Full replay records remain in `output/urban_quad_airflow`; the static builder copies them into the local demo.

## Provenance

- Quad visual: autonomous-drone-racing-local, commit
  c642a9b588f5ebe7545d49956baeba3fe5bc30e8 (MIT, licenses/drone.txt).
- Earlier village rendering reference: fastpathplanning, commit
  b2a720710bde19e56d98778f13d6cfe75666c432 (Apache 2.0). This revision uses new
  industrial materials and the user's deterministic layout; no optimizer swap.
- Lab plant/controller: lab-experiment-sim-starter, commit
  dc78ff9a14736644c068013e0a9166e3d9acd2d7 (MIT, quadrotor package LICENSE).
- Three.js: vendored distribution; license retained.
