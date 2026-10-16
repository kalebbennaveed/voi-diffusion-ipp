# Persistent whole-field monitoring

The persistent configurations simulate **122.4 seconds**, with 34 replanning
periods of 3.6 seconds. The earlier 18-second full-POAM recordings are integration
checks; they do not establish sustained monitoring or long-term learning value.

## Information dynamics and the objective

All 475 equal-area cells in the 4.8 m by 3.7 m rectangle contribute to the same
unsmoothed monitoring objective: the time integral of
`mean(max(target - clarity, 0))`. There is no ROI-arrival reward, designated final
destination, uncertainty bonus, or reset when a target is reached.

For each cell, the QoI model is an independent continuous-time random walk with
process covariance rate `Qp`. Its conditional variance satisfies
`Pdot = Qp - C(x)^2 P^2 / R`. With `q = 1/(1+P)`, this is exactly the implemented
clarity equation `qdot = C(x)^2/R * (1-q)^2 - Qp*q^2`. The diagonal entries of the
process matrix are stored for **every cell** in `environment.process_noise`.
The quantity-of-interest covariance is diagonal; this model does not claim
spatial QoI cross-correlation. Wind uses a separate, spatially correlated POAM
belief. The QoI covariance dynamics do not require fabricated observed field
values: for this linear Gaussian model, observation locations determine variance.

Actual clarity follows executed positions, after the full spline RPCBF and its
soft-constraint QP. Without sensing, `q(t) = q(0)/(1 + Qp*q(0)*t)`; consequently a
previously satisfactory cell becomes deficient again. The demo draws this
analytic no-sensing reference for the selected cell. Numerical clarity has also
been checked against a separately integrated covariance Riccati equation,
including spatially varying Q.

The persistent cases explicitly start with QoI clarity 0.04 above each cell's
target. This is an assumed initial covariance, not a simulated survey or a
warm-up hidden from the mission cost. Wind starts untrained. All three cases
lose target clarity everywhere during the mission if no observations occur.
Starting satisfied makes the maintenance problem distinct from a first survey.

## Declared scenarios

| Scenario | Geometry and demand | Process covariance rate | Paired seed |
| --- | --- | --- | --- |
| `persistent_archipelago` | Six obstacle islands; seven distributed demand peaks over a positive background | Uniform Qp = 0.035 | 400 |
| `persistent_channels` | Five staggered obstacles; two broad demand bands | Striped Qp, 0.025–0.075 | 401 |
| `persistent_mosaic` | Five scattered obstacles; smooth demand mosaic across the rectangle | Patchy Qp, approximately 0.03–0.09 | 402 |

Each case uses an independent held-out spatial vortex/jet wind field. Both
planners in a pair have the same physical field, initial state, initial clarity,
and observation-noise sequence. **Both assimilate real wind observations.**
The `current_belief` comparator freezes only its hypothetical future belief.

The proposed optimizer is the joint two-stage MBD implementation, with a
3.6-second first stage and one 3.6-second future stage, two shared future branches,
four outer worlds, four inner RPCBF samples, four diffusion levels and twelve
proposals per level. It retains full POAM inference and spline RPCBF. These are
development compute budgets, not a convergence study. A longer executed mission
does not by itself solve limited planning lookahead or Monte Carlo error.
Warm-start destinations account for known no-sensing decay over the lookahead;
both methods use the same rule and are evaluated on the actual clarity dynamics.

## What counts as persistent behavior

- A sensing pass requires `C^2 >= 0.25` for at least two execution samples.
  Gaps shorter than three seconds merge into one pass. Hovering does not create
  a new pass every frame.
- A clarity recovery requires a cell to have met its target, then remain below
  target for at least one second, and subsequently meet its target again.
- The report retains per-cell pass/recovery counts, median revisit interval,
  mean deficit after 30 seconds, all-cell deficit histories, the analytic
  no-sensing cost, collision/slack diagnostics and wind estimation/calibration.

Repeated visits alone are not proof of a good controller. They must accompany
useful clarity recovery and lower sustained deficit. A single robot may be
unable to maintain every target at the chosen sensing range and decay rates;
the remaining deficit is reported, not hidden by excluding cells.

## Reproduction and recovery

From the repository root, the suite entry point is:

```sh
JAX_PLATFORMS=cpu uv run python scripts/run_persistent_suite.py --workers 3
```

The first invocation declares all cases and freezes the numerical source/config
hashes in `docs/persistent_poam_validation/protocol.json`. It writes raw data,
numeric optimizer checkpoints, partial executed trajectories, belief snapshots,
and worker logs beneath `results/persistent_poam/recordings/`. Source/config
changes are rejected on recovery. Re-running the same command skips completed
missions and recovers unfinished ones. Numeric checkpoints use no pickle.

Recovery reuses computed optimizer results, then replays actual dynamics,
observations and POAM updates from the seed to restore state. It does not fit a
replacement wind model or substitute an inexpensive controller. Tests verify
identical recovered execution and full-POAM demo telemetry. Original plan times
are retained; the invocation's total wall time excludes time spent before an
interruption and therefore is not a clean end-to-end timing measurement.

Completed pairs are validated, bundled into the demo, and plotted automatically.
Open `http://127.0.0.1:8765/`, select a **Persistent POAM · 122 s** recording, and
use **Play comparison**, **Final result**, or **Denoising**. Select **Sensing
passes**, **Process noise Qp**, **Clarity**, or **Clarity deficit**; click a cell
to inspect its history. Wind mean, uncertainty, error, and physical truth are
shown below the field plots. The progress panel reports actual finished periods
while offline experiments run; unfinished missions are not presented as final
outcomes.

This is a paired development study with one seed per geometry. Scientific
superiority, successful persistent maintenance, and solver advantage must be
assessed from completed results and broader controls; they do not follow from
the existence of these configurations or animations.
