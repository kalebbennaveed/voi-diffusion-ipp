# Whole-field clarity and POAM

The new default demo/CLI profile is `configs/field_poam.yaml`. Historical ROI
experiments are retained as legacy results and do not validate this model.

## Monitoring task

There are 25 × 19 equal-area cells covering the entire 4.8 × 3.7 m rectangle.
Each has a clarity state, initial value, target, and the supplied clarity ODE:

`q_dot[p] = C[p](x)^2/R * (1-q[p])^2 - Q[p]*q[p]^2`.

Here `q=1/(1+P)` for the cell posterior variance `P`; its Kalman covariance
equation is `P_dot=Q-C(x)^2*P^2/R`. An independent SciPy covariance integration
checks the implemented clarity evolution and integrated task cost on all 475 cells.

The Gaussian sensing footprint uses executed robot positions. RK4 propagates
all 475 states, including cells outside high-demand patches. The task objective
is the time integral of `mean(max(target-q, 0))`, with equal weight per cell.
Seven distributed patches increase the target above a positive background
requirement. Flight obstacles do not remove ground monitoring cells: these
remain observable through the sensing footprint. There is no ROI arrival reward,
fixed destination, or privileged passage in this profile. Warm starts choose
locations from current distributed deficit and sensing benefit; MBD optimizes
all policy knots under the same full-field objective.

The [persistent-monitoring extension](persistent_monitoring.md) runs 122.4-second
missions, supplies spatial diagonal process-noise matrices, and adds explicit
revisit/recovery diagnostics. Its warm starts forecast known no-sensing decay
over the planning window, so initially satisfied cells can still motivate
maintenance. Both compared planners use that same warm-start rule.

This is the uncertainty dynamics of the supplied independent-cell linear Gaussian
QoI model. Its covariance, and hence clarity, does not depend on the observed
scalar values. An environmental posterior mean is not needed for this objective;
clarity is computed from the model rather than painted around a desired route.
The three scenarios use different obstacle arrangements and independently
sampled stationary vortex/jet wind fields. Truth is not drawn from the estimator
basis and is never supplied to POAM, the planner, or RPCBF.

## Wind inference and planning

The two wind components are independent GP outputs sharing kernel hyperparameters
and inducing locations; the variational mean has one column per component and
the ELBO sums both outputs. This is an explicit vector-output extension of the
scalar POAM equations.

POAM uses 24 movable inducing points, a neural attentive kernel with four RBF
length scales, transported unweighted sufficient statistics, analytic variational
updates, and minibatch kernel training over the accumulated measurement history.
Noise standard deviation is known (0.1 by default), rather than estimated from
motion. Measurements are direct simulated noisy wind observations at executed
positions. The wind is an acceleration disturbance in the double-integrator.

Prediction and planning both retain the full sparse predictive covariance,
including conditional residual uncertainty away from inducing locations. During
one optimization call the kernel and inducing positions are frozen. Hypothetical
observations update the POAM sufficient statistics and variational posterior;
real updates between plans also relocate inducing points and train the kernel.
The comparator receives real updates too but does not anticipate future updates.

Continuous posterior samples retain the residual process using decoupled Matheron
sampling. A joint full-kernel draw on 667 spatial support points corrects the
prior paths; 16 Sobol spectral nodes per RBF component retain the residual
between support points. A second correction conditions on the POAM inducing
distribution. This finite numerical sampler is checked against full POAM
covariance and is not an exact continuous GP draw. Post-survey covariance error
was 0.22–0.27% across the three fields; the rejected spectral-only version had
12–17% error. Both checks are archived. `scripts/validate_full_poam.py` reports resolution convergence;
mission metrics report the final full-grid covariance error. Outer and inner
samples remain independent and spatially coherent. Real MBD reverse-score updates
and the spline RPCBF/QP are retained. Allowed barrier slack is reported separately
from collisions and task deficit.

## What the optimizer evaluates

The feedback variant uses three waypoint knots for the first period and for each
of two shared continuation branches (18 latent policy coordinates). The nominal
controller tracks these waypoints with wind-mean compensation; RPCBF then filters
its bounded control. MBD retains the paper's forward noising, Gaussian proposals,
weighted score estimate and reverse updates. It optimizes this policy vector,
not a drawn path in the visualization.

For each candidate, a fixed outer wind draw generates a physical rollout and noisy
measurements. The first-period filter uses the currently available POAM belief.
At the boundary, hypothetical POAM conditioning uses those measurement values
and positions. A task-cost router chooses a shared future branch from the
observable robot state, the full clarity vector and the posterior. Its predictive
samples are separate from the hidden outer field. Later hypothetical periods
repeat conditioning at their boundaries. All stages propagate all 475 QoI clarity
states and accumulate their positive deficit; no wind-information bonus is added.

Execution uses only the first period. Real observations then update POAM, including
kernel training and inducing relocation, before a new MBD optimization. The
current-belief comparator uses the same feedback controller, full POAM, RPCBF,
policy budget and real update cadence; it freezes the wind belief only in its
future predictions. Thus its label does not mean that it is prevented from learning
in the actual mission.

The demo distinguishes three objects: sampled diffusion forecasts, the selected
forecast, and the actually executed trajectory and clarity field. The final cost
and the whole-field curves always come from the third object.

## Reproduction and interpretation

```
JAX_PLATFORMS=cpu uv run pytest -q
JAX_PLATFORMS=cpu uv run python scripts/validate_full_poam.py
JAX_PLATFORMS=cpu uv run python -m voi_diffusion_ipp.experiments.run --config configs/field_poam.yaml
JAX_PLATFORMS=cpu uv run python -m voi_diffusion_ipp.viz.demo_data --config configs/field_poam.yaml --seed 300
JAX_PLATFORMS=cpu uv run python -m voi_diffusion_ipp.viz.demo_server --port 8765
```

The estimator survey is a separate diagnostic, not a planner success claim.
Whole-field planner outcomes must be assessed from paired completed missions,
including regressions, collisions, and optimizer failures. The old corridor's
percentage improvements do not transfer to these experiments. CPU planning time
is measured separately from compilation and visualization rollouts. Linux/CUDA
and a real-time planning rate remain unvalidated on this machine.
