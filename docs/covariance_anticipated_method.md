# Covariance-anticipated MBD with sampled spline RPCBF

User-authorized separate experiment, October 17, 2026. The original stochastic
realization-anticipated planner, original scenarios, archived results and 3D demo
are retained. Scheduled research remains paused. No information reward is added.

## Mathematical object and approximation

The decision variable is the nominal feedback-policy vector theta. Its cost is
the cumulative whole-grid clarity deficit of the dynamics after RPCBF filtering.
The rollout includes measurements and causal belief updates. Several policies can
produce the same filtered motion; invertibility and a density on executed paths
are neither assumed nor required. MBD averages **nominal policy parameters** using
their executed-rollout costs, not scenario-specific executed control vectors.

The original planner approximates E[C(F(theta,xi;b))] with common random spatial
wind worlds and observation noise. It exponentiates the *average cost*, not the
average of exponentiated world costs. The new covariance approximation evaluates
one deterministic physical forecast using the earned current posterior mean.
It removes outer worlds, not the independent coherent inner RPCBF field samples.
All physical execution still uses the hidden physical field, 20 Hz RPCBF, 5 Hz
measurements and full real online POAM with kernel learning/inducing relocation.

At 2, 4, 6 and 8 seconds of a 10-second forecast, the new planner conditions the
full current POAM predictive Gaussian process on the predicted **executed**
measurement locations Z. The kernel, inducing representation and mean stay fixed:

    K_new(x,y) = K_b(x,y) - K_b(x,Z)[K_b(Z,Z)+R I]^-1 K_b(Z,y).

K_b includes Kxx - Kxz Kzz^-1 Kzx as well as the inducing posterior covariance.
No diagonal-only uncertainty or finite-anchor projection is substituted.
Inactive fixed-size buffer entries have zero cross-covariance and unit diagonal.
A 1e-7 observation-matrix regularizer is numerical jitter. Conditional inner
sample paths use a Matheron correction with independent sensor-noise draws and
the same original field draws throughout a candidate's forecast. Continuous
draws retain the existing POAM spectral/support approximation; covariance
inference is analytic. C1 spatial tables approximate field evaluation only.

This is exact Gaussian covariance conditioning of the **current POAM predictive
process with fixed locations/kernel**, not an exact prediction of a later POAM
variational refit, inducing relocation or kernel learning. The test suite verifies
agreement with the refit at inducing observations and their distinction away
from inducing inputs. Full POAM remains the real estimator. The approximation
also omits the random changes in posterior mean and observation-dependent future
measurement locations. A second approximation replaces uncertain physical wind
by its current mean in the robot dynamics. With nonlinear safety interventions
and sensing, motion under mean wind is generally not expected motion, and its
monitoring cost is not the expected monitoring cost. The `mean_frozen` control
is therefore necessary: covariance vs realization anticipation changes both
physical forecast and future-belief assumptions. Neither optimality nor universal
safety is claimed.

Both stages propagate robot and clarity states continuously. No state/clarity
reset, early access to a future covariance, information bonus, forced A/B
alternation or hidden-wind control input is permitted. High uncertainty alone
does not lower the objective. A reduced covariance is useful only if its changed
RPCBF/controller produces better monitoring.

## Controlled comparisons

* Original `proposed`: sampled outer worlds and hypothetical full POAM updates.
* `current_belief`: same sampled worlds, frozen forecast belief.
* `covariance_anticipated`: mean dynamics and full covariance conditioning.
* `mean_frozen`: same mean dynamics, frozen covariance; isolates conditioning.
* `wind_blind_matched`: wind-blind nominal planning, same wind-aware nominal
  execution compensation and RPCBF as the other arms. Historical
  `wind_blind_planner` is retained; it also omits execution nominal compensation
  and is a different comparison.

Before modifying search, compare seven shared geometric seeds and the recorded
winner at original recorded periods 0, 5, 18. Full earned POAM reconstruction must
match archived means/stds. Compare frozen covariance, updated covariance, and
updated-filter-only (frozen reference features). Record common first-stage
identity, continuation cost, correction, clearance, slack, trajectory changes,
and central-probe vs whole-grid uncertainty response. Eight independently drawn
posterior physical fields stress the zero-innovation approximation. These fields
are simulator inputs only; they are not used to select the planning policy.

Retain failures and all candidate outcomes. The completed phases include a
proper bounded target with matching weights; fixed-state 4x8/8x16/16x32 search
comparisons at a controlled noise range; separate independent selection and
evaluation banks; equal-total-evaluation multiple chains; and fifteen declared
matched 120-second missions. Reference numerical results are not global optima.
Fresh-seed confirmation must be separate from method/configuration selection.

## Sources

* Chen et al., POAM, RSS 2024, predictive Eq. 4 and variational updates:
  https://arxiv.org/abs/2406.03669
* Rasmussen and Williams, GPML, Chapter 2, Gaussian posterior covariance:
  https://gaussianprocess.org/gpml/chapters/RW2.pdf
* Wilson et al., ICML 2020, conditional sample-path construction:
  https://proceedings.mlr.press/v119/wilson20a.html
* Pan et al., MBD, Eqs. 6, 8, 9. The finite Monte Carlo ratio is an approximation;
  target concentration is not a finite-compute optimizer convergence guarantee:
  https://arxiv.org/html/2407.01573v1

Implementation: `wind/covariance_forecast.py`, `planning/covariance.py`.
Evidence and exact protocols: `docs/covariance_study/`.

## Proper finite-domain MBD target and controlled search experiments

The new declared experiments use Theta=[-6,6]^18 and a uniform reference measure.
The target is proportional to exp(-J(theta)/temperature) inside Theta and zero
outside. Rejected physical forecasts also receive zero weight. This is a proper
density if the finite-cost feasible subset has positive volume. If all sampled
weights vanish, the finite algorithm has failed to find a valid proposal; no
normalization or safety guarantee is inferred from that failure.

`mbd_latent_bound=6` enforces the support in the evaluated costs/importance
weights, including initial-seed ranking and final evaluation. Gaussian proposals
are not clipped, reflected or replaced by executed controls. No variance reward,
quadratic control regularizer or nonuniform policy prior is added. The chosen
domain still restricts the policy class and is reported as such. Default zero
retains the archived unbounded *optimizer* for reproducibility, without claiming
that its bounded-cost target is a normalized density on all of R^18.

Gaussian convolution of this target gives the same analytic score identity as
MBD Eq. 9. The self-normalized finite Monte Carlo ratio is generally biased but
consistent as proposal count grows when its denominator is positive. That fact
does not establish convergence of a four-level warm-started optimizer or of the
receding-horizon robot. Our filtered rollout can be nonsmooth; differentiability
through RPCBF is not required by the zeroth-order score calculation.

Budget comparisons interpolate log cumulative alpha between the original
four-level schedule's first/last cumulative-alpha values. The 4-level schedule
itself is bitwise unchanged. The 8- and 16-level comparisons therefore retain
the initial/final exploration scales. They are not claimed to match every
intermediate density of the original schedule.

Equal-budget restart comparison: one chain with 4 levels x 7 proposals vs three
chains with 4 levels x 2 proposals each. Both use 7 initial seed evaluations and
30 optimizer objective evaluations, totaling 37. Each chain preserves the
original Gaussian score estimator. Three shared seed families are central/stay,
left bypass, right bypass, each considering both region directions. No route is
compulsory; the lowest evaluated cost wins. Other comparisons use 4x8, 8x16,
16x32, two independent optimizer seeds, and three declared saved states.

Independent reranking uses four fresh posterior worlds/observation streams;
reported post-selection evaluation uses eight further fresh worlds/streams.
The inner RPCBF base draws are held fixed to isolate outer-scenario error.
All candidates are evaluated with causal full hypothetical POAM updates,
including changed posterior means, since actual execution will learn those
means. This is an offline approximation stress test, not the covariance
planner's training objective and not a new independent test of physical-model
calibration. Contacts and raw task costs are reported separately. These small
development banks cannot support a statistical general-success claim.

## Measured development status

The 36 component/regression tests pass. Fixed-state protocols and all results are
in `covariance_study/`. The bounded-target 120-second experiment uses the same
original environment, not an edited version of the archived scenario. Tiny
compiled numerical differences can change receding-horizon selections: in seed
8101 the first observed policy discrepancy is 8.4e-6 at period4, despite zero
proposals outside the finite box. Do not attribute a changed mission cost to
truncation without a controlled test. The archived unbounded 12-second prefix
still reproduces exactly. See `covariance_study/target_numerics_note.json`.

Across the three declared development seeds in the original physical wind,
covariance anticipation has 0.94–4.44% higher cost than its mean-dynamics frozen
control. The realized reduction in uncertainty is therefore not sufficient to
establish improved task performance. Its average of mission-median full-cycle
latencies is 1.84s, versus 2.05s for realization anticipation. The latter has
3.81–12.93% lower cost than wind-aware frozen, and 9.76–14.30% lower cost than
wind-blind matched execution. No new online search heuristic is promoted.

A controlled estimator check at three saved states gives each update operator
the same actual next-period measurement locations (post-hoc diagnostic only).
The full GP covariance, a fixed-inducing POAM refit, a relocated refit and the
actual real-data/kernel-trained POAM update disagree spatially. The actual next
POAM covariance has lower grid-average variance than the GP forecast but is
higher at 19.6–37.8% of cells. Thus the mismatch is not uniformly optimistic or
pessimistic; it must be addressed as model consistency rather than an extra
variance reward. These checks identify plausible limitations, not a causal
proof of the mission regressions. All reports, raw maps and protocols are in
`covariance_study/estimator_consistency_01/` and `covariance_study/README.md`.
