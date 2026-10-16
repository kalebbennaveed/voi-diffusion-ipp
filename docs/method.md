# Sequential VoI diffusion IPP

## Physical state and information

The state is `x=(p_x,p_y,v_x,v_y)`, with `p_dot=v`, `v_dot=u+d(p)`.
One coefficient draw defines the entire unknown episode field. RK4 queries that
same function at every intermediate integration location. Direct wind samples
are `y=d(p_executed)+epsilon`; they are not inferred from motion residuals.

Monitoring is a separate quantity. At cell p,

```
q_dot[p] = C[p](x)^2/R * (1-q[p])^2 - Q*q[p]^2
C[p](x) = exp(-||position-cell[p]||^2/(2*sigma_sensor^2))
c(q) = mean(max(target-q,0))
```

Clarity integration uses executed endpoint/midpoint positions. Mission cost is
the time integral of the unsmoothed deficit. Cells represent a uniform-area
sampling of the two monitoring ROIs; the larger ROI contains more cells and
therefore contributes more to the specified mean, without a hidden reward weight.

## Spatial belief

The default GP prototype is `d(r)=phi(r)^T W`, where normalized RBF features
`phi` are smooth functions of arbitrary coordinates. Independent wind components
share coefficient covariance S, so

```
mean(X) = Phi(X) M
Cov(d_j(X),d_j(Z)) = Phi(X) S Phi(Z)^T
```

Conditioning uses the observed values and full covariance with Joseph-form
updates. Fields are sampled by one Cholesky transform of coefficient noise;
queries within a field do not consume new random keys. This finite-rank kernel
is a declared modeling restriction, not a claim to implement an infinite-rank
stationary GP. The independent-cell ablation replaces the features by nearest-cell
one-hot features only in the belief; the physical truth remains correlated.

The optional POAM adapter is described in `implementation_choices.md` and is
validated independently. Its complete sparse-GP joint prediction and finite-anchor
planning snapshot must be distinguished.

## RPCBF execution

For each constraint j, positive values mean failure. Circle constraints use
`(radius+margin)^2 - squared_distance`; workspace-wall constraints are affine.
A distinct design policy damps velocity and compensates posterior mean wind.
From the actual state, independently drawn inner belief fields drive its rollouts.
The sampled policy value is

```
B_j(x;belief) = max_l max_{0<=tau<=T_B} spline(h_j(x_l(tau)))
```

The full implementation constructs the not-a-knot cubic interpolant over every
backup sample, checks endpoints and derivative roots, then differentiates the
maximum with respect to the initial state. The seven-point local-window variant
is explicitly called `fast`; a discrete maximum is only a diagnostic alternative.

The QP minimizes `0.5 ||u-u_nom||^2`, subject to actuator bounds and

```
grad_v B_j dot u <= -grad_p B_j dot v
                    - max_l grad_v B_j dot d_l(p)
                    - alpha*B_j
```

The instantaneous disturbance term is retained. With two inputs, an exact
active-set enumeration is practical. When the polytope is empty, a separately
reported bounded relaxed solution minimizes common barrier slack. Finite horizons,
finite disturbance samples, integration and spline approximation preclude claiming
formal safety. Collisions, clearance, infeasibility and interventions are logged.

## Causal two-stage objective

A policy vector contains common first-period control knots and B future knot
sequences. Coordinates are unconstrained; tanh maps them to physical actuator
limits, and linear interpolation supplies execution-rate controls.

For each outer scenario drawn from the current belief:

1. Execute the common first-period sequence through RPCBF under the current belief.
2. Generate measurements at executed positions using the same outer physical field.
3. Condition the GP at the first-period boundary; retain actual terminal x and q.
4. Route to one future sequence using only the posterior's corridor crosswind
   mean and uncertainty. All scenarios in that branch share the sequence.
5. Execute the selected future sequence from that terminal x and q using the
   updated RPCBF and the unchanged physical field.

The objective is the scenario-average sum of first and future integrated deficits,
plus optional `terminal_seconds*c(q_final)`. Default terminal value is zero. No
information gain, variance bonus, positive clipping, or counterfactual reset is
part of the proposed objective. Sampled collisions have zero feasibility density
(infinite optimization cost), while QP failure is explicitly recorded. There is
no additional penalty replacing the safety filter.

The real robot executes only the first period of the larger two-period plan,
updates from real simulated measurements at its boundary, and replans. The fixed
continuation approximation replaces future knots with a clarity-feedback policy;
it remains labeled separately from joint optimization.

Outer fields simulate physics and observations. Inner fields are independently
sampled from the controller's available belief. Separate named types and APIs
keep true-world fields out of the filter and router. Scenario buffers and
measurement-noise draws are shared across candidate vectors as common random
numbers; key namespaces separate physical worlds, planning, sensing and safety.

## Exact MBD mechanics

For `alpha_i=1-beta_i` and `ab_i=product(alpha_1...alpha_i)`, warm-start forward
noising initializes `Y_i=sqrt(ab_i)*Y_initial+sqrt(1-ab_i)*epsilon`. At every
reverse level, draw

```
Y_0^k ~ N(Y_i/sqrt(ab_i), ((1-ab_i)/ab_i) I)
w_k proportional to exp(-(J_k-min(J))/temperature)
score = (-Y_i + sqrt(ab_i)*sum_k w_k Y_0^k)/(1-ab_i)
Y_{i-1} = (Y_i+(1-ab_i)*score)/sqrt(alpha_i)
```

This is Pan et al.'s MCSA update, not reverse-SDE sampling or a one-step MPPI
update. Invalid costs receive zero weight. All-invalid iterations preserve the
current mean; a mission with no valid candidate logs failure and uses the
explicit safety-filtered fallback. The best evaluated candidate is retained.
CEM and MPPI use their own updates under the same objective-evaluation budget.

## Optional revised feedback and task-valued recourse

The default formulation above retains the original open-loop approximation. The
research variant in `configs/recourse_dev.yaml` changes the common candidate
parameters to waypoint knots. Interpolated targets define the nominal policy

```
u_nom(x, b, target) = clip(1.2*(target-position) - 1.8*velocity - mean_wind_b(position), ±umax)
```

The full RPCBF filters this input at every control step. All worlds share
the same policy parameters; observed state and available belief may produce
different controls. This is an explicit change in the policy class, not an
open-loop-control implementation detail.

With `branch_rule: task`, branch selection at the first-period terminal state is

```
j* = argmin_j (1/M) sum_m J_future(x_H, q_H, policy_j, b_H, W_m)
W_m ~ p(W | b_H)
```

The predictive fields use separate fixed standard-normal draws, common across
candidates. They are independent of the outer physical field, which is never an
argument to the router. Each action value includes actual safety-filtered dynamics
and clarity evolution. The outer objective then evaluates the selected branch
in the unchanged outer physical field. The comparator substitutes the old belief
for `b_H` during prediction, while retaining the same real measurement updates
between executed MPC periods. Both methods have the same computation structure.

This nested finite-library decision is approximate. Two predictive fields per
branch and four inner safety fields are deliberately small development budgets;
the finite action library and sample noise can misrank branches. The future
period still freezes its available belief throughout, even though real execution
will replan and assimilate more data. Multi-boundary belief recourse is a separate
proposed extension. No information bonus or hidden-world branch assignment is used.

The localized corridor uses `w(p)=phi(p)*a`, where
`phi(p)=exp(-||p||²/(2*length²))` and the two components of `a` have independent
Gaussian priors. Its known spatial shape creates a controlled mechanism study:
measurements near the passage are more informative than remote measurements.
It is a rank-one spatial GP per component and does not establish generality to
unknown spatial structures. See [research workflow](research_workflow.md).

## Diagnostics and interpretation

A matched no-information continuation starts at exactly the same first-period
terminal x and q and uses the same true field. It uses the old belief and routes
with that belief. Its fixed library of future controls is the same as the
learning rollout's library. Report the signed cost difference; do not clip it or
add it to the proposed reward. This isolates a policy/filter effect and is not
an optimally re-solved no-information bound.

Probe-region occupancy is a measured geometric proxy. It does not establish that
probing was deliberate. Mission improvement, initial monitoring sacrifice, later
recovery, no-measurement ablations, branch use, posterior-to-filter effects and
negative controls must be interpreted together. ROI access is physical access;
read it alongside collision and clearance metrics.
