# Implementation choices and limitations

- CPU baseline: JAX 0.4.38, float32, Python 3.11, UV lock. All environments and
  download caches are inside this repository.
- GP prototype: finite normalized radial-basis feature kernel. Conditioning is
  exact for this kernel; smooth spatial cross-covariance and a fixed episode field
  are retained. This is not a full RBF GP or POAM.
- MBD follows paper Eqs. (6), (8), (9). Unbounded Gaussian latent coordinates map
  through tanh to bounded physical controls. Warm starts use forward noising;
  every diffusion level uses a fresh key. Fixed temperature in task-cost units.
  Best evaluated candidate is retained, including initial and final iterates.
- RPCBF: finite-horizon coherent sampled-field design-policy rollouts, RK4,
  not-a-knot spline extrema including initial state, state jacobian, robust
  derivative inequalities over all inner instantaneous winds. Backup policy is
  damped velocity with posterior mean compensation. The QP is exactly solved in
  two control dimensions by enumerating active sets. Infeasible QPs minimize a
  common relaxed barrier slack while respecting hard actuator limits; every such
  occurrence is reported. No claim of safety in infeasible states is made.
- Full spline is default. `fast` uses seven samples around the discrete maximum;
  it can miss other extrema and is separately compared. `discrete` is diagnostic.
- Two-stage objective uses unsmoothed positive-part deficit, trapezoidal time
  quadrature, and optional terminal-seconds times final deficit. No actuation or
  uncertainty bonus is present in the proposed objective.
- Branch router uses posterior crosswind mean and uncertainty at the corridor.
  It is fixed before optimization. Scenarios in one branch share all future
  nominal controls. The controller does observe its physical state.
- Matched no-information continuation holds the branch-control library fixed,
  starts at exactly the same terminal state/clarity, and routes using the old
  belief. It is a policy/filter diagnostic, not a separately optimized VoI bound.
- Direct noisy wind observations are simulated; no dynamics-residual sensing
  claim is made. Beliefs and inner random fields freeze until a horizon boundary.
- Sampled RPCBF is approximate and does not provide a formal safety guarantee.

## Numerical feasibility and recovery

Observed or simulated collisions are not rewarded or hidden. Candidate scoring
rejects sampled physical collisions with infinite cost (MBD's feasibility factor);
the same criterion is used by CEM and MPPI. QP infeasibility remains a separate
recorded diagnostic because the bounded relaxed fallback may avoid a collision.
If every candidate has infinite cost, execution uses the fixed clarity-feedback
controller through the same RPCBF and logs an optimizer failure. This is a
fallback episode, not a successful optimized plan. Include it in mission tables.

The `heuristic` method directly optimizes the draft's positive-clipped reset/replay
score. `draft_selection` more closely reproduces the draft structure: passive MBD
produces a library of denoising iterates, then the reset/replay score selects one.
Stein generation is intentionally replaced by a clearly named MBD generator;
this is not a reproduction of the complete Stein pipeline. Its extra evaluations
are counted separately. CEM and MPPI use exactly the same joint objective,
initialization and objective-evaluation budget as proposed MBD.

`warm_starts: 3` compares the current warm start with two task-only seeds (go to
ROI A; hold the current position), then runs every diffusion iteration normally.
Their three additional objective calls are counted identically for all solvers.
It is an initialization ablation; no probe reward or hidden wind enters it.

## POAM adapter

The online POAM implementation retains unweighted accumulated kernel statistics,
transports them when inducing inputs/kernel change, selects inducing inputs by
pivoted Cholesky, analytically updates the variational distribution, and adapts
an attentive-kernel NN/amplitude with minibatch variational EM. Observation-noise
variance stays fixed. The full sparse predictive joint covariance includes the
conditional residual. Planning freezes a finite-anchor snapshot and omits the
residual between anchors, which is logged on monitoring cells. This permits
coherent field samples and static array shapes; it can underestimate uncertainty
between anchors. The default feature-GP experiments do not validate POAM-based
mission performance. `configs/poam.yaml` is only an integration smoke check.
