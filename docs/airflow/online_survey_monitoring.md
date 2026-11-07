# Online survey and disturbance-aware monitoring

This added experiment combines monitoring with short surveys of wind along useful geometric routes. All earlier environment animations, robot tests, surveyed comparisons, unsurveyed comparisons and Bayesian look-ahead results remain available. A survey is executed by the same robot with the same sensor and safety filter; observations are not supplied before the mission.

## Why change the previous method?

The earlier finite look-ahead did not produce the desired inter-region monitoring. Untargeted information objectives also spent time revisiting nearby space. Two implementation details mattered: geometric starts proposed monitoring destinations without explicit survey frontiers, and the sparse estimator could retain distant inducing locations even after repeated local observations. Its conditional residual then remained large at the sensor, so the robust safety filter could continue blocking motion. This motivates both route-directed survey starts and observation-supported inducing allocation. It does not establish that an exploration reward alone solves the problem. A further diagnosis found that a 0.25-second safety-filter forecast could predict a passage that the same controller at its actual 0.05-second rate would not traverse. The added final validation therefore uses the execution rates over the entire horizon. Refining only the first five seconds left a false delayed-progress prediction in the tail and was rejected.

## Architecture and decision sequence

- Update the wind belief from the newest actual local measurement and update the full clarity field along the executed robot motion.
- Search the collision-free geometry for valuable monitoring destinations and alternative passages. Construct full monitoring routes and shorter survey prefixes on those routes.
- Estimate whether knowing wind could improve the cost of a useful geometric route. Reward candidate measurements by their route-specific covariance reduction, scaled by that task value and a bounded remaining-use factor.
- Optimize every acceleration coordinate with the original model-based diffusion method. Independently validate the final candidates with the hard RPCBF and fresh posterior wind fields.
- Execute the first five seconds, filtering the control every 0.05 seconds. Assimilate wind measurements every 0.2 seconds and refresh the execution belief. Replan after the committed interval.

The nominal planning horizon is 15 seconds. There is no branching look-ahead and no hypothetical update of the executed estimator. The covariance-reduction score below is an explicit survey surrogate, not the exact long-term value of information.

## Physical models and monitoring objective

The state is position p and velocity v. The unchanged robot dynamics are p_dot = v and v_dot = u + (0.1/0.680) ||w(p) - v|| (w(p) - v), with each control component bounded by 1.6 m/s². The wind velocity w(p) is spatially varying and constant in time. Both experiments use the same existing stationary CFD fields, three circular obstacles, walls and monitoring rectangles.

Clarity evolves in all 1,884 free grid cells: q_dot = (C²/R)(1 - q)² - Q q². The sensor footprint C depends on the robot position; Q is 0.16 in Regions A and B and 0.02 elsewhere. The monitoring cost J is the integral of mean positive clarity deficit, plus the existing clearance and speed penalties. Region targets, sensor parameters and decay rates are unchanged.

## Geometric monitoring routes and survey starts

An online 128 × 74 geometric grid represents the narrow upper passage. Clearance-weighted graph search selects four destinations from predicted clarity deficits and obtains two alternative passages toward a distant valuable destination. The finer grid changes initialization only: the original clarity cells, analytic circular geometry, double-integrator reference and physical rollouts remain.

Forty-eight target locations sample the complete monitoring routes. Their weights reflect predicted endpoint clarity need and progress along each route. Eighteen candidate survey prefixes end at several distances along three alternative task routes. Full current predictive-GP covariance reduction ranks these prefixes, with a geometric travel penalty. Two spatially distinct survey starts join three passage starts, one local clarity-directed start and one hold start. The shifted previous control sequence is an additional incumbent. No destination is hardcoded as A or B, and no true wind values enter this search.

## When is a survey useful?

Three paired route-value probes use the same reference and posterior-sampled physical wind. One controller uses the current uncertain belief; the other is given that sampled field as a perfect-information diagnostic. These probes retain the old 0.25-second forecast and are explicitly a value surrogate. They never query the true environment field, determine final physical progress, or generate candidates from fresh validation fields.

For each route r, define V_r = max(0, min_s J_uncertain,s - J_known,r). A known-field route that predicts contact receives zero value. A route therefore earns no reward merely because its tracking error improves: its known-field task cost must beat the best current route. Only targets on positive-value routes receive information weight. Weighted wind standard deviation must also exceed 0.30 m/s and aerodynamic acceleration variation 0.10 m/s². Zero wind-to-force coupling disables the reward. These finite tests do not prove that every irrelevant uncertainty is rejected or every useful survey is discovered.

## Task-valued survey objective

Let T_r be weighted wind targets on a useful route and X the prospective sensor locations during a candidate's first five seconds. The current full predictive covariance is Sigma. Conditioning this GP on noisy observations would reduce target covariance by Delta Sigma_TT = Sigma_TX (Sigma_XX + sigma_n² I)^(-1) Sigma_XT. Both wind components and all spatial cross-covariance terms are retained. The route gain g_r is its weighted trace reduction divided by current weighted route variance.

Candidates minimize J - B, where B = N max_r(V_r g_r) and N = clip(remaining mission seconds / 15, 1, 20). The maximum avoids adding benefits from alternative routes as though all were taken simultaneously. N is an explicit linear repeated-use assumption for stationary wind; it is not a solved future policy or exact value of information. It decreases with remaining mission time. A further constraint limits J to the best candidate's monitoring cost plus 1.0. Search and independent validation each establish this allowance from their own candidate bank.

The selected action invests in exploration only when its learning reward repays additional predicted monitoring cost. The recorded exploration flag means that it incurs more monitoring cost than the best validated monitoring candidate; enabling information does not by itself mean that exploration was selected. Monitoring-only sets B to zero. If every V_r is zero or the force/uncertainty gates fail, the objectives coincide.

Final gains use the robot's predicted positions at all 25 actual 5 Hz measurement times from the execution-rate forecast. Search uses its existing coarse physical forecast to rank proposals. Predicted contact excludes final candidates and earns no information reward. Full-GP conditioning gain is not asserted to equal the reduction achieved by a relocated sparse POAM update. Actual measurements and posterior changes are retained separately; no hypothetical observation is inserted into the executed estimator.

## Original model-based diffusion and safety filter

The optimizer retains the original cosine schedule, Gaussian proposals, cost-weighted score, reverse updates, seven geometric modes and shifted incumbent. The standard budget has two reverse levels and fourteen proposals per level. Every one of the 120 reference-acceleration coordinates remains free; no trajectory is replaced by a spline parameterization or fixed geometric path.

MBD search uses the original relaxed RPCBF with two coherent construction fields and a 0.06 m margin. Five finalists receive independent full 15-second validation with the hard RPCBF, eight construction fields and a 0.12 m margin. They are the global MBD training winner, the original hold, and the best trained candidate in each of the three passage modes. This finite candidate subset is selected using training scores only. One independent posterior physical wind draw evaluates performance; reducing that Monte Carlo count from two to one funds execution-rate validation without reducing the eight safety construction fields. Execution uses that same hard-filter definition. The relaxation remains a search device; it is not substituted for the executed filter. Original spline extrema and backup-policy rollouts are retained.

If the hard-filter constraints are infeasible, the existing solver uses its explicit minimum-common-slack fallback while retaining actuator bounds. Such a step is not certified safe. Contact-free execution and zero filter infeasibility are different outcomes; both the number of infeasible steps and maximum fallback slack are reported. Final validation rejects predicted contacts but does not assert a global safety guarantee.

## Wind learning and remembered survey locations

There is no initial survey. Both methods start from the same zero-mean, geometry-initialized POAM prior and receive the same observation-noise model. Actual noisy wind readings arrive every 0.2 seconds, with standard deviation 0.1 m/s. The kernel stays fixed and all conditional residual covariance is retained.

The added estimator keeps 128 inducing variables. A mask records centers supported by actual measurements. A new measurement farther than 0.15 m from all protected centers replaces the nearest unprotected center and protects it. Previously measured centers remain until all slots are occupied. At capacity, the declared nearest-center replacement applies and saturation is reported. Accumulated unweighted POAM statistics are transported with the original equations; they are never reset. Removing support can increase uncertainty in its former neighborhood, and that increase is retained.

The allocation policy is a declared change to the sparse estimator's representation. It does not drop covariance terms or replace the wind model. Actual inference uses float64 to avoid ill-conditioned local-center updates. The recorded physical readings are converted to float32 observation inputs before that update, matching the runner's explicit numerical contract. The support mask is saved with every planning checkpoint and execution archive.

## Matched comparison

Online survey + monitoring adds the task-valued covariance-reduction reward when the relevance gate is active. Monitoring only always minimizes the ordinary monitoring objective. Both use the same protected observation-supported POAM estimator, geometric monitoring and survey starts, wind-aware physical forecasts, MBD budget and execution filter. The comparison therefore tests the objective-selection mechanism with matched estimator and initialization, rather than comparing it to a planner denied wind measurements. The older still-air baseline remains in the earlier simulations.

## Computation and verification

The one-second warm-planning guard covers geometry, posterior-field sampling, the three relevance probes, MBD search and independent final validation through executable-control selection. Compilation is reported separately. The webpage reuses the selected final-validation trajectory; it requires no additional physical rollout. Data serialization and offline audits are not included in online planning time. A rejected over-budget plan is not executed; any explicit checkpoint resume preserves and reports that rejected call.

The physical execution uses the existing 0.01-second RK4 integration and 0.05-second control updates in float64. Actual POAM inference and posterior field generation use float64. Tables are then converted to float32 for the forecast; execution retains float64 tables. Final validation updates tracking and the hard RPCBF every 0.05 seconds and integrates the plant and all clarity cells every 0.01 seconds for all 15 seconds. The original 0.25-second forecast remains only in MBD search and paired route-value surrogates. All use the existing 0.2 m C1 wind tables. These are declared numerical discretizations, not exact continuous-time solutions.

Robot trajectories are computed before clarity in the refined forecast because clarity does not enter the tracker or filter. Sensor coefficients are then evaluated in batches, followed by the same sequential clarity RK4 recurrence. Regression tests compare every clarity state, contact handling and integrated cost against the original interleaved execution equations. This is an algebraic reordering, not coarser sensing or reduced cells.

The survey covariance calculation uses Cholesky triangular solves with the existing kernel regularization, rather than multiplying a rounded stored inverse. A diagnostic pilot exposed a negative float32 observation-covariance eigenvalue from that inverse multiplication despite a valid float64 posterior. The solve-based calculation retained positive observation covariance and finite information scores at the same state. No residual covariance is removed or clipped to obtain that result.

The audit verifies every sensor reading against the stationary field and seeded noise, every actual posterior update and protected-center mask, every displayed belief frame, every recorded held-control robot/clarity/cost step, posterior-generated planning fields, saved MBD equations and candidate selections. Full optimizer replay coverage is reported explicitly. Neither these audits nor one development seed establish global safety or a general performance guarantee. Contact, region visits, time to reach A, early/late monitoring cost, mode durations, learned wind maps, runtime and capacity saturation are reported from the actual runs, including unsuccessful outcomes.
