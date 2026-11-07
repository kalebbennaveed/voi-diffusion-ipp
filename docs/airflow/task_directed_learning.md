# Task-directed learning with bounded Bayesian look-ahead

This is an added experiment. The existing surveyed and unsurveyed simulations remain unchanged. The purpose is to test whether a robot should temporarily sacrifice monitoring to obtain wind measurements that improve later motion decisions.

## Decision objective

There is **no entropy, variance, information-gain or novelty bonus**. The cost remains the time integral of whole-space clarity deficit plus the original clearance and speed penalties. Candidate control prefixes are compared by the monitoring cost of their future measurement-dependent continuations.

For a candidate prefix a, draw a stationary wind field W from the current full POAM belief. Forecast the physical robot, RPCBF and clarity for five seconds. Generate noisy wind observations at the actual 5 Hz sensor timestamps along that forecast. At the five-second decision boundary, condition the full POAM estimator, including inducing-location relocation. Draw one conditional wind field. Evaluate four geometrically initialized continuation controls in that draw and choose its lowest task cost. This defines a posterior-sampling continuation policy, not an exact minimizer of posterior expected cost. Assess that choice in the same outer W which generated the observations. This last assessment prevents choosing the next action with access to the sampled physical wind itself.

The score is prefix task cost plus continuation task cost, averaged over the declared outer samples. Predicted contact is absorbing; candidates that contact in the evaluated outer trajectory or have no finite contact-free conditional continuation are rejected. Execution always uses the current measured belief and the shared hard RPCBF.

For clarity, the finite decision rule is: b_next = POAM(b, hypothetical observations); draw W_child from b_next; choose the continuation that minimizes task cost in W_child; score the prefix plus that chosen continuation in the original W. The current-belief comparator substitutes b for b_next. Neither rule adds a term proportional to posterior entropy or variance. The future control selection is deliberately restricted to a small library, and the one-draw posterior-sampling policy may perform worse than an exact Bayesian optimum.

If changing wind knowledge cannot affect the feasible controls, physical path, or monitoring outcome, this objective gives learning no benefit. A detour then adds cost without a compensating reward. This does not mean ignoring wind merely because the geometric route name remains the same: information can also improve traversal speed, tracking, or safety along that route. Finite Monte Carlo estimates can still produce ranking errors; there is no universal guarantee that every selected exploration action will help.

## Relationship to original MBD

Original MBD remains the proposal optimizer: cosine schedule, Gaussian proposals, cost-weighted score, reverse updates, seven geometric chains, shifted incumbent, original relaxed RPCBF inside forecasts, and independent final validation. Every reference-acceleration coordinate is free. The added selector considers its winner, hold, and one geometrically diverse alternative from the original seed bank. The look-ahead uses the shared execution filter definition (eight construction fields and 0.12 m margin) to assess these choices.

The matched comparator has exactly the same candidate and continuation machinery but freezes the forecast belief at its current value. Both methods still assimilate real measurements during execution. This comparator isolates anticipated learning; it is distinct from the still-air baseline preserved above it on the webpage.

## Explicit finite planning approximation

The decision horizon is 15 s: a 5 s committed prefix and a 10 s continuation. The final budget uses three root candidates, one outer physical-wind/noisy-observation realization, one conditional posterior draw, and four continuation controls. There is one hypothetical posterior update, at 5 s; measurements are batched there. Actual execution assimilates each observation every 0.2 s. Future continuation controls come from a small online geometric library, not a future full optimization. This is a finite policy and Monte Carlo approximation to belief-space planning, authorized for this added experiment. It is not exact dynamic programming, and the single outer/conditional draws have high sampling variance.

There is no optimistic tail selection. Costs use arithmetic means, and no true CFD values enter the planner. The four continuation policies are hold, a direct geometric path, and two alternative passages to a clarity-selected destination. Destinations and paths are recomputed online from geometry and the full current clarity field.

## Fixed physical and estimation definitions

- Double integrator with the existing air-force mapping: p_dot = v; v_dot = u + (0.1 / 0.680) ||w(p) - v|| (w(p) - v). Per-axis control bound 1.6 m/s².
- The same stationary CFD fields, circles, walls, monitoring rectangles and initial state.
- All 1,884 free clarity cells; q_dot = C²/R (1-q)² - Q q²; Q = 0.16 in A/B and 0.02 elsewhere.
- Full current POAM covariance, including conditional residual covariance, coherent sampling and 128 inducing points. No initial survey. Actual measurements contain 0.1 m/s standard-deviation noise.
- Original full RPCBF backup rollouts and spline extrema. Actual hard filter: eight coherent samples, 0.12 m margin, 0.05 s control interval, posterior/field refresh every 0.2 s.
- Existing numerics: C1 wind tables at 0.2 m spacing; forecasts use 0.25 s RK4 and float32; execution uses 0.01 s RK4 and float64. These are numerical discretizations, not claims of exact continuous integration.

## Runtime and rejected approaches

The 1 s warm-planning cutoff remains active. It covers initialization, wind sampling, original MBD, validation, candidate selection and Bayesian look-ahead. First-call compilation is reported separately. Plot serialization and offline audit replay are outside the online planning timer. Long processes have an external 240 s cap.

A larger Bayesian tree exceeded the budget. Full predictive-GP information gain produced local revisits without reaching A. Inducing-variable information gain also failed to establish progress and exceeded the timing cutoff at 170 simulated seconds. Both information objectives are diagnostics, not the proposed task-directed method. A launch with inherited larger MBD defaults was stopped by the runtime guard after 5 simulated seconds; its settings and timing are retained. A two-outer-branch variant stopped at 45 simulated seconds after a 1.179 s call. With one outer and two conditional draws, both learning runs completed 300 s, but the matched Scenario 1 comparator stopped at 75 s after a 1.135 s call. The final configuration uses one conditional draw to leave more runtime margin. These failures are retained rather than reported as successful comparisons.

## Scope of evidence

The experiments use one development seed per environment. Implementation identities, recorded dynamics, learning maps and runtime checks can verify the implementation; they cannot establish a global safety theorem or a general performance improvement. Actual region visits, contact, filter infeasibility and integrated deficit must be reported even when the proposed method fails to improve them.

The finite Bayesian decision structure follows the motivation in [Arcari, Hewing and Zeilinger, An Approximate Dynamic Programming Approach for Dual Stochastic Model Predictive Control](https://arxiv.org/abs/1911.03728). That paper's guarantees are not asserted for this nonlinear wind-force, POAM and RPCBF implementation. [Wabersich et al., Bayesian Model Predictive Control](https://proceedings.mlr.press/v120/wabersich20a/wabersich20a.pdf) motivates posterior sampling as a computational alternative; the timing-only prototype tested here is not presented as a validated mission result.
