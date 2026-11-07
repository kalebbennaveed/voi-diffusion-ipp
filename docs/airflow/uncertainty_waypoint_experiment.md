# Executed uncertainty waypoints and learn-then-MBD

The first experiment is **one waypoint policy with RPCBF in its execution loop**.
There is no separate unfiltered run, and no MBD in that policy. Baseline 1 runs
the same learning policy first, then hands its actual state and wind belief to
the existing disturbance-aware MBD. All earlier demo sections and data are kept.

## Waypoints

The free-space grid uses the existing 0.125 m spacing and analytic circular
obstacle clearance. Dijkstra supplies travel distance and a geometric route.
Candidate score is

\[
  s(z)=\frac{\operatorname{Var}[w_x(z)\mid\mathcal D_t]}{1+L(p_t,z)/(1\,\mathrm m)}.
\]

Both wind components share this marginal variance. Candidates must be at least
0.35 m from the robot. A target persists until arrival (0.18 m), its variance
falls below 0.25 (m/s)^2, or a progress check triggers a 30 s local cooldown.
The progress check compares the path length to its selection-time value after
15 s. This is a heuristic waypoint policy, not an optimal information planner.

The nominal edge velocity is 0.4 m/s. Edges are screened using the existing exact
wind-force law at the POAM mean: either endpoint requiring more than 1.2 m/s^2
of componentwise compensation is rejected. This reserves 0.4 m/s^2 relative to
the existing 1.6 m/s^2 actuator limit. The screen is a nominal feasibility test,
not a robust safety certificate. Short visibility chords retain the grid corridor.

Each 0.2 s, the actual state and current posterior produce a new path and local
velocity setpoint. The existing tracking controller and full-uncertainty RPCBF
run every 0.05 s; the physical robot and clarity integrate at 0.01 s. All
infeasible hard QPs and minimum-slack fallback commands are recorded.

## Actual learning

Measurements have the existing 0.1 m/s Gaussian sensor noise and are taken only
at executed robot positions, every 0.2 s. There are no hypothetical or map-wide
synthetic observations. The true field is restricted to plant integration,
actual sensor generation, and offline RMSE diagnostics.

The recorded final experiment uses the original POAM pivoted-Cholesky inducing
selection, 128 variables, full sufficient-statistic transport, and full
conditional residual covariance. Unchanged point sets use exact permutation
transport. The fixed-kernel pilot and earlier observed-slot allocation were
retained as diagnostic runs.

The final experiment enables the repository's existing variational EM objective:
four Adam steps with learning rate 0.001 on a 32-measurement minibatch every 2 s.
Only actual historical measurements enter the minibatch. An observation-free
transported E step follows the M step; all prediction and prior-sampling caches
are refreshed to match the updated kernel. Training never occurs inside a
candidate planning rollout.

## Baseline 1

The tested budget is 60 s learning + 240 s monitoring, totaling 300 s. This budget
was selected during development after a longer 120 s learning pilot failed; it
is not an independently validated budget. The robot state, velocity, clarity,
POAM statistics, and measurement history are retained exactly at the handoff.
The two experiments' first 60 s are checked for exact equality.

MBD remains the current V5 implementation: 15 s horizon, six geometric starts
plus hold and the shifted incumbent, independent chains, 43 search evaluations,
two coherent training fields, and independent hard-filter finalist validation.
Its existing measurement-support distribution is retained:

\[
 w_{\rm plan}=r\mu+\sqrt r\,(w_{\rm posterior}-\mu),\qquad
 r=\max(0,1-\sigma_{\rm posterior}^2/\sigma_{\rm prior}^2).
\]

Safety uses the full, unattenuated posterior. Learning and training continue
during monitoring. The MBD warm planning guard is one second. First JIT calls
are separate. Waypoint timing includes posterior prediction, graph search, and
the most recent actual inference update; the first kernel-training compilation
is present in its recorded maximum and must not be mistaken for steady runtime.

## Reproduction and evidence

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/airflow/run_waypoint_learning.py \
  --output results/airflow/waypoint_learning_new \
  --seconds 300 --survey-seconds 60 --modes survey,baseline1 \
  --scenarios 1,2 --train-kernel --wall-limit 360
```

The script refuses an existing output directory and stops a run at contact.
All run outcomes, including failures, must remain visible. Lower posterior
variance is not evidence of lower actual error: the demo reports both. A run
that reaches A but later contacts a boundary is not a successful 300 s baseline.

Results are under `results/airflow/waypoint_learning_20261104/`. The final audit
and preservation checks are under `output/waypoint_learning_20261104/`.
