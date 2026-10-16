# Full-POAM whole-field integration results

Completed on 2026-10-16 on the native M3 CPU. These are **18-second development
missions**, with one declared seed per scenario. They validate integration and
expose limitations; they are not evidence of persistent steady-state monitoring
or statistical superiority.

The numerical implementation was frozen for all six runs. The source hashes,
config, scenario/seed selection and raw-data hashes are archived in
`field_poam_validation/protocol.json` and `raw_manifest.json`. Every cell, executed
clarity history and final cost was checked by `scripts/report_field_poam.py`.
Both planners receive actual POAM updates; only the proposed planner anticipates
future information in its stochastic objective.

| Scenario / seed | Proposed integrated deficit | Current-belief integrated deficit | Proposed change |
| --- | ---: | ---: | ---: |
| Vortices / 300 | 5.46429 | 5.22799 | **4.52% higher** |
| Shear / 301 | 5.35462 | 5.23498 | **2.29% higher** |
| Mixed flow / 302 | 5.57270 | 5.95262 | **6.38% lower** |

No collisions or optimizer fallbacks occurred in these six executions. Allowed
barrier slack is retained in the raw metrics and demo; it is not classified as a
physical collision. These are six observations, not a safety guarantee.

The mixed-flow example is a useful full-model case where anticipating learning
improves the actual task. The two regressions remain visible and prevent a claim
of consistent benefit. There is no positive-clipped VoI reward or added wind
information bonus in any proposed run.

## What the diagnostics establish

- In vortices, proposed wind RMSE was 0.218 versus 0.268 for the comparator,
  while monitoring cost was worse. Better wind estimation is not a sufficient
  success criterion.
- In mixed flow, proposed wind RMSE was 0.268 versus 0.239, while monitoring
  cost was better. The relevant objective is the monitoring task, not minimum
  wind RMSE everywhere.
- Mean matched continuation benefits were 0.00955, 0.00584 and 0.02581 for the
  proposed vortex, shear and mixed runs. These compare learned and unlearned
  continuations from the same terminal robot/clarity state and physical wind.
  A local positive diagnostic does not ensure a lower total mission cost.
- Diffusion changed the selected warm-start policy in 8/10, 7/10 and 6/10
  proposed replans, respectively. Mean cost improvement after the first diffusion
  level was small (0.00171, 0.00509, 0.00240). This is evidence of optimization
  activity, not evidence that MBD outperforms CEM or MPPI.
- Final posterior sampler covariance errors over the whole grid ranged from
  0.027% to 0.046%. These quantify the numerical field sampler; inference itself
  retains the full sparse POAM covariance. They do not establish that the GP prior
  perfectly models the independently generated physical wind. Empirical 95%
  coverage ranged from 91.2% to 100% across the six completed missions.

The independent POAM survey in `field_poam_validation/mapping.json` reduced wind
RMSE in every environment. A coarser sampler-support benchmark and a candidate
batching benchmark are retained in the same directory. Coarse supports increased
covariance error, and larger batches produced no meaningful measured throughput
gain on this machine; the persistent study retains the 667-point support and
candidate chunk size one.

## Next experiment and interpretation

The [persistent monitoring study](persistent_monitoring.md) runs 122.4-second
missions over obstacle islands, staggered channels and a demand mosaic, with
explicit spatial process covariance rates. It measures repeated visits,
recovery after clarity loss, and deficit after the initial 30 seconds.
All six scenario/method combinations are declared before their outcomes.

Longer execution is necessary to test maintenance, but is not presumed to fix the
regressions. If value remains weak, the next controlled checks should separate
limited planning lookahead, Monte Carlo error, branch/policy expressiveness,
solver budget and the actual change in RPCBF behavior after learning. Increasing
rollout budgets must be assessed against the same budget for the comparator.
An arbitrary information reward or a hidden-truth controller would change the
method and is not an acceptable fix.

Full-model negative controls, multiple seeds per environment, convergence with
larger budgets, and matched-budget solver comparisons remain necessary before
calling the proposed method research-validated. The analytic-field known-wind
oracle is still explicitly unsupported; it is not silently replaced with a GP
fit. CPU planning is offline, and GPU acceleration has not been established.
