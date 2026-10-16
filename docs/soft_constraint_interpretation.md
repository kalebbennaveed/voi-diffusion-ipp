# Soft safety constraints and current research maturity

The user clarified that safety is a soft constraint: relaxation is allowed.
Nonzero RPCBF slack is therefore an observed tradeoff, not automatically a
failed mission. Collision, clearance, slack and optimizer failure remain separate
outcomes. The existing numerical implementation already permits relaxation;
this clarification does not change the saved trajectories.

## Exact current policy

The filter first projects the nominal input onto the sampled RPCBF inequalities
and actuator box. If those unrelaxed inequalities are infeasible, it computes
the smallest nonnegative common slack that makes the barrier inequalities
feasible, keeping actuator limits fixed. It then projects the design-policy
input into the relaxed set. This is a minimum-required-slack policy, not a
tunable weighted objective that freely trades slack against nominal tracking.
The default does not reject a rollout merely because it uses slack.

The stored `feasible` field describes feasibility **without slack**. A false value
does not mean the relaxed control was unavailable, a collision occurred, or the
diffusion optimizer failed. The optional strict candidate-feasibility experiment
remains disabled. Physical collision checks in candidate scoring remain in place;
softening the derivative inequalities does not disable collision checking.

Slack is measured in common RPCBF derivative-constraint units, not penetration
distance, and depends on constraint scaling. Its integral is the sum of
nonnegative per-step slack times step duration. Counts alone should not determine
which planner is preferable.

## Existing validation with allowed slack

These are the same 30 predeclared missions (ten per arm), totaling 120 seconds
and 800 control steps per arm. No results were discarded or rerun for this table.

| Planner | Mean task cost | Time using slack | Peak slack | Total integrated slack | Physical collisions |
|---|---:|---:|---:|---:|---:|
| No anticipated learning | 4.3803 | 0.15 s | 0.0061 | 0.0009 | 0 |
| Two-stage learning | 3.3722 | 0.30 s | 0.4767 | 0.1029 | 0 |
| Repeated-update learning | 3.1507 | 1.35 s | 0.8935 | 0.4817 | 0 |

Repeated updates improve task performance while using more allowed relaxation.
The nine relaxed steps constitute 1.125% of executed steps in that arm. They do
not invalidate its monitoring improvement under the soft-constraint requirement.
Zero observed collisions remains empirical evidence on this small study.
Reproduce with `uv run python scripts/report_soft_constraints.py`; per-mission
values are in [the summary](multistage_validation/soft_constraint_summary.json).

## What the demo distinguishes

- **Final result** opens first with completed actual trajectories and total,
  initial and later task costs, arrival time, final clarity, clearance and slack.
- **Denoising** shows the selected planner's actual candidates and reverse
  updates for one planning call. Either result card opens its own optimization.
  “Show final selected plan” jumps to the final evaluated selection. Predictions
  are distinct from the actual completed mission.
- **Mission playback** shows both actual paths at the same execution time:
  proposed in teal, comparator in amber. Playback ends at the final comparison.
  Combined denoising/execution playback remains available as an option.

Both main methods assimilate real measurements during execution. The difference
is anticipation: the proposed method simulates future observations and posterior
updates when choosing current motion; the comparator evaluates simulated future
motion using the belief available at the planning boundary. “No measurements”
is a separate control and is labeled separately.

Default seed 200 comes from the completed validation study. Its new demo exports
reproduce archived states, clarity, controls, slack and feasibility bit-for-bit,
as well as exactly reproducing cost. It has cost 3.3777 versus 5.4224 and reaches
ROI A at 6.90 versus 11.70 seconds. This single-run 37.7% reduction is distinct
from the 28.1% mean reduction across all ten paired seeds. Losses and runs with
allowed slack remain selectable.

## Is the method at a good stage?

It is a useful, tested research prototype with a credible controlled demonstration
of task-valued learning. Actual MBD, posterior inference, causal continuation,
and full spline RPCBF are integrated; the improved policy shows a repeatable
paired cost benefit. Treating allowed slack appropriately makes the result easier
to interpret but does not enlarge its experimental scope.

It is not yet broadly validated or paper-complete. The strongest result uses a
known localized wind shape, a small finite recourse policy and small Monte Carlo
budgets. The improved objective still needs multiple geometries and wind models,
informative negative controls, sampling sensitivity and matched-work MBD/CEM/MPPI
comparisons. Solver superiority is unestablished. The validated M3 backend is CPU,
with planning around several seconds per call.

The next gate is a task-cost/slack/compute operating region: vary disturbance,
sensing quality and geometry, report the full tradeoff, and freeze a new untouched
seed set after development. A soft constraint removes the need to demand zero
slack; it does not remove the need for fair comparisons.
