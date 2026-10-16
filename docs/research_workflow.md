# Research workflow after the first implementation

The aim is to develop a useful task-aware dual planner, with permission to revise
the original approximation. The original implementation and its 120-run evidence
remain a reproducible reference. New variants have explicit configuration fields
and separate outputs. Reference repositories remain read-only.

## Working hypothesis

Information is valuable when it changes a consequential future action. The first
implementation often used one branch and acquired useful measurements while
already doing the task. It therefore gave active planning little opportunity to
improve on ordinary online learning. A fixed crosswind threshold also ignored
the actual monitoring value of candidate futures. Open-loop acceleration knots
made performance sensitive to wind and consumed optimization budget recovering
basic feedback behavior.

The next experiments separate three questions: does a useful information-gathering
decision exist; can the approximate belief policy find it; and does diffusion
solve that problem better than competing optimizers?

## Literature informing the changes

- [Arcari, Hewing and Zeilinger, approximate dynamic programming for dual SMPC](https://arxiv.org/abs/1911.03728):
  sampled belief evolution and a dual/exploitation horizon motivate checking the
  quality of recourse before enlarging optimization budgets.
- [Hu et al., shielding-aware dual control](https://arxiv.org/abs/2302.00171):
  future inference and safety interventions should be considered together. We
  retain actual RPCBF execution inside all candidate evaluations.
- [Knaup et al., dual model predictive diffusion](https://arxiv.org/abs/2502.09918):
  dual control with diffusion already exists. Its multimodal receding-horizon
  optimizer motivates a later controlled comparison of warm-start distributions
  and multiple particles. Our potential contribution must concern task-relevant
  spatial disturbance learning and its effect on persistent monitoring through
  the safety filter, rather than merely combining diffusion and dual control.

These are methodological directions, not imported implementations or inherited
guarantees. We retain sampled observation-conditioned posteriors, one fixed
physical field per world, and the paper-equation Pan et al. MBD update. In
particular, we do not collapse all future observations to one optimistic posterior.

## Iteration 1: task-conditioned feedback recourse

Implemented as optional configuration choices:

1. `control_representation: waypoint`: MBD optimizes waypoint knots for a shared
   position/velocity feedback policy with available-mean wind compensation.
   RPCBF still filters every physical input. This deliberately expands the
   original common-open-loop-control approximation to common feedback-policy
   parameters. Actual controls may differ when observed states differ; no hidden
   wind is available to the policy.
2. `branch_rule: task`: select the future policy with the lowest predicted
   safety-filtered monitoring cost under the available posterior. These costs
   use fresh posterior-predictive fields independent of the outer true-world
   sample. Selecting the branch by its realized outer-world future cost would
   reveal hidden information and is explicitly excluded.
3. `localized_corridor`: a smooth Gaussian plume with known spatial shape and
   unknown two-component strength. This is a rank-one spatial GP per component,
   not a general wind benchmark. Remote monitoring provides weak measurements
   of the passage; approaching the plume provides stronger measurements.
   Physics and inference use the identical coherent feature model.

The task cost remains executed clarity deficit. No information bonus is added.
The old acceleration and threshold configurations remain the defaults. New
recourse uses only a small finite library and Monte Carlo action values, so it
is still an approximation to belief-space dynamic programming. Inner and outer
sampling errors must be checked before claiming robust performance.

## Decision gates for the next iterations

1. **Mechanism screen.** On development configurations, evaluate a small library
   of monitoring, approach and probe policies. Check whether useful observations
   can change the preferred continuation, improve admissibility and repay their
   initial clarity cost. Include known-wind and no-measurement controls. If even
   a strong continuation cannot exploit learning, revise the scenario or policy
   class before tuning diffusion.
2. **Paired development trials.** Use seeds 20–22 to debug and compare design
   choices. Record every configuration tried. The first variants are feedback
   recourse, task-based routing, and localized uncertainty. Ablate them separately
   after a promising combination emerges.
3. **Frozen fresh-seed evaluation.** Freeze the configuration before running
   seeds 100–109. Compare proposed and frozen-future-belief planning with the
   same real online updates, controller representation, random-key namespaces
   and computational budget. Report paired cost differences, bootstrap intervals,
   access times, collision/QP/fallback metrics and initial sacrifice/later recovery.
   These seeds become validation data afterward, never fresh test data again.
4. **Generalization and negative controls.** Sweep prior uncertainty, spatial
   correlation, measurement quality, corridor width and mission length. Identify
   an operating region, including cases where probing should not help. Evaluate
   more outer worlds, more inner RPCBF fields and more predictive recourse worlds.
   A promising case is a mechanism example; a stable region supports a method claim.
5. **Optimizer and formulation study.** Compare MBD with CEM/MPPI using the same
   improved objective and policy class, matched total rollout work and wall time.
   Then investigate multimodal MBD particles/warm starts and recourse with several
   belief-update boundaries. The current second period freezes its posterior for
   the entire future window, which can underestimate the benefit of continued
   sensing. Preserve genuine diffusion updates and the actual RPCBF.
6. **Research-scale report.** Freeze the best defensible variant, reserve a new
   untouched seed set, broaden scenarios and increase budgets. Report failure
   regions and compute costs alongside successful behaviors. Only then assess
   whether the evidence supports a paper-level contribution.

Use short experiments to decide the next change, followed by the smallest test
that can falsify its rationale. Keep favorable demonstrations, but retain all
development attempts and independently evaluate any selected configuration.

## Commands

```sh
JAX_PLATFORMS=cpu uv run python scripts/screen_recourse.py
JAX_PLATFORMS=cpu uv run python -m voi_diffusion_ipp.experiments.run \
  --config configs/recourse_dev.yaml --output results/recourse_pilot
JAX_PLATFORMS=cpu uv run python -m voi_diffusion_ipp.viz.report results/recourse_pilot
```

The nested action-value calculation increases work per objective evaluation:
each outer scenario now evaluates both future policies over the configured
number of recourse worlds as well as its actual future rollout. Comparisons
between new and old variants must account for this, not just count MBD proposals.
