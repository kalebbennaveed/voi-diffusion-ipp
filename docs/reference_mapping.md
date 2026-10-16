# Reference mapping (inspection before implementation)

Read the complete 22-page research discussion and macOS instructions on 2026-10-16. The implementation follows the causal formulation on discussion pp. 7-13; the reset-and-replay draft appears only as a labeled baseline. References below are read-only sibling paths for scholarly traceability, never runtime dependencies.

| Component | Source | Implementation decision |
| --- | --- | --- |
| MBD | Pan et al., supplied `papers/mbd_paper.pdf`, Eqs. (3), (6), (8), (9), Algorithms 1-2; `model-based-diffusion/mbd/blackbox/mbd_opt.py` and `mbd/planners/mbd_planner.py::reverse_once` | Explicit Gaussian proposal variance `(1-alphabar)/alphabar`, weighted candidate mean, analytic score, reverse MCSA. Trajectory example uses narrower proposal and reward standardization; use the paper proposal and fixed temperature in physical cost units. |
| RPCBF | Supplied Safety on the Fly paper, Sec. III C-D, Eqs. (14)-(21), Algorithm 1; `rpcbf-safety-on-the-fly/src/rpcbf/{interp.py,sampler/sampler.py,sampler/uniform_sampler.py,pcbf_rollout_jax.py}` | Independently implement not-a-knot cubic spline extrema, differentiable sampled design-policy value, box-constrained QP. Coherent spatial fields replace piecewise temporal noise. Robust derivative inequalities cover all inner sampled instantaneous disturbances. |
| Shielded scoring | `safe-mpd/mbd/planners/mbd_planner.py` | Architectural reference: filter each candidate during evaluation. No transfer of its deterministic safety claims. |
| Clarity | Supplied Stein clarity paper, clarity/Riccati dynamics; `vg-dip/src/vg_dip/vendor/stein/environments/seven_hotspots.py` | Implement stated `C^2/R (1-q)^2-Q q^2`, smooth footprint, actual positive-part deficit. No Stein optimizer. |
| Wind sensing | Supplied disturbance-aware draft and discussion pp. 8-10 | Direct noisy wind sensing at executed positions; fixed coherent true field, posterior mean and full covariance; boundary-only updates. |
| POAM | `vg-dip/src/vg_dip/vendor/poam/models/poam_model.py::GPyTorchModel.update_variational`, `POAMModel.update_inducing` | Initial finite-feature GP explicitly labeled prototype. Later adapter must retain accumulated sufficient statistics and their transport when inducing points/kernel change. The supplied vendor does not contain an attentive-kernel implementation. |
| Sequential VoI | Discussion pp. 7-13 and `codex_instructions/CODEX_IMPLEMENTATION_INSTRUCTIONS_MAC.md` | Shared first stage; posterior-only branch; continuation from actual terminal robot and clarity; no information bonus in proposed objective. |

## License review

MBD is Apache-2.0; vg-dip and its POAM vendor carry MIT licenses. Copies are retained in `docs/licenses`. RPCBF and safe-mpd clones do not contain a license file: their source is inspected for mathematical understanding, not copied. The RPCBF implementation here is written independently from the published equations. Repository MIT license and initial README are preserved.

## Validation gates

1. CPU environment and physical GP/clarity tests.
2. Spline, RPCBF finite-difference gradient, 2D QP against an independent solver, truth separation.
3. MBD analytic score, schedule, quadratic/nonconvex tests and equal-budget solvers.
4. Sequential rollout, observation values, shared branches, matched no-information continuation.
5. Full coupled experiments, negative controls, diagnostics and held-out seeds.

Passing software tests alone does not establish a useful VoI effect or empirical safety.

## POAM adapter added after prototype validation

Primary source: Chen, Liu and Khardon, RSS 2024, https://www.roboticsproceedings.org/rss20/p019.pdf,
Sec. III-B attentive kernel; Sec. IV-B pivoted Cholesky inducing selection;
Eqs. (19)-(21) old-statistic transport; Eqs. (7)-(8) variational update;
Eq. (10) variational EM with minibatch kernel learning. JAX implementation is in
`wind/poam.py`. Supplied vendor's `GPyTorchModel.update_variational` was also
inspected. No Torch dependency is imported. Fixed observation noise is an explicit
restriction; kernel NN and amplitude adapt between planning calls. Full sparse
posterior retains conditional residual covariance. Finite-anchor planning
snapshots omit the residual between anchors and log the variance discrepancy.
Do not describe the default feature-GP missions as POAM missions.
