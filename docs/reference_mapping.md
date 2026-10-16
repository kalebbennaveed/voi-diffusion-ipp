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

## Historical POAM adapter after prototype validation (superseded below)

Primary source: Chen, Liu and Khardon, RSS 2024, https://www.roboticsproceedings.org/rss20/p019.pdf,
Sec. III-B attentive kernel; Sec. IV-B pivoted Cholesky inducing selection;
Eqs. (19)-(21) old-statistic transport; Eqs. (7)-(8) variational update;
Eq. (10) variational EM with minibatch kernel learning. JAX implementation is in
`wind/poam.py`. Supplied vendor's `GPyTorchModel.update_variational` was also
inspected. No Torch dependency is imported. Fixed observation noise is an explicit
restriction; kernel NN and amplitude adapt between planning calls. Full sparse
posterior retained conditional residual covariance, but its historical finite-anchor
planning snapshots omitted the residual between anchors. That projection has now
been removed. Historical feature-GP missions must not be described as POAM missions;
the current default uses the full-field POAM profile described below.
# Interactive demo reference

The added read-only `mbd-playground` checkout was inspected via its README,
`web/safe-mbd/solver-core.js` history/summary routines, and
`web/dmbd/playback.js`. It informs the UI separation between denoising and
execution. Its browser Safe-MPD solver/shield is not substituted for this
repository's MBD or RPCBF. No code or assets were copied. See
[interactive demo provenance](interactive_demo.md).

## Full-field POAM revision (October 16, 2026)

The full-field profile (`configs/field_poam.yaml`) uses the complete sparse POAM
predictive distribution (Chen et al., RSS 2024, Eq. 4), including the conditional
residual `Kxx - Kxz Kzz^-1 Kzx`. It no longer projects the distribution onto a
finite-anchor wind model. Both prediction and hypothetical POAM updates preserve
the learned attentive kernel. Real updates relocate inducing points by pivoted
Cholesky, transport unweighted statistics (Eqs. 19–21), update `q(u)` (7–8), and
train kernel hyperparameters by variational EM (10). Adam moments persist across
updates. Observation variance is known from the simulated sensor and held fixed.

The continuous sample-path implementation uses Wilson et al., *Efficiently
sampling functions from Gaussian process posteriors*, ICML 2020,
https://proceedings.mlr.press/v119/wilson20a.html. It retains a sampled conditional
residual using `g(x) + Kxz Kzz^-1 (u-g(z))`, with independent `u~q(u)` and prior
`g`. Sobol spectral quadrature approximates prior draws; the inference kernel and
predictive covariance are not replaced by Fourier features. This is a numerical
sampling approximation, not an exact continuous GP draw. Its covariance error
is measured against the complete model, including after online learning.

`tests/test_poam.py` checks full residual covariance, independent matrix identities,
frozen-kernel real versus hypothetical POAM updates, empirical sample moments,
spatial coherence, derivatives, and equivalent contracted field evaluation.
`tests/test_field_monitoring.py` checks that all cells contribute to the objective,
that clarity increases near sensing and decays away from it, that temporal
integration converges, and that held-out physics does not depend on estimator
inducing points or monitoring-grid resolution.

The accepted sampler additionally conditions its prior paths on a full-kernel
joint draw at 667 dense support points, retaining a spectral residual between
them. This reduces post-learning covariance error from 12–17% for spectral-only
paths to 0.22–0.27% in the three survey diagnostics. Dense support is a numerical
sampling device; it neither replaces POAM inducing inputs nor projects away its
predictive residual. Inference continues to evaluate the full kernel.
