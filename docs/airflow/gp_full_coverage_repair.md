# GP-only full-coverage test and spatial POAM repair

2026-11-04. The user requested unrestricted access to all parts of the
environment to separate GP failure from robot motion and safety restrictions.
This experiment contains **no robot trajectory, MBD, or RPCBF**. All previous
robot recordings and webpage assets are preserved.

## Controlled test

Both unchanged stationary CFD fields are sampled at all 2,361 free grid-cell
centers. A geometry-only random permutation determines acquisition order.
Every measurement includes independent 0.1 m/s Gaussian noise per wind
component. The GP receives positions and these noisy vectors, not true values
or a time coordinate. All kernel fitting uses only the supplied measurements.

Evaluation compares posterior means with truth at those cell centers and at
1,000 off-grid positions that are not used for fitting. Off-grid truth is the
same physical field interpolator used by the robot simulation. Those points
were used for configuration validation, not an untouched benchmark dataset.
A second run changes measurement noise, acquisition order, and training
minibatches while retaining the same environment and validation locations.

This tests statistical mapping ability when coverage is supplied. It does not
demonstrate a dynamically feasible survey or 300-second monitoring mission.
At the previous 5 Hz sensor rate, acquiring 2,361 measurements would require
at least 472.2 seconds, before accounting for physical travel restrictions.
Here the horizontal axis is **measurement count**, not mission time.

## Failure reproduced without safety or motion constraints

| Model, same noisy data | S1 grid RMSE | S2 grid RMSE | S1 off-grid RMSE | S2 off-grid RMSE |
|---|---:|---:|---:|---:|
| Original online POAM, 128 inducing points | 0.7605 | 0.9820 | 0.7037 | 0.8340 |
| Batch sparse fit, original final kernel/inducing points | 0.6761 | 0.8641 | 0.6092 | 0.7485 |
| Full GP, original final attentive kernel | 0.2964 | 0.3608 | 0.2275 | 0.2714 |
| Full GP, RBF lengthscale 0.3 m | 0.1270 | 0.1411 | 0.1250 | 0.1530 |
| Full GP, RBF lengthscale 0.125 m | 0.1377 | 0.1353 | 0.1667 | 0.1928 |

All errors are vector RMSE in m/s, not component RMSE. Full-GP/RBF fits are
offline diagnostic controls; they have not replaced POAM in the method.

These results rule out inadequate spatial coverage or RPCBF interference as
the sole explanation. They also separate errors introduced by online
compression, the sparse representation, and the learned kernel.

The original model's average squared attention weights at completion were:

| Scenario | 0.3 m | 0.8 m | 1.3 m | 1.8 m |
|---|---:|---:|---:|---:|
| 1 | 1.15% | 2.83% | 5.05% | 90.97% |
| 2 | 2.17% | 11.66% | 4.78% | 81.40% |

The kernel became dominated by a long spatial scale even though the actual
field contains narrow jets and wakes. A dense GP with that same learned kernel
still fits poorly, while recomputing sparse statistics alone also fails to
repair it. Independent capacity sweeps show that 128 inducing points lose
substantial detail even before kernel learning. The failure is a combination
of insufficient spatial representation and oversmoothing during adaptation,
not a time-dependent wind or a missing full-posterior residual term.

This behavior is consistent with the coupling between sparse representation
and lengthscale learning discussed in the [POAM paper, section IV-A](https://arxiv.org/html/2406.03669v1).
The paper's benchmark configuration in the local author repository uses 500
inducing points and batches of 256, whereas the failing adapter used 128 and 32.
Its attention network also received raw coordinates instead of scaled inputs.
Those reference settings are not a guarantee that 500 points suffice here.

## Implemented correction

`scripts/airflow/spatial_poam.py` provides an explicit spatial inference profile:

| Setting | Original experiment | Repaired diagnostic |
|---|---|---|
| Inducing points | 128 | 768 |
| Attentive-kernel lengthscales, m | 0.3, 0.8, 1.3, 1.8 | 0.2, 0.5, 1.0, 2.0 |
| Attention-network coordinates | Raw meters | Geometry scaled to [-1,1] in each axis |
| Kernel distance calculation | Meters | Meters, unchanged |
| Hyperparameter minibatch | 32, repeated for four steps | Four independent batches of 256 |
| Attention-network Adam rate | 0.001 | 0.001 |
| Amplitude Adam rate | 0.001 | 0.01 |
| Observation batch | One per update | 50 per diagnostic update |
| Prior component std / sensor std | 1.8 / 0.1 m/s | 1.8 / 0.1 m/s, unchanged |

The inducing selection, transported unweighted statistics, variational
posterior, attentive kernel, and conditional residual covariance remain POAM.
This increases resolution rather than replacing the estimator with RBF or
inflating its sensor noise. The normalization constants are fixed geometry
and have zero training gradient. No temporal kernel, wind decay, or additional
exploration objective is introduced.

The settings were selected using these controlled simulator diagnostics; they
are not claimed as universally appropriate for other wind fields. They are
explicitly separate from the preserved robot experiments. The old profiles
and two-dimensional repeated-minibatch call remain reproducible.

## Repaired results

| Run | S1 grid RMSE | S2 grid RMSE | S1 off-grid RMSE | S2 off-grid RMSE |
|---|---:|---:|---:|---:|
| First acquisition order/noise seed | 0.1400 | 0.1483 | 0.1414 | 0.1640 |
| Independent order/noise repetition | 0.1426 | 0.1545 | 0.1358 | 0.1722 |

All four scenario/seed combinations pass the 0.2 m/s whole-map accuracy
requirement in this unrestricted test. Final maximum pointwise vector errors
are still around 1 m/s: an RMSE threshold is not a uniform error bound.

The original 500-point correction was retained as an intermediate result:
normalized attention and faster amplitude learning gave S2 RMSE 0.2013 m/s,
but latent three-sigma coverage was only 65.5%. A different 500-point profile
with a 0.125 m component gave 0.2427/0.3316 m/s. These outcomes are saved;
neither was promoted as a successful repair.

![RMSE versus actual measurement count](../../output/gp_repair_20261104/full-coverage-rmse.png)

![True, original and repaired wind maps](../../output/gp_repair_20261104/full-coverage-maps.png)

## Uncertainty and runtime limits

Latent posterior three-sigma disk coverage improves from 50.4%/38.2% on the
original fully measured grid to 92.2%/92.9% in the first repaired run. The
second run gives 93.4%/90.6%. Off-grid coverage is approximately 90–94%.
This remains below the 98.89% pointwise coverage of an ideally calibrated
isotropic 2D three-sigma Gaussian disk. Thus mean accuracy is repaired for this
exercise, but uncertainty is not fully calibrated. No robust safety guarantee
or successful RPCBF mission follows from this test.

The repaired 768-point batches took median 0.532/0.553 seconds, with observed
maxima 0.737/0.583 seconds, including conditioning and four EM steps and
excluding compilation and the last differently shaped batch. Those figures
refer to one **50-measurement GP update**, not an MBD trajectory. The two-scenario
test took 57.2 seconds including compilation, diagnostics and output. The
independent repetition took 57.1 seconds. Total MBD planning plus this larger
GP update has not been shown to meet the existing one-second budget.

The repaired sampling/cache accuracy must also be checked before restoring
RPCBF or MBD. The existing 32-node spectral sampler and 0.2 m field cache were
not used in the GP-only prediction test. They must not be assumed accurate
for the revised posterior merely because its direct predictions improve.

## Checks and reproduction

21 selected regression tests passed. The new checks verify physical-distance
units after attention normalization, invariance under translating the whole
workspace, fixed normalization during kernel training, learning from supplied
measurements, and compatibility with the old repeated-minibatch training path.
Existing checks cover POAM updates and full covariance, spatial-only inputs,
and retained robot/safety APIs. Tests passing do not establish calibration.

Run from the `voi-diffusion-ipp` repository, with a fresh output directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/airflow/diagnose_full_coverage_gp.py \
  --output output/gp_repair_new \
  --inducing 768 --normalize-attention --amplitude-rate .01 \
  --independent-batches --batch-size 256 --observation-batch 50 \
  --lengths .2,.5,1.,2. --skip-comparators
```

Use `--seed-offset 10000` for the saved independent repetition. Omit the repair
flags to reproduce the original profile and full-GP controls. Each invocation
has a 180-second wall-time limit and refuses to overwrite an existing output
directory. Parameters, measurement sets, acquisition orders, predictions and
variance maps are saved under `output/gp_repair_20261104/`.
