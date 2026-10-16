# Implementation checkpoints

## Phases 0-1

Native ARM64 macOS 15.7.4, Xcode command-line tools present. UV 0.10.8,
Python 3.11.15, JAX/jaxlib 0.4.38 CPU. Environment workload passes JIT,
Cholesky, grad, vmap, scan; four physical/GP tests pass. See generated
`results/environment_cpu.json`. Metal assessment is pending separately.

The finite-feature GP is exact for its explicitly declared kernel; it is not POAM.
Next: spline RPCBF and constrained QP, then diffusion and sequential integration.
