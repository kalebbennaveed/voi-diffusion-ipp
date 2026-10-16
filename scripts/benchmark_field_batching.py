"""Compare mathematically identical candidate batching for full POAM/RPCBF."""

import json
import resource
import time
from pathlib import Path

import jax
import numpy as np

from voi_diffusion_ipp.config import load_config
from voi_diffusion_ipp.experiments.scenarios import make_scenario
from voi_diffusion_ipp.planning.mpc import initial_policy
from voi_diffusion_ipp.planning.two_stage import make_buffers, objective

cfg = load_config("configs/field_poam.yaml")
env, belief, x, q = make_scenario(cfg)
y = initial_policy(x, q, belief, env, cfg)
buffers = make_buffers(belief, jax.random.PRNGKey(876), cfg)
vectors = y[None] + 0.2 * jax.random.normal(jax.random.PRNGKey(877), (4, y.size))
rows = []
reference = None
for chunk in [1, 2, 4]:
    fn = jax.jit(
        lambda ys, chunk=chunk: jax.lax.map(
            lambda y: objective(y, x, q, belief, buffers, env, cfg),
            ys,
            batch_size=None if chunk == 1 else chunk,
        )
    )
    start = time.perf_counter()
    executable = fn.lower(vectors).compile()
    compile_s = time.perf_counter() - start
    executable(vectors).block_until_ready()
    start = time.perf_counter()
    result = np.asarray(executable(vectors).block_until_ready())
    seconds = time.perf_counter() - start
    if reference is None:
        reference = result
    np.testing.assert_allclose(result, reference, rtol=5e-5, atol=5e-5)
    row = {
        "chunk": chunk,
        "compile_seconds": compile_s,
        "seconds_per_candidate": seconds / len(vectors),
        "values": result.tolist(),
        "peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    print(json.dumps(row), flush=True)
    rows.append(row)
    Path("docs/field_poam_validation/batching.json").write_text(json.dumps(rows, indent=2) + "\n")
