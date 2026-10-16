"""Compile versus steady execution of the actual GP/spline/RPCBF two-stage kernel."""

import argparse
import json
import resource
import sys
import time
from dataclasses import replace

import jax
import jax.numpy as jnp

from voi_diffusion_ipp.config import Config
from voi_diffusion_ipp.experiments.scenarios import make_scenario
from voi_diffusion_ipp.planning.two_stage import make_buffers, objective

p = argparse.ArgumentParser()
p.add_argument("--output", default="results/backend_benchmark.json")
a = p.parse_args()
cfg = replace(Config(), stage_steps=4, future_steps=5, backup_steps=6, inner_samples=3, worlds=2)
env, b, x, q = make_scenario(cfg)
buf = make_buffers(b, jax.random.PRNGKey(123), cfg)
y = jnp.zeros((1 + cfg.branches) * cfg.knots * 2)
f = jax.jit(lambda yy: objective(yy, x, q, b, buf, env, cfg))
t = time.perf_counter()
compiled = f.lower(y).compile()
compile_s = time.perf_counter() - t
values = []
times = []
for _ in range(4):
    t = time.perf_counter()
    v = compiled(y).block_until_ready()
    times.append(time.perf_counter() - t)
    values.append(float(v))
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
r = {
    "backend": jax.default_backend(),
    "devices": [str(d) for d in jax.devices()],
    "jax": jax.__version__,
    "output_devices": [str(d) for d in v.devices()],
    "compile_seconds": compile_s,
    "execution_seconds": times,
    "costs": values,
    "peak_process_bytes": peak if sys.platform == "darwin" else peak * 1024,
}
from pathlib import Path

Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(json.dumps(r, indent=2))
print(json.dumps(r, indent=2))
