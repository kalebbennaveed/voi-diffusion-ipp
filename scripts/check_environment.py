"""Print actual device placement and test JIT, Cholesky, grad, vmap, scan."""

import importlib.metadata as md
import json
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np


def workload(x):
    a = x @ x.T + jnp.eye(x.shape[0])
    L = jnp.linalg.cholesky(a)
    g = jax.grad(lambda z: jnp.sum(jnp.sin(z) ** 2))(L)

    def body(c, _):
        return c + 0.01 * jax.vmap(jnp.tanh)(g), None

    return jax.lax.scan(body, L, None, length=5)[0]


def main():
    report = {
        "architecture": platform.machine(),
        "system": platform.platform(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "versions": {p: md.version(p) for p in ("jax", "jaxlib", "numpy")},
        "devices": [str(d) for d in jax.devices()],
        "backend": jax.default_backend(),
    }
    f = jax.jit(workload)
    x = jnp.eye(32, dtype=jnp.float32)
    t = time.perf_counter()
    y = f(x).block_until_ready()
    report["compile_and_first_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    f(x).block_until_ready()
    report["steady_seconds"] = time.perf_counter() - t
    report["output_devices"] = [str(d) for d in y.devices()]
    report["representative_compute_pass"] = bool(np.isfinite(y).all())
    print(json.dumps(report, indent=2))
    if not report["representative_compute_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
