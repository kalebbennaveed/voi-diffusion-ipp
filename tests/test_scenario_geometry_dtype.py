"""Continuous geometry must compile when JSON coordinates use integer literals."""
import sys
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/airflow'))
import cluster_environment


def test_single_obstacle_integer_bounds_compile_in_planner_precision():
    cfg=SimpleNamespace(region_target=.7,background_target=.35,
                        clarity_decay=.16,background_decay=.02)
    env,_,_=cluster_environment.environment(cfg,ROOT/'configs/scenarios/calm_monitoring_20261106.json')
    assert np.issubdtype(env.bounds.dtype,np.floating)
    env=jax.tree.map(lambda z:z.astype(jnp.float32) if z is not None else None,env)
    with jax.experimental.disable_x64():
        fn=jax.jit(env.clearance)
        x=jnp.asarray([1.4,4.15,0.,0.],dtype=jnp.float32)
        compiled=fn.lower(x).compile()
        np.testing.assert_allclose(compiled(x),1.4,rtol=1e-6)
