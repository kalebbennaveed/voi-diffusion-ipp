import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
from profile_io import load_profile,save_profile

def test_float64_checkpoint_survives_a_float32_callers_context(tmp_path):
    with jax.experimental.enable_x64():
        value=jnp.asarray([.123456789012345,1.000000000000003],dtype=jnp.float64)
    save_profile(tmp_path/'profile.pkl',{'value':value})
    with jax.experimental.disable_x64():restored=load_profile(tmp_path/'profile.pkl')['value']
    assert restored.dtype==np.float64
    np.testing.assert_array_equal(restored,value)
