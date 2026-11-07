import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/airflow'))
from uncertainty_handoff import posterior_vector_rms,ready

def test_mask_excludes_irrelevant_uncertainty_without_changing_posterior():
    variance=np.array([.04,.04,3.24]);before=variance.copy()
    assert ready(variance,[1,1,0],.3)
    assert not ready(variance,[1,1,1],.3)
    np.testing.assert_array_equal(variance,before)

def test_empty_domain_is_not_a_free_pass():
    assert posterior_vector_rms([1,2],[0,0]) is None
    assert not ready([1,2],[0,0],1.1)

def test_scale_and_two_vector_components():
    assert posterior_vector_rms([.5,.5],[1,1])==1.
    assert posterior_vector_rms([.5,.5],[7,7])==1.

@pytest.mark.parametrize('variance,weights', [([np.nan],[1]),([1],[-1]),([-1],[1]),([1,2],[1])])
def test_invalid_inputs_rejected(variance,weights):
    with pytest.raises(ValueError):posterior_vector_rms(variance,weights)
