import numpy as np
import pytest
from hypothesis import given, settings

from PyDynamic import FIRuncFilter
from ..conftest import FIRuncFilter_input


@given(FIRuncFilter_input())
@settings(deadline=None)
@pytest.mark.slow
def test_FIRuncFilter_non_negative_main_diagonal_covariance(fir_unc_filter_input):
    _, Uy_fir = FIRuncFilter(**fir_unc_filter_input, return_full_covariance=True)
    assert np.all(np.diag(Uy_fir) >= 0)
