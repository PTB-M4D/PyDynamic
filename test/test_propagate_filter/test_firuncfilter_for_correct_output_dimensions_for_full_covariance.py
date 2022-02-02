import pytest
from hypothesis import given, settings
from numpy.testing import assert_equal

from PyDynamic import FIRuncFilter
from ..conftest import FIRuncFilter_input


@given(FIRuncFilter_input())
@settings(deadline=None)
@pytest.mark.slow
def test(fir_unc_filter_input):
    y_fir, Uy_fir = FIRuncFilter(**fir_unc_filter_input, return_full_covariance=True)
    assert_equal(len(fir_unc_filter_input["y"]), len(y_fir))
    assert_equal(Uy_fir.shape, (len(y_fir), len(y_fir)))
