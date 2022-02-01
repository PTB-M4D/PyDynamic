import pytest
from hypothesis import given, settings
from numpy.testing import assert_equal

from PyDynamic import FIRuncFilter
from ..conftest import FIRuncFilter_input


@given(FIRuncFilter_input())
@settings(deadline=None)
@pytest.mark.slow
def test_FIRuncFilter_for_correct_dimension_of_y(fir_unc_filter_input):
    y_fir = FIRuncFilter(**fir_unc_filter_input)[0]
    assert_equal(len(fir_unc_filter_input["y"]), len(y_fir))
