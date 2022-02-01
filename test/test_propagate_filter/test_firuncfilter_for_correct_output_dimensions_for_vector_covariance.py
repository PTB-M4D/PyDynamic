import pytest
from hypothesis import given, settings
from numpy.testing import assert_equal

from PyDynamic import FIRuncFilter
from ..conftest import FIRuncFilter_input


@given(FIRuncFilter_input())
@settings(deadline=None)
@pytest.mark.slow
def test(fir_unc_filter_input):
    _, Uy = FIRuncFilter(**fir_unc_filter_input)
    assert_equal(Uy.shape, (len(fir_unc_filter_input["y"]),))
