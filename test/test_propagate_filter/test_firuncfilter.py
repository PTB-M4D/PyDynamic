import pytest
from hypothesis import given, settings

from PyDynamic import FIRuncFilter
from ..conftest import FIRuncFilter_input


@given(FIRuncFilter_input())
@settings(deadline=None)
@pytest.mark.slow
def test(fir_unc_filter_input):
    # Check expected output for thinkable permutations of input parameters.
    y, Uy = FIRuncFilter(**fir_unc_filter_input)
    assert len(y) == len(fir_unc_filter_input["y"])
    assert len(Uy) == len(fir_unc_filter_input["y"])

    # note: a direct comparison against scipy.signal.lfilter is not needed,
    #       as y is already computed using this method
